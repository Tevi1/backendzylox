"""Adaptive ask router with intent-aware formatting + filter safety."""
from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_config
from ..gemini_client import RAG_MODEL, ask_with_file_search, plan_filters
from ..models import Document, Workspace
from ..prompts.system_answer_style import SYSTEM_ANSWER_STYLE
from ..prompts.retrieval_planner import FILTER_PLANNER_PROMPT

ALLOWED_FIELDS = {"company", "doc_type", "time_scope", "sensitivity"}
FILE_EXT_HINTS = (".pdf", ".docx", ".pptx", ".txt", ".md", ".rtf", ".json", ".csv", ".xlsx")
CONTRACT_Q = re.compile(r"\b(obligation|clause|liabilit|indemn|sla|termination)\b", re.IGNORECASE)

# Complex question triggers for enhanced retrieval
COMPLEX_TRIGGERS = [
    "MSA", "clause", "promise", "obligation", "risk", "architecture",
    "security", "roadmap", "pricing", "market", "contradiction",
    "versus", "vs", "compare", "reality", "fall short", "gap"
]

# Doc-type boosting rules for question patterns
DOC_TYPE_BOOSTS = {
    "legal_promise": {
        "triggers": ["MSA", "clause", "contract", "promise", "liability", "obligation", "breach", "reality"],
        "boost_doc_types": ["contract", "security", "roadmap", "memo"],
        "top_k": 12,
    },
    "pricing": {
        "triggers": ["pricing", "ACV", "£60k", "tier", "plan", "cost", "margin"],
        "boost_doc_types": ["deck", "memo"],
        "top_k": 10,
    },
    "risk_register": {
        "triggers": ["risk register", "risk", "risks", "mitigation"],
        "boost_doc_types": ["contract", "security", "roadmap", "memo", "deck"],  # All doc types
        "top_k": 12,
        "strategy": "multi_doc",  # Force retrieval from all available doc_types
    },
    "comparison": {
        "triggers": ["compare", "versus", "vs", "promises vs reality", "what we say vs what we do"],
        "boost_doc_types": ["contract", "security", "roadmap", "memo"],
        "top_k": 12,
        "strategy": "multi_doc",  # Force multi-doc retrieval
    },
    "infrastructure": {
        "triggers": ["architecture", "infra", "implementation", "KMS", "VPC", "security"],
        "boost_doc_types": ["security", "roadmap"],
        "top_k": 10,
    },
    "business": {
        "triggers": ["market", "positioning", "GTM", "strategy"],
        "boost_doc_types": ["deck", "memo", "roadmap"],
        "top_k": 10,
    },
}

_config = get_config()
DEV_MODE = _config.is_dev


def _is_real_value(value: Optional[str]) -> bool:
    if not value:
        return False
    lowered = value.lower()
    if lowered in {"unknown", "misc", "documents", "document"}:
        return False
    if "/" in value or "\\" in value:
        return False
    if lowered.endswith(FILE_EXT_HINTS):
        return False
    return True


def _filter_real_values(values: List[str]) -> List[str]:
    return [value for value in values if _is_real_value(value)]


def _detect_company_in_question(question: str, companies: List[str]) -> Optional[str]:
    lowered = (question or "").lower()
    for company in companies:
        if company and company.lower() in lowered:
            return company
    return None


def _has_contract_intent(question: str) -> bool:
    return bool(CONTRACT_Q.search(question or ""))


def is_comparison_question(question: str) -> bool:
    """
    Detect if question requires cross-document comparison.
    
    Examples: "compare promises vs reality", "list contradictions", "build risk register"
    """
    question_lower = (question or "").lower()
    comparison_patterns = [
        r"\bcompare\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bpromise.*reality\b",
        r"\breality.*promise\b",
        r"\bcontradiction\b",
        r"\brisk\s+register\b",
        r"\bwhere.*fall\s+short\b",
        r"\bgap\b",
        r"\balignment\b",
    ]
    return any(re.search(pattern, question_lower, re.IGNORECASE) for pattern in comparison_patterns)


def get_boosted_doc_types(question: str) -> List[str]:
    """
    Return doc_types to boost based on question content.
    
    Returns list of doc_types that should be prioritized for this question.
    """
    question_lower = (question or "").lower()
    boosted = set()
    
    for boost_rule in DOC_TYPE_BOOSTS.values():
        triggers = boost_rule["triggers"]
        if any(trigger.lower() in question_lower for trigger in triggers):
            boosted.update(boost_rule["boost_doc_types"])
    
    return list(boosted) if boosted else []


def get_retrieval_strategy(question: str) -> Dict[str, any]:
    """
    Return retrieval strategy (doc_types, top_k, strategy) based on question content.
    
    Returns dict with:
    - boost_doc_types: List of doc_types to prioritize
    - top_k: Suggested number of chunks (if applicable)
    - strategy: "multi_doc" if should fetch from all doc_types, None otherwise
    """
    question_lower = (question or "").lower()
    
    # Check each boost rule
    for rule_name, boost_rule in DOC_TYPE_BOOSTS.items():
        triggers = boost_rule["triggers"]
        if any(trigger.lower() in question_lower for trigger in triggers):
            return {
                "boost_doc_types": boost_rule["boost_doc_types"],
                "top_k": boost_rule.get("top_k", 8),
                "strategy": boost_rule.get("strategy"),
            }
    
    # Default
    return {
        "boost_doc_types": [],
        "top_k": 8,
        "strategy": None,
    }


def _has_context(grounding: Optional[dict]) -> bool:
    if not grounding:
        return False
    return bool(
        grounding.get("retrieved_contexts")
        or grounding.get("retrievedContexts")
        or grounding.get("grounding_chunks")
        or grounding.get("citations")
    )


async def _distinct_values(session: AsyncSession, workspace_id: str) -> Dict[str, List[str]]:
    async def fetch(column) -> List[str]:
        stmt: Select = (
            select(column)
            .where(
                Document.workspace_id == workspace_id,
                column.is_not(None),
                column != "",
            )
            .distinct()
        )
        result = await session.execute(stmt)
        return _filter_real_values([row[0] for row in result if row[0]])

    return {
        "company": await fetch(Document.company),
        "doc_type": await fetch(Document.doc_type),
        "time_scope": await fetch(Document.time_scope),
    }


def _sanitize_and_whitelist(
    filter_str: Optional[str],
    diagnostics: Dict[str, List[str]],
) -> tuple[Optional[str], List[str]]:
    if not filter_str:
        return None, []

    tokens = re.split(r"\s+(AND|OR)\s+", filter_str.strip())
    warnings: List[str] = []
    result_tokens: List[str] = []
    expect_connector = False

    equality_re = re.compile(
        r"(\w+)\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9_\-\.]+))",
        re.IGNORECASE,
    )

    for token in tokens:
        stripped = token.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper in {"AND", "OR"}:
            if expect_connector:
                result_tokens.append(upper)
                expect_connector = False
            continue

        match = equality_re.fullmatch(stripped)
        if not match:
            warnings.append(f"Ignored filter fragment: {stripped}")
            continue

        field = match.group(1).lower()
        value = match.group(2) or match.group(3) or match.group(4) or ""
        value = value.strip().strip('"').strip("'")

        if field not in ALLOWED_FIELDS:
            warnings.append(f"Dropped unsupported field {field}=\"{value}\"")
            continue

        allowed_values = diagnostics.get(field)
        if allowed_values:
            if value not in allowed_values:
                warnings.append(f'Dropped filter {field}="{value}" (not present in workspace metadata)')
                continue

        result_tokens.append(f'{field}="{value}"')
        expect_connector = True

    if expect_connector is False and result_tokens and result_tokens[-1] in {"AND", "OR"}:
        result_tokens.pop()

    if not result_tokens:
        return None, warnings

    sanitized = []
    prev_connector = False
    for token in result_tokens:
        if token in {"AND", "OR"}:
            if not prev_connector:
                sanitized.append(token)
                prev_connector = True
            continue
        sanitized.append(token)
        prev_connector = False

    if sanitized and sanitized[-1] in {"AND", "OR"}:
        sanitized.pop()

    final_filter = " ".join(sanitized) if sanitized else None
    return final_filter, warnings


def _build_suggestions(distinct: Dict[str, List[str]]) -> List[str]:
    suggestions: List[str] = []
    for key in ("doc_type", "company", "time_scope"):
        for value in distinct.get(key, []):
            suggestion = f'{key}="{value}"'
            if suggestion not in suggestions:
                suggestions.append(suggestion)
    return suggestions


def _build_filter(company: Optional[str], doc_types: Optional[List[str]], time_scope: Optional[str]) -> Optional[str]:
    parts = []
    if company:
        parts.append(f'company="{company}"')
    if doc_types:
        ors = " OR ".join(f'doc_type="{d}"' for d in doc_types if d)
        if ors:
            parts.append(f"({ors})")
    if time_scope:
        parts.append(f'time_scope="{time_scope}"')
    return " AND ".join(parts) if parts else None


async def ask_router(
    session: AsyncSession,
    workspace: Workspace,
    question: str,
    user_filter: Optional[str],
    system_vars: Optional[dict] = None,
) -> Dict[str, object]:
    diagnostics = await _distinct_values(session, workspace.id)
    safe_filter, warnings = _sanitize_and_whitelist(user_filter, diagnostics)

    plan = await asyncio.to_thread(
        plan_filters,
        question,
        diagnostics,
        RAG_MODEL,
        FILTER_PLANNER_PROMPT,
    )

    planner_filter = _build_filter(
        plan.get("company"),
        plan.get("doc_types"),
        plan.get("time_scope"),
    )

    temperature = plan.get("temperature", 0.2) or 0.2

    contract_intent = _has_contract_intent(question)
    is_comparison = is_comparison_question(question)
    boosted_doc_types = get_boosted_doc_types(question)

    attempts: List[tuple[str, Optional[str]]] = []
    
    # For comparison questions, ensure we get multiple doc_types
    if is_comparison and not safe_filter:
        # Build filter that includes multiple doc_types
        available_doc_types = diagnostics.get("doc_type", [])
        relevant_types = [dt for dt in boosted_doc_types if dt in available_doc_types]
        if not relevant_types:
            # Fallback: try to get contract, security, roadmap if available
            for preferred in ["contract", "security", "roadmap"]:
                if preferred in available_doc_types:
                    relevant_types.append(preferred)
                    if len(relevant_types) >= 3:
                        break
        
        if relevant_types:
            comparison_filter = " OR ".join(f'doc_type="{dt}"' for dt in relevant_types[:3])
            attempts.append(("comparison_multi_doc", comparison_filter))
    if not safe_filter and contract_intent:
        attempts.append(("intent_contract", 'doc_type="contract"'))
    # Add boosted doc_types filter if available
    if boosted_doc_types and not safe_filter:
        available_doc_types = diagnostics.get("doc_type", [])
        relevant_boosted = [dt for dt in boosted_doc_types if dt in available_doc_types]
        if relevant_boosted:
            boost_filter = " OR ".join(f'doc_type="{dt}"' for dt in relevant_boosted)
            attempts.append(("boosted_doc_types", boost_filter))
    if safe_filter:
        attempts.append(("user_filter", safe_filter))
    if planner_filter:
        attempts.append(("planner", planner_filter))
    attempts.append(("no_filter", None))

    relaxed_attempts = [
        (plan.get("company"), plan.get("doc_types"), None),
        (plan.get("company"), None, None),
        (None, plan.get("doc_types"), None),
        (None, None, None),
    ]

    tried: set[str] = set()
    response: Dict[str, object] | None = None

    for scope, filt in attempts:
        key = f"{scope}:{filt}"
        if key in tried:
            continue
        tried.add(key)
        answer, grounding = await asyncio.to_thread(
            ask_with_file_search,
            workspace.file_search_store_name,
            question,
            filt,
            SYSTEM_ANSWER_STYLE,
            model=RAG_MODEL,
            temperature=temperature,
        )
        response = {
            "answer": answer,
            "metadata_filter": filt,
            "grounding_metadata": grounding,
            "scope": scope,
            "plan": plan,
        }
        if _has_context(grounding):
            break
    else:
        response = None

    if not response or not _has_context(response.get("grounding_metadata")):
        for comp, docs, tscope in relaxed_attempts:
            filt = _build_filter(comp, docs, tscope)
            key = f"relaxed:{filt}"
            if key in tried:
                continue
            tried.add(key)
            answer, grounding = await asyncio.to_thread(
                ask_with_file_search,
                workspace.file_search_store_name,
                question,
                filt,
                SYSTEM_ANSWER_STYLE,
                model=RAG_MODEL,
                temperature=temperature,
            )
            if _has_context(grounding):
                response = {
                    "answer": answer,
                    "metadata_filter": filt,
                    "grounding_metadata": grounding,
                    "scope": "relaxed",
                    "plan": plan,
                }
                if DEV_MODE:
                    response["relaxed"] = True
                break

    if not response or not _has_context(response.get("grounding_metadata")):
        suggestions = _build_suggestions(diagnostics)
        response = {
            "answer": "Not enough evidence in the provided documents.",
            "metadata_filter": None,
            "grounding_metadata": None,
            "scope": "no_evidence",
            "suggested_filters": suggestions[:8],
            "plan": plan,
        }
        if safe_filter:
            warnings.append("Filter returned no contexts; retried without filter.")

    if warnings and DEV_MODE:
        response["warnings"] = warnings

    return response

