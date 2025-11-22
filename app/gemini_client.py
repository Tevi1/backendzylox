"""Google Gemini helpers for File Search + classification + RAG."""
from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from .config import get_config
from .prompts.system_prompt import build_system_instruction
from .prompts.system_answer_style import SYSTEM_ANSWER_STYLE
from .prompts.retrieval_planner import FILTER_PLANNER_PROMPT

# Get model names from config (allows override via environment variables)
_config = get_config()
CLASSIFICATION_MODEL = _config.CLASSIFICATION_MODEL
RAG_MODEL = _config.RAG_MODEL


class GeminiRAGError(RuntimeError):
    """Raised when Gemini RAG operations fail."""


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    """Get cached Gemini client instance."""
    config = get_config()
    api_key = config.get_gemini_api_key()
    if not api_key:
        raise GeminiRAGError("GEMINI_API_KEY environment variable is required.")
    return genai.Client(api_key=api_key)


def create_file_search_store(display_name: str) -> types.FileSearchStore:
    """
    Create a new Gemini File Search store.

    Args:
        display_name: Human-readable name for the store.

    Returns:
        FileSearchStore instance with store name.

    Raises:
        GeminiRAGError: If store creation fails.
    """
    try:
        return _client().file_search_stores.create(config={"display_name": display_name})
    except Exception as exc:
        raise GeminiRAGError(f"Failed to create File Search store: {exc}") from exc


def upload_file_to_store(store_name: str, tmp_path: str, cfg: Dict[str, object]) -> Dict[str, object]:
    """Upload and poll until indexing completes."""

    client = _client()
    operation = client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=store_name,
        file=tmp_path,
        config=cfg,
    )

    # Poll operation until complete to ensure documents are queryable.
    while not getattr(operation, "done", False):
        time.sleep(2)
        operation = client.operations.get(operation)

    return getattr(operation, "response", {}) or {}


def classify_document(company_candidates, path: str, text_preview: str) -> str:
    prompt = (
        "You are an ingestion agent. Classify this document and return JSON only with keys "
        "company, doc_type, time_scope, sensitivity, summary.\n"
        f"Candidates: {company_candidates}\nPath: {path}\nContent: {text_preview[:3000]}"
    )
    response = _client().models.generate_content(
        model=CLASSIFICATION_MODEL,
        contents=[prompt],
    )
    return getattr(response, "text", "") or "{}"


def _grounding_dict(candidate: Optional[types.Candidate]) -> Optional[dict]:
    grounding = getattr(candidate, "grounding_metadata", None)
    if grounding is None and hasattr(candidate, "model_dump"):
        data = candidate.model_dump()
        grounding = data.get("grounding_metadata") or data.get("groundingMetadata")
    if grounding is None:
        return None
    if hasattr(grounding, "model_dump"):
        return grounding.model_dump()
    if isinstance(grounding, dict):
        return grounding
    return None


def plan_filters(
    question: str,
    distinct_values: Dict[str, List[str]],
    model: str = RAG_MODEL,
    prompt: str = FILTER_PLANNER_PROMPT,
) -> dict:
    """
    Run the planner pass (no tools) to get JSON filter suggestions.

    Args:
        question: User question to analyze.
        distinct_values: Dict with keys 'company', 'doc_type', 'time_scope' and lists of available values.
        model: Gemini model name (default: gemini-2.5-pro).
        prompt: System prompt for filter planning.

    Returns:
        Dict with suggested filters: company, doc_types, time_scope, temperature, etc.

    Raises:
        GeminiRAGError: If API call fails or response cannot be parsed.
    """

    cfg = types.GenerateContentConfig(
        system_instruction=prompt,
        response_mime_type="application/json",
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        max_output_tokens=1024,
    )
    seeds = {
        "question": question,
        "known_companies": distinct_values.get("company", []),
        "known_doc_types": distinct_values.get("doc_type", []),
        "known_time_scopes": distinct_values.get("time_scope", []),
    }
    try:
        response = _client().models.generate_content(
            model=model,
            contents=json.dumps(seeds),
            config=cfg,
        )
        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise GeminiRAGError(f"Failed to parse filter plan JSON: {exc}") from exc
    except Exception as exc:
        raise GeminiRAGError(f"Filter planning API call failed: {exc}") from exc


def ask_with_file_search(
    store_name: str,
    question: str,
    metadata_filter: Optional[str],
    system_instruction: str = SYSTEM_ANSWER_STYLE,
    *,
    model: str = RAG_MODEL,
    temperature: float = 0.2,
) -> Tuple[str, Optional[dict]]:
    """
    Call Gemini with File Search tool and return answer + grounding metadata.

    Args:
        store_name: Gemini File Search store name to query.
        question: User question to answer.
        metadata_filter: Optional metadata filter string (e.g., 'company="Acme" AND doc_type="contract"').
        system_instruction: System prompt for answer formatting.
        model: Gemini model name (default: gemini-2.5-pro).
        temperature: Generation temperature (default: 0.2).

    Returns:
        Tuple of (answer_text, grounding_metadata_dict).

    Raises:
        GeminiRAGError: If API call fails or response is invalid.
    """

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store_name],
                    metadata_filter=metadata_filter or "",
                )
            )
        ],
        temperature=float(temperature),
        top_p=0.9,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    try:
        response = _client().models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part(text=question)])],
            config=config,
        )
    except Exception as exc:
        raise GeminiRAGError(f"Gemini File Search API call failed: {exc}") from exc

    candidate = response.candidates[0] if getattr(response, "candidates", None) else None
    grounding_dict = _grounding_dict(candidate)

    answer_text = getattr(response, "text", None)
    if not answer_text and candidate and candidate.content:
        parts = getattr(candidate.content, "parts", []) or []
        answer_text = "\n".join(part.text for part in parts if getattr(part, "text", None))

    return answer_text or "", grounding_dict
