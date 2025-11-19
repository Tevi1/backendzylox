"""Google Gemini helpers for File Search + classification + RAG."""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from .prompts.system_prompt import build_system_instruction
from .prompts.system_answer_style import SYSTEM_ANSWER_STYLE
from .prompts.retrieval_planner import FILTER_PLANNER_PROMPT

CLASSIFICATION_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
RAG_MODEL = "gemini-2.5-pro"


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is required.")
    return genai.Client(api_key=api_key)


def create_file_search_store(display_name: str) -> types.FileSearchStore:
    return _client().file_search_stores.create(config={"display_name": display_name})


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
    """Run the planner pass (no tools) to get JSON filter suggestions."""

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
    response = _client().models.generate_content(
        model=model,
        contents=json.dumps(seeds),
        config=cfg,
    )
    try:
        return json.loads(response.text)
    except Exception:
        return {
            "company": None,
            "doc_types": None,
            "time_scope": None,
            "strict_quote": False,
            "answer_shape": "sections",
            "temperature": 0.2,
        }


def ask_with_file_search(
    store_name: str,
    question: str,
    metadata_filter: Optional[str],
    system_instruction: str = SYSTEM_ANSWER_STYLE,
    *,
    model: str = RAG_MODEL,
    temperature: float = 0.2,
) -> Tuple[str, Optional[dict]]:
    """Call Gemini with File Search tool and return answer + grounding metadata."""

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

    response = _client().models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=question)])],
        config=config,
    )

    candidate = response.candidates[0] if getattr(response, "candidates", None) else None
    grounding_dict = _grounding_dict(candidate)

    answer_text = getattr(response, "text", None)
    if not answer_text and candidate and candidate.content:
        parts = getattr(candidate.content, "parts", []) or []
        answer_text = "\n".join(part.text for part in parts if getattr(part, "text", None))

    return answer_text or "", grounding_dict
