"""Twin Agent: Combines Gemini 2.5 RAG retrieval with Gemini 3 reasoning."""
from __future__ import annotations

import asyncio
import base64
import logging
import re
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from ..gemini_client import GeminiRAGError, ask_with_file_search
from ..models import Workspace
from ..services.gemini3_client import Gemini3Client, Gemini3Error

LOGGER = logging.getLogger(__name__)

# Fallback thought signature constant
FALLBACK_THOUGHT_SIGNATURE = "context_engineering_is_the_way_to_go"

# Keywords that suggest document-related queries
DOCUMENT_KEYWORDS = [
    r"\b(contract|memo|email|document|file|report|agreement|proposal|pitch|deck)\b",
    r"\b(upload|indexed|stored|workspace|document|file)\b",
    r"\b(company|doc_type|time_scope|sensitivity)\b",  # Metadata fields
]


def _needs_retrieval(question: str, workspace: Optional[Workspace] = None) -> bool:
    """
    Determine if a question requires document retrieval.

    Args:
        question: User question text.
        workspace: Optional workspace to check for document availability.

    Returns:
        True if question likely needs document retrieval, False otherwise.
    """
    if not workspace:
        return False

    question_lower = question.lower()

    # Check for document-related keywords
    for pattern in DOCUMENT_KEYWORDS:
        if re.search(pattern, question_lower, re.IGNORECASE):
            LOGGER.debug("Question matches document keyword pattern: %s", pattern)
            return True

    # Check for references to specific metadata values
    # (This is a simple heuristic; could be enhanced with NLP)
    if any(word in question_lower for word in ["contract", "memo", "email", "report", "agreement"]):
        return True

    return False


def _format_retrieved_evidence(
    retrieved_text: str,
    grounding_metadata: Optional[Dict],
    question: str,
) -> str:
    """
    Format retrieved evidence for Gemini 3 prompt.

    Args:
        retrieved_text: Raw answer from Gemini 2.5 RAG.
        grounding_metadata: Grounding metadata with sources.
        question: Original user question.

    Returns:
        Formatted prompt string for Gemini 3.
    """
    evidence_parts = []

    # Add retrieved answer as evidence
    if retrieved_text:
        evidence_parts.append(f"## Retrieved Evidence\n{retrieved_text}")

    # Add source citations if available
    if grounding_metadata:
        sources = []
        retrieved_chunks = grounding_metadata.get("retrieved_contexts", [])
        if retrieved_chunks:
            for idx, chunk in enumerate(retrieved_chunks[:5], 1):  # Limit to top 5
                chunk_data = chunk.get("chunk", {})
                file_name = chunk_data.get("file_name", "Unknown")
                page = chunk_data.get("page", "")
                sources.append(f"{idx}. {file_name}" + (f" (page {page})" if page else ""))

        if sources:
            evidence_parts.append(f"\n## Sources\n" + "\n".join(sources))

    evidence_text = "\n\n".join(evidence_parts) if evidence_parts else retrieved_text

    # Construct final prompt for Gemini 3
    prompt = f"""You are a sovereign AI assistant with deep reasoning capabilities. A user has asked a question, and relevant evidence has been retrieved from their document workspace.

**User Question:**
{question}

**Retrieved Evidence:**
{evidence_text}

**Your Task:**
Provide a comprehensive, well-reasoned answer based on the evidence above. Your response should:
- Be deeply analytical and thoughtful
- Synthesize information from multiple sources when relevant
- Maintain a personalized, conversational tone
- Cite specific sources when making claims
- Show your reasoning process when appropriate
- Avoid hallucination by grounding all claims in the provided evidence

**Answer:**"""

    return prompt


async def retrieve_with_gemini25(
    workspace: Workspace,
    question: str,
    metadata_filter: Optional[str] = None,
) -> Tuple[str, Optional[Dict]]:
    """
    Retrieve evidence using Gemini 2.5 Pro with File Search.

    Args:
        workspace: Workspace containing indexed documents.
        question: User question.
        metadata_filter: Optional metadata filter string.

    Returns:
        Tuple of (retrieved_text, grounding_metadata).

    Raises:
        GeminiRAGError: If retrieval fails.
    """
    try:
        retrieved_text, grounding_metadata = await asyncio.to_thread(
            ask_with_file_search,
            workspace.file_search_store_name,
            question,
            metadata_filter,
        )
        LOGGER.info(
            "Retrieved evidence for workspace %s: %d chars, %d sources",
            workspace.id,
            len(retrieved_text),
            len(grounding_metadata.get("retrieved_contexts", [])) if grounding_metadata else 0,
        )
        return retrieved_text, grounding_metadata
    except Exception as exc:
        LOGGER.error("Gemini 2.5 retrieval failed: %s", exc)
        raise GeminiRAGError(f"Document retrieval failed: {exc}") from exc


async def reason_with_gemini3(
    client: Gemini3Client,
    prompt: str,
    conversation_id: Optional[str] = None,
    thinking_level: Optional[str] = "high",
    temperature: float = 1.0,
    stored_signatures: Optional[List[str]] = None,
) -> Tuple[str, List[str], Dict]:
    """
    Generate final answer using Gemini 3 Pro with reasoning.

    Args:
        client: Gemini 3 client instance.
        prompt: Formatted prompt with evidence.
        conversation_id: Optional conversation ID for persistence.
        thinking_level: Thinking level (low/high).
        temperature: Generation temperature.
        stored_signatures: Previous thought signatures to inject.

    Returns:
        Tuple of (answer_text, thought_signatures, raw_response_dict).

    Raises:
        Gemini3Error: If generation fails.
    """
    try:
        # Build messages with thought signatures if available
        messages = [{"role": "user", "parts": [{"text": prompt}]}]
        if stored_signatures:
            # Inject thought signatures into the message
            for sig in stored_signatures:
                messages[0]["parts"].append({"thoughtSignature": sig})

        response = await asyncio.to_thread(
            client.generate_text,
            messages=messages,
            temperature=temperature,
            thinking_level=thinking_level,
        )

        # Extract answer text
        answer_text = getattr(response, "text", "") or ""
        if not answer_text and hasattr(response, "candidates"):
            candidate = response.candidates[0] if response.candidates else None
            if candidate and hasattr(candidate, "content"):
                parts = getattr(candidate.content, "parts", []) or []
                answer_text = "\n".join(
                    part.text for part in parts if hasattr(part, "text") and part.text
                )

        # Extract thought signatures (same logic as gemini3 router)
        thought_signatures = []
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    # Check for thought_signature attribute
                    if hasattr(part, "thought_signature") and part.thought_signature:
                        sig = part.thought_signature
                        # Handle both string and bytes signatures
                        if isinstance(sig, bytes):
                            sig = base64.b64encode(sig).decode("utf-8")
                        thought_signatures.append(str(sig))
                    # Also check for thoughtSignature (camelCase)
                    elif hasattr(part, "thoughtSignature") and part.thoughtSignature:
                        sig = part.thoughtSignature
                        if isinstance(sig, bytes):
                            sig = base64.b64encode(sig).decode("utf-8")
                        thought_signatures.append(str(sig))

        if not thought_signatures:
            thought_signatures = [FALLBACK_THOUGHT_SIGNATURE]

        # Get raw response for metadata
        raw_dict = response.to_json_dict() if hasattr(response, "to_json_dict") else {}

        LOGGER.info(
            "Gemini 3 generated answer: %d chars, %d thought signatures",
            len(answer_text),
            len(thought_signatures),
        )

        return answer_text, thought_signatures, raw_dict

    except Exception as exc:
        LOGGER.error("Gemini 3 reasoning failed: %s", exc)
        raise Gemini3Error(f"Reasoning generation failed: {exc}") from exc


async def twin_agent_query(
    session: AsyncSession,
    workspace: Optional[Workspace],
    question: str,
    gemini3_client: Gemini3Client,
    conversation_id: Optional[str] = None,
    metadata_filter: Optional[str] = None,
    thinking_level: Optional[str] = "high",
    temperature: float = 1.0,
    stored_signatures: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Combined RAG→Gemini 3 pipeline for document-grounded reasoning.

    Flow:
    1. If question needs retrieval and workspace exists: Retrieve with Gemini 2.5
    2. Format evidence for Gemini 3
    3. Generate final answer with Gemini 3
    4. Return combined response

    Args:
        session: Database session.
        workspace: Optional workspace for document retrieval.
        question: User question.
        gemini3_client: Gemini 3 client instance.
        conversation_id: Optional conversation ID.
        metadata_filter: Optional metadata filter for retrieval.
        thinking_level: Gemini 3 thinking level.
        temperature: Generation temperature.
        stored_signatures: Previous thought signatures.

    Returns:
        Dict with answer, sources, thought_signatures, and metadata.
    """
    retrieved_text = None
    grounding_metadata = None
    needs_rag = _needs_retrieval(question, workspace)

    # Step 1: Retrieve evidence if needed
    if needs_rag and workspace:
        LOGGER.info("Question requires retrieval, using Gemini 2.5 RAG")
        try:
            retrieved_text, grounding_metadata = await retrieve_with_gemini25(
                workspace,
                question,
                metadata_filter,
            )
        except GeminiRAGError as exc:
            LOGGER.warning("RAG retrieval failed, proceeding without evidence: %s", exc)
            # Continue without evidence - let Gemini 3 handle it
            retrieved_text = None
    else:
        LOGGER.info("Question does not require retrieval, using Gemini 3 directly")

    # Step 2: Format prompt for Gemini 3
    if retrieved_text:
        prompt = _format_retrieved_evidence(retrieved_text, grounding_metadata, question)
    else:
        # General reasoning question - use original question
        prompt = question

    # Step 3: Generate final answer with Gemini 3
    answer_text, thought_signatures, raw_response = await reason_with_gemini3(
        gemini3_client,
        prompt,
        conversation_id,
        thinking_level,
        temperature,
        stored_signatures,
    )

    # Step 4: Format response
    response = {
        "answer": answer_text,
        "thought_signatures": thought_signatures,
        "raw": raw_response,
        "retrieved_evidence": retrieved_text is not None,
        "sources": [],
    }

    # Extract sources from grounding metadata
    if grounding_metadata:
        retrieved_chunks = grounding_metadata.get("retrieved_contexts", [])
        for chunk in retrieved_chunks[:10]:  # Limit to top 10
            chunk_data = chunk.get("chunk", {})
            source = {
                "file_name": chunk_data.get("file_name", "Unknown"),
                "page": chunk_data.get("page"),
                "text_preview": chunk_data.get("text", "")[:200] if chunk_data.get("text") else None,
            }
            response["sources"].append(source)

    return response

