"""Gemini 3 Pro preview chat + structured endpoints."""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from google.genai import errors as genai_errors
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import ConversationHistory
from ..schemas import (
    Gemini3ChatRequest,
    Gemini3ChatResponse,
    Gemini3StructuredRequest,
    Gemini3StructuredResponse,
)
from ..services.gemini3_client import (
    DEFAULT_TEMPERATURE,
    MEDIA_RESOLUTION_PRESETS,
    ConfigError,
    Gemini3Error,
    Gemini3Client,
    inline_data_part,
    _media_resolution_from_levels,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat/gemini3", tags=["gemini3"])

FALLBACK_THOUGHT_SIGNATURE = "context_engineering_is_the_way_to_go"
PDF_MIME = "application/pdf"


def get_gemini3_client() -> Gemini3Client:
    return Gemini3Client()


async def _fetch_conversation(session: AsyncSession, conversation_id: Optional[str]) -> Optional[ConversationHistory]:
    if not conversation_id:
        return None
    result = await session.execute(
        select(ConversationHistory).where(ConversationHistory.conversation_id == conversation_id)
    )
    return result.scalars().first()


def _to_base64(value: bytes) -> str:
    return base64.b64encode(value).decode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return _to_base64(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    return value


async def _persist_conversation(
    session: AsyncSession,
    conversation_id: Optional[str],
    history_messages: List[Dict[str, Any]],
    thought_signatures: List[str],
) -> None:
    if not conversation_id:
        return
    convo = await _fetch_conversation(session, conversation_id)
    safe_messages = _json_safe(history_messages)
    safe_signatures = _json_safe(thought_signatures)
    if convo:
        convo.messages = safe_messages
        convo.thought_signatures = safe_signatures
    else:
        convo = ConversationHistory(
            conversation_id=conversation_id,
            messages=safe_messages,
            thought_signatures=safe_signatures,
        )
        session.add(convo)
    await session.commit()


def _serialize_request_messages(payload: Gemini3ChatRequest) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for message in payload.messages:
        parts: List[Dict[str, Any]] = []
        for part in message.parts:
            part_data = part.model_dump(by_alias=True, exclude_none=True)
            if "thoughtSignature" in part_data:
                part_data["thought_signature"] = part_data.pop("thoughtSignature")
            parts.append(part_data)
        serialized.append({"role": message.role, "parts": parts})
    return serialized


def _clone_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return copy.deepcopy(list(messages))


def _inject_signatures(messages: List[Dict[str, Any]], signatures: Sequence[str]) -> List[Dict[str, Any]]:
    if not signatures:
        return messages
    signature_parts = [{"thought_signature": sig} for sig in signatures if sig]
    if not signature_parts:
        return messages
    return [{"role": "system", "parts": signature_parts}] + messages


def _latest_user_index(messages: Sequence[Dict[str, Any]]) -> Optional[int]:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            return idx
    return None


def _infer_media_level(mime_type: str, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in MEDIA_RESOLUTION_PRESETS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported media_level '{explicit}'.",
            )
        return explicit
    lowered = mime_type.lower()
    if lowered.startswith("image/"):
        return "image"
    if lowered == PDF_MIME:
        return "pdf"
    if lowered.startswith("video/"):
        return "video_general"
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported media type '{mime_type}'.",
    )


def _attach_media_parts(
    messages: List[Dict[str, Any]],
    media_items: Optional[Sequence[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not media_items:
        return messages, []
    if not messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Messages are required for media uploads.")
    target_idx = _latest_user_index(messages)
    if target_idx is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Media attachments require at least one user message.",
        )
    levels: List[str] = []
    for item in media_items:
        level = _infer_media_level(item["mime_type"], item.get("media_level"))
        levels.append(level)
        try:
            part = inline_data_part(item["mime_type"], item["base64_data"])
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        messages[target_idx]["parts"].append(part)
    return messages, levels


def _extract_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    collected: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                collected.append(part_text)
    return "\n".join(collected).strip()


def _extract_usage(response) -> Dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "to_json_dict"):
        return usage.to_json_dict()
    return {}


def _extract_signatures(response) -> List[str]:
    signatures: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            signature = getattr(part, "thought_signature", None)
            if isinstance(signature, bytes):
                signature = _to_base64(signature)
            if signature:
                signatures.append(signature)
    if not signatures:
        signatures.append(FALLBACK_THOUGHT_SIGNATURE)
    return signatures


def _media_payload(media_models: Optional[Sequence[Any]]) -> Optional[List[Dict[str, Any]]]:
    if not media_models:
        return None
    payload: List[Dict[str, Any]] = []
    for media in media_models:
        payload.append(
            {
                "mime_type": media.mime_type,
                "base64_data": media.base64_data,
                "media_level": media.media_level,
            }
        )
    return payload


async def _call_model_with_retry(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    **kwargs,
):
    """Call model function with exponential backoff retry for transient errors."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(func, **kwargs)
        except genai_errors.ServerError as exc:
            # Retry on 500/503 errors from Google
            status_code = getattr(exc, "status_code", 500)
            if status_code in (500, 503) and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                LOGGER.warning(
                    "Gemini API returned %s (attempt %d/%d), retrying in %.1fs",
                    status_code,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                last_exc = exc
                continue
            raise
        except Exception as exc:
            # Don't retry on other errors (400, 401, 403, etc.)
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Retry loop exhausted without result")


async def _call_model(func, **kwargs):
    """Legacy wrapper - use _call_model_with_retry for automatic retries."""
    return await _call_model_with_retry(func, **kwargs)


def _streaming_response(payload: Dict[str, Any]) -> StreamingResponse:
    async def iterator():
        yield json.dumps(payload).encode("utf-8")

    return StreamingResponse(iterator(), media_type="application/json")


@router.post("", response_model=Gemini3ChatResponse)
async def chat_gemini3(
    payload: Gemini3ChatRequest,
    session: AsyncSession = Depends(get_session),
    client: Gemini3Client = Depends(get_gemini3_client),
):
    if not payload.messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="messages are required")
    if payload.thinking_budget is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="thinking_budget is not supported for Gemini 3 requests.",
        )

    temperature = payload.temperature if payload.temperature is not None else DEFAULT_TEMPERATURE
    if payload.temperature is not None and payload.temperature != DEFAULT_TEMPERATURE:
        LOGGER.warning("Non-default temperature requested: %s", payload.temperature)

    conversation = await _fetch_conversation(session, payload.conversation_id)
    stored_signatures = conversation.thought_signatures if conversation else []
    history_messages = conversation.messages if conversation else []

    incoming_messages = _serialize_request_messages(payload)
    combined_history = history_messages + incoming_messages

    messages_for_model = _inject_signatures(_clone_messages(combined_history), stored_signatures)

    media_payload = _media_payload(payload.media)
    messages_with_media, media_levels = _attach_media_parts(messages_for_model, media_payload)
    media_resolution = _media_resolution_from_levels(media_levels)

    try:
        if media_payload:
            response = await _call_model_with_retry(
                client.generate_multimodal,
                messages=messages_with_media,
                temperature=temperature,
                thinking_level=payload.thinking_level,
                media_resolution=media_resolution,
                stream=False,
            )
        else:
            response = await _call_model_with_retry(
                client.generate_text,
                messages=messages_with_media,
                temperature=temperature,
                thinking_level=payload.thinking_level,
                stream=False,
            )
    except ConfigError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except genai_errors.ServerError as exc:
        status_code = getattr(exc, "status_code", 500)
        LOGGER.error("Gemini API ServerError %s after retries: %s", status_code, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": f"Gemini 3 Pro API error ({status_code}). Please retry in a moment.", "code": "GEMINI_SERVER_ERROR"},
        ) from exc
    except genai_errors.APIError as exc:
        status_code = getattr(exc, "status_code", 500)
        LOGGER.error("Gemini API error %s: %s", status_code, exc)
        if status_code == 429:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"message": "Rate limit exceeded. Please wait before retrying.", "code": "RATE_LIMIT"},
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": f"Gemini API error ({status_code}). Please retry.", "code": "GEMINI_API_ERROR"},
        ) from exc
    except Exception as exc:  # pragma: no cover - network err
        LOGGER.exception("Gemini 3 invocation failed with unexpected error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Gemini 3 Pro is temporarily unavailable. Please retry.", "code": "UNKNOWN_ERROR"},
        ) from exc

    response_text = _extract_text(response)
    usage = _extract_usage(response)
    signatures = _extract_signatures(response)
    raw_dict = response.to_json_dict() if hasattr(response, "to_json_dict") else {}

    updated_history = combined_history + [{"role": "model", "parts": [{"text": response_text}]}]
    await _persist_conversation(session, payload.conversation_id, updated_history, signatures)

    payload_dict = Gemini3ChatResponse(
        text=response_text,
        usage=usage,
        model_response=raw_dict,
        thought_signatures=signatures,
    ).model_dump()

    if payload.stream:
        return _streaming_response(payload_dict)
    return payload_dict


@router.post("/structured", response_model=Gemini3StructuredResponse)
async def structured_output(
    payload: Gemini3StructuredRequest,
    client: Gemini3Client = Depends(get_gemini3_client),
):
    temperature = payload.temperature if payload.temperature is not None else DEFAULT_TEMPERATURE
    if payload.temperature is not None and payload.temperature != DEFAULT_TEMPERATURE:
        LOGGER.warning("Non-default temperature requested (structured): %s", payload.temperature)

    tool_flags = payload.tools.model_dump() if payload.tools else {}
    tool_mapping = {
        "google_search": tool_flags.get("google_search"),
        "url_context": tool_flags.get("url_context"),
        "code_execution": tool_flags.get("code_execution"),
    }

    try:
        response = await _call_model_with_retry(
            client.generate_structured,
            prompt=payload.prompt,
            json_schema=payload.json_schema,
            temperature=temperature,
            tools=tool_mapping,
        )
    except ConfigError as exc:
        LOGGER.error("Gemini 3 structured configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(exc), "code": "CONFIG_ERROR"},
        ) from exc
    except Gemini3Error as exc:
        LOGGER.error("Gemini 3 structured client error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": str(exc), "code": "GEMINI3_ERROR"},
        ) from exc
    except genai_errors.ServerError as exc:
        status_code = getattr(exc, "status_code", 500)
        LOGGER.error("Gemini structured API ServerError %s after retries: %s", status_code, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": f"Gemini 3 structured API error ({status_code}). Please retry.", "code": "GEMINI_SERVER_ERROR"},
        ) from exc
    except genai_errors.APIError as exc:
        status_code = getattr(exc, "status_code", 500)
        LOGGER.error("Gemini structured API error %s: %s", status_code, exc)
        if status_code == 429:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"message": "Rate limit exceeded. Please wait before retrying.", "code": "RATE_LIMIT"},
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": f"Gemini structured API error ({status_code}). Please retry.", "code": "GEMINI_API_ERROR"},
        ) from exc
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Structured Gemini call failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Gemini 3 structured output is unavailable. Retry later.", "code": "UNKNOWN_ERROR"},
        ) from exc

    response_text = _extract_text(response)
    raw_dict = response.to_json_dict() if hasattr(response, "to_json_dict") else {}
    try:
        data = json.loads(response_text or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned malformed JSON.",
        ) from exc

    return Gemini3StructuredResponse(data=data, raw=raw_dict)

