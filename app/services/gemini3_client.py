"""Gemini 3 Pro preview helpers with text + multimodal support."""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from google import genai
from google.genai import types

from ..config import get_config

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "gemini-3-pro-preview"
DEFAULT_TEMPERATURE = 1.0

MEDIA_RESOLUTION_PRESETS = {
    "image": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    "pdf": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "video_general": types.MediaResolution.MEDIA_RESOLUTION_LOW,
    "video_text": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}


class Gemini3Error(RuntimeError):
    """Base exception for Gemini 3 operations."""


class ConfigError(Gemini3Error):
    """Raised when Gemini configuration is invalid."""


def _require_api_key(explicit_key: Optional[str] = None) -> str:
    """Get API key from config or explicit parameter."""
    if explicit_key:
        return explicit_key
    config = get_config()
    api_key = config.get_gemini_api_key()
    if not api_key:
        raise ConfigError("GEMINI_API_KEY environment variable is required.")
    return api_key


def _build_thinking_config(thinking_level: Optional[str]) -> Optional[types.ThinkingConfig]:
    if not thinking_level:
        return None
    level = thinking_level.lower()
    include = level != "low"
    return types.ThinkingConfig(include_thoughts=include)


def _part_from_dict(part: Dict[str, Any]) -> types.Part:
    if "text" in part and part["text"] is not None:
        return types.Part(text=part["text"])
    if part.get("thought_signature"):
        return types.Part(thought_signature=part["thought_signature"])
    if part.get("inline_data"):
        inline = part["inline_data"]
        return types.Part(
            inline_data=types.Blob(
                mime_type=inline["mime_type"],
                data=inline["data"],
            )
        )
    raise ValueError("Unsupported message part payload.")


def _content_from_messages(messages: Sequence[Dict[str, Any]]) -> List[types.Content]:
    contents: List[types.Content] = []
    for message in messages:
        role = message.get("role", "user")
        parts: List[types.Part] = []
        for part in message.get("parts", []):
            if isinstance(part, types.Part):
                parts.append(part)
                continue
            parts.append(_part_from_dict(part))
        if not parts:
            continue
        contents.append(types.Content(role=role, parts=parts))
    return contents


def _tool_list(tool_flags: Optional[Dict[str, bool]]) -> Optional[List[types.Tool]]:
    if not tool_flags:
        return None
    tools: List[types.Tool] = []
    if tool_flags.get("google_search"):
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    if tool_flags.get("url_context"):
        tools.append(types.Tool(url_context=types.UrlContext()))
    if tool_flags.get("code_execution"):
        tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
    return tools or None


def _media_resolution_from_levels(levels: Iterable[str]) -> Optional[types.MediaResolution]:
    order = [
        types.MediaResolution.MEDIA_RESOLUTION_LOW,
        types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    ]
    target = None
    for level in levels:
        preset = MEDIA_RESOLUTION_PRESETS.get(level)
        if not preset:
            continue
        if not target or order.index(preset) > order.index(target):
            target = preset
    return target


def inline_data_part(mime_type: str, data_b64: str) -> types.Part:
    """Convert base64 media into a Gemini inline data part."""
    try:
        decoded = base64.b64decode(data_b64)
    except Exception as exc:  # pragma: no cover - fast failure
        raise ValueError("Invalid base64 media payload") from exc
    return types.Part(inline_data=types.Blob(mime_type=mime_type, data=decoded))


@dataclass
class GeminiResponsePayload:
    text: str
    raw: Dict[str, Any]
    usage: Dict[str, Any]
    thought_signatures: List[str]
    response: types.GenerateContentResponse


class Gemini3Client:
    """
    Convenience wrapper for Gemini 3 Pro preview API.

    Provides separate clients for text (v1beta) and multimodal (v1alpha) operations.
    All methods raise Gemini3Error on failure.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize Gemini 3 client with API key.

        Args:
            api_key: Optional explicit API key. If not provided, uses config.

        Raises:
            ConfigError: If API key is missing.
        """
        key = _require_api_key(api_key)
        self._text_client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})
        self._media_client = genai.Client(api_key=key, http_options={"api_version": "v1alpha"})

    def _run(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
        thinking_level: Optional[str] = None,
        tools: Optional[Dict[str, bool]] = None,
        media_resolution: Optional[types.MediaResolution] = None,
        response_mime_type: Optional[str] = None,
        response_json_schema: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        stream: bool = False,
        structured: bool = False,
        use_media_client: bool = False,
    ):
        contents = _content_from_messages(messages)
        config = types.GenerateContentConfig(
            temperature=float(temperature),
            thinking_config=_build_thinking_config(thinking_level),
            tools=_tool_list(tools),
            media_resolution=media_resolution,
            response_mime_type=response_mime_type,
            response_json_schema=response_json_schema,
            system_instruction=system_instruction,
        )
        client = self._media_client if use_media_client else self._text_client
        if stream:
            return client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        )

    def generate_text(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
        thinking_level: Optional[str] = None,
        tools: Optional[Dict[str, bool]] = None,
        system_instruction: Optional[str] = None,
        stream: bool = False,
    ):
        """
        Generate text-only response using Gemini 3 Pro (v1beta client).

        Args:
            messages: List of message dicts with 'role' and 'parts'.
            temperature: Generation temperature (default: 1.0).
            thinking_level: Optional 'low' or 'high' thinking level.
            tools: Optional dict with google_search, url_context, code_execution flags.
            system_instruction: Optional system instruction for answer style/behavior.
            stream: Whether to stream response (not yet fully implemented).

        Returns:
            GenerateContentResponse from Gemini API.

        Raises:
            Gemini3Error: If API call fails.
        """
        try:
            return self._run(
                messages=messages,
                temperature=temperature,
                thinking_level=thinking_level,
                tools=tools,
                system_instruction=system_instruction,
                stream=stream,
            )
        except Exception as exc:
            raise Gemini3Error(f"Gemini 3 text generation failed: {exc}") from exc

    def generate_multimodal(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
        thinking_level: Optional[str] = None,
        tools: Optional[Dict[str, bool]] = None,
        media_resolution: Optional[types.MediaResolution] = None,
        stream: bool = False,
    ):
        """
        Generate multimodal response (text + images/PDF/video) using Gemini 3 Pro (v1alpha client).

        Args:
            messages: List of message dicts with 'role' and 'parts' (may include inline_data).
            temperature: Generation temperature (default: 1.0).
            thinking_level: Optional 'low' or 'high' thinking level.
            tools: Optional dict with google_search, url_context, code_execution flags.
            media_resolution: Optional MediaResolution enum for image/PDF/video processing quality.
            stream: Whether to stream response (not yet fully implemented).

        Returns:
            GenerateContentResponse from Gemini API.

        Raises:
            Gemini3Error: If API call fails.
        """
        try:
            return self._run(
                messages=messages,
                temperature=temperature,
                thinking_level=thinking_level,
                tools=tools,
                media_resolution=media_resolution,
                stream=stream,
                use_media_client=True,
            )
        except Exception as exc:
            raise Gemini3Error(f"Gemini 3 multimodal generation failed: {exc}") from exc

    def generate_structured(
        self,
        *,
        prompt: str,
        json_schema: Dict[str, Any],
        temperature: float = DEFAULT_TEMPERATURE,
        tools: Optional[Dict[str, bool]] = None,
    ):
        """
        Generate structured JSON output matching the provided schema.

        Args:
            prompt: User prompt to generate structured data for.
            json_schema: JSON schema dict defining the expected output structure.
            temperature: Generation temperature (default: 1.0).
            tools: Optional dict with google_search, url_context, code_execution flags.

        Returns:
            GenerateContentResponse with JSON text matching the schema.

        Raises:
            Gemini3Error: If API call fails or response doesn't match schema.
        """
        try:
            message = {"role": "user", "parts": [{"text": prompt}]}
            return self._run(
                messages=[message],
                temperature=temperature,
                tools=tools,
                response_mime_type="application/json",
                response_json_schema=json_schema,
            )
        except Exception as exc:
            raise Gemini3Error(f"Gemini 3 structured generation failed: {exc}") from exc


__all__ = [
    "ConfigError",
    "Gemini3Client",
    "GeminiResponsePayload",
    "MEDIA_RESOLUTION_PRESETS",
    "DEFAULT_TEMPERATURE",
    "inline_data_part",
    "_media_resolution_from_levels",
]

