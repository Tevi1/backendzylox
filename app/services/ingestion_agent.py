"""Ingestion pipeline that classifies, encrypts, and uploads documents."""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import secrets
import tempfile
from pathlib import PurePosixPath
from typing import Dict, List, Sequence, Tuple

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ..gemini_client import GeminiRAGError, classify_document, upload_file_to_store
from ..ingestion.heuristics import chunk_preset_for_doc_type, derive_metadata
from ..models import Document, Workspace
from .encryption import encrypt_bytes

LOGGER = logging.getLogger(__name__)


def _detect_companies(files: Sequence[UploadFile]) -> List[str]:
    companies: set[str] = set()
    for file in files:
        path = PurePosixPath(file.filename or "")
        if len(path.parts) >= 2:
            companies.add(path.parts[1])
    return sorted(companies)


def _text_preview(raw_bytes: bytes, limit: int = 5000) -> str:
    if not raw_bytes:
        return ""
    return raw_bytes[:limit].decode("utf-8", errors="ignore")


def _normalize_company(value: str | None) -> str | None:
    if not value:
        return None
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    slug = "-".join(filter(None, slug.split("-")))
    slug = slug.strip("-")
    if not slug:
        return None
    EXT_HINTS = ("pdf", "docx", "pptx", "txt", "md", "rtf", "json", "csv", "xlsx")
    for ext in EXT_HINTS:
        if slug.endswith(f"-{ext}") or slug == ext:
            return None
    return slug or None


def _default_metadata(company: str | None) -> Dict[str, str | None]:
    return {
        "company": company,
        "doc_type": None,
        "time_scope": None,
        "sensitivity": "internal",
        "summary": "",
    }


def _safe_parse_classification(payload: str | Dict[str, object], fallback_company: str | None) -> Dict[str, str | None]:
    if isinstance(payload, dict):
        data = payload
    else:
        try:
            data = json.loads(payload or "{}")
        except json.JSONDecodeError:
            data = {}

    if isinstance(data, dict) and "sensitivity_level" in data and "sensitivity" not in data:
        data["sensitivity"] = data["sensitivity_level"]

    defaults = _default_metadata(fallback_company)
    defaults.update({k: v for k, v in data.items() if k in defaults})
    return defaults


async def _upload_to_file_search(store_name: str, tmp_path: str, config: Dict[str, object]) -> Dict[str, object]:
    return await asyncio.to_thread(upload_file_to_store, store_name, tmp_path, config)


async def process_uploaded_folder(
    workspace: Workspace,
    files: Sequence[UploadFile],
    session: AsyncSession,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """
    Process multi-file folder uploads (webkitdirectory).

    For each file:
    1. Extract metadata via heuristics (filename/path patterns)
    2. Classify via Gemini (company, doc_type, sensitivity)
    3. Encrypt file bytes (AES-256-GCM)
    4. Upload to Gemini File Search with metadata tags
    5. Persist encrypted blob + metadata to database

    Args:
        workspace: Target workspace for uploads.
        files: Sequence of uploaded files.
        session: Database session.

    Returns:
        Tuple of (imported_documents_list, discovered_companies_list).

    Raises:
        GeminiRAGError: If classification or File Search upload fails.
    """
    if not files:
        LOGGER.warning("Empty file list provided for workspace %s", workspace.id)
        return [], []

    LOGGER.info("Processing %d files for workspace %s", len(files), workspace.id)
    company_candidates = _detect_companies(files)
    imported: List[Dict[str, object]] = []
    discovered: set[str] = set()

    for upload in files:
        file_path = PurePosixPath(upload.filename or "")
        raw_bytes = await upload.read()
        if not raw_bytes:
            continue

        folder_path = str(file_path.parent)
        company_guess = file_path.parts[1] if len(file_path.parts) >= 2 else None
        mime_type = (
            upload.content_type
            or mimetypes.guess_type(file_path.name)[0]
            or "application/octet-stream"
        )

        heuristics_meta = derive_metadata(folder_path, file_path.name, company_guess)
        text_preview = _text_preview(raw_bytes)
        try:
            classification_payload = await asyncio.to_thread(
                classify_document,
                company_candidates,
                str(file_path),
                text_preview,
            )
            classification_meta = _safe_parse_classification(classification_payload, company_guess)
        except GeminiRAGError as exc:
            LOGGER.warning("Classification failed for %s: %s, using heuristics only", file_path.name, exc)
            classification_meta = _default_metadata(company_guess)
        except Exception as exc:
            LOGGER.warning("Unexpected classification error for %s: %s, using heuristics only", file_path.name, exc)
            classification_meta = _default_metadata(company_guess)

        combined_meta = {
            "company": classification_meta.get("company") or heuristics_meta.get("company"),
            "doc_type": classification_meta.get("doc_type") or heuristics_meta.get("doc_type") or "misc",
            "time_scope": classification_meta.get("time_scope") or heuristics_meta.get("time_scope") or "unknown",
            "sensitivity": classification_meta.get("sensitivity") or heuristics_meta.get("sensitivity") or "internal",
            "summary": classification_meta.get("summary") or "",
        }

        company_final = _normalize_company(combined_meta.get("company")) or _normalize_company(company_guess)
        if company_final == "unknown":
            company_final = None
        doc_type_final = combined_meta.get("doc_type") or "misc"
        time_scope_final = combined_meta.get("time_scope") or "current"
        sensitivity_final = combined_meta.get("sensitivity") or "internal"
        summary_final = combined_meta.get("summary") or ""

        encrypted = encrypt_bytes(raw_bytes)

        doc_id = secrets.token_hex(16)
        document = Document(
            id=doc_id,
            workspace_id=workspace.id,
            company=company_final,
            doc_type=doc_type_final,
            time_scope=time_scope_final,
            sensitivity=sensitivity_final,
            summary=summary_final,
            folder_path=folder_path,
            filename=file_path.name,
            metadata_json={
                "classification": classification_meta,
                "heuristics": heuristics_meta,
                "final": combined_meta,
            },
            nonce_b64=encrypted["nonce"],
            blob_b64=encrypted["blob"],
            size_bytes=len(raw_bytes),
        )
        session.add(document)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            custom_metadata = [
                {"key": "workspace_id", "string_value": workspace.id},
                {"key": "doc_id", "string_value": doc_id},
                {"key": "sensitivity", "string_value": document.sensitivity},
            ]
            if document.doc_type:
                custom_metadata.append({"key": "doc_type", "string_value": document.doc_type})
            if document.time_scope:
                custom_metadata.append({"key": "time_scope", "string_value": document.time_scope})
            if document.company:
                custom_metadata.append({"key": "company", "string_value": document.company})
            if document.folder_path:
                custom_metadata.append({"key": "folder_path", "string_value": document.folder_path})

            tokens, overlap = chunk_preset_for_doc_type(doc_type_final)
            upload_config = {
                "display_name": file_path.name,
                "mime_type": mime_type,
                "custom_metadata": custom_metadata,
                "chunking_config": {
                    "white_space_config": {
                        "max_tokens_per_chunk": int(tokens),
                        "max_overlap_tokens": int(overlap),
                    }
                },
            }
            LOGGER.debug("Uploading %s to File Search store %s", file_path.name, workspace.file_search_store_name)
            response = await _upload_to_file_search(
                workspace.file_search_store_name,
                tmp_path,
                upload_config,
            )
            LOGGER.info("Successfully indexed %s (doc_id=%s)", file_path.name, doc_id)
        except GeminiRAGError as exc:
            LOGGER.error("File Search upload failed for %s: %s", file_path.name, exc)
            raise
        except Exception as exc:
            LOGGER.exception("Unexpected error uploading %s to File Search", file_path.name)
            raise GeminiRAGError(f"File Search upload failed: {exc}") from exc
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

        imported.append(
            {
                "doc_id": doc_id,
                "filename": document.filename,
                "company": document.company,
                "metadata": upload_config["custom_metadata"],
                "file_search_response": response,
            }
        )

        if document.company:
            discovered.add(document.company)

    LOGGER.info("Ingestion complete: %d files imported, %d companies discovered", len(imported), len(discovered))
    return imported, sorted(discovered)

