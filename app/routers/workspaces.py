"""Workspace management, ingestion, diagnostics, and ask endpoints."""
from __future__ import annotations

import asyncio
import secrets
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import schemas
from ..db import get_session
from ..gemini_client import create_file_search_store
from ..models import Document, Workspace
from ..services.ingestion_agent import process_uploaded_folder
from ..services.retrieval import ask_router


FILE_EXT_HINTS = (".pdf", ".docx", ".pptx", ".txt", ".md", ".rtf")


def _is_real_value(value: str | None) -> bool:
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


router = APIRouter(prefix="/workspaces", tags=["workspaces"])


async def _ingest_files_response(
    workspace: Workspace,
    files: List[UploadFile],
    session: AsyncSession,
) -> schemas.FolderUploadResponse:
    imported, companies = await process_uploaded_folder(workspace, files, session)
    await session.commit()

    return schemas.FolderUploadResponse(
        workspace_id=workspace.id,
        file_search_store_name=workspace.file_search_store_name,
        imported=[
            schemas.IngestedDocument(
                doc_id=item["doc_id"],
                filename=item["filename"],
                company=item["company"],
            )
            for item in imported
        ],
        companies=companies,
    )


@router.post("", response_model=schemas.WorkspaceOut, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    body: schemas.WorkspaceCreate,
    session: AsyncSession = Depends(get_session),
):
    workspace_id = secrets.token_hex(8)
    store = await asyncio.to_thread(create_file_search_store, body.name)

    workspace = Workspace(
        id=workspace_id,
        name=body.name,
        file_search_store_name=store.name,
    )
    session.add(workspace)
    await session.commit()

    return schemas.WorkspaceOut(
        workspace_id=workspace_id,
        file_search_store_name=store.name,
    )


@router.post(
    "/{workspace_id}/folder-upload",
    response_model=schemas.FolderUploadResponse,
)
async def upload_folder(
    workspace_id: str,
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
):
    workspace = await session.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    return await _ingest_files_response(workspace, files, session)


@router.post(
    "/{workspace_id}/files",
    response_model=schemas.FolderUploadResponse,
)
async def upload_files(
    workspace_id: str,
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
):
    workspace = await session.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    return await _ingest_files_response(workspace, files, session)


@router.post("/{workspace_id}/ask", response_model=schemas.AskResponse)
async def ask_workspace(
    workspace_id: str,
    payload: schemas.AskRequest,
    session: AsyncSession = Depends(get_session),
):
    workspace = await session.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    if not payload.question:
        raise HTTPException(status_code=400, detail="Missing question")

    metadata_filter = payload.metadata_filter
    if metadata_filter is not None:
        metadata_filter = metadata_filter.strip()
        if not metadata_filter:
            metadata_filter = None

    system_vars: dict = {}

    try:
        result = await ask_router(session, workspace, payload.question, metadata_filter, system_vars=system_vars)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))

    return schemas.AskResponse(**result)


@router.get("/{workspace_id}/diagnostics", response_model=schemas.DiagnosticsResponse)
async def workspace_diagnostics(
    workspace_id: str,
    session: AsyncSession = Depends(get_session),
):
    workspace = await session.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    counts_stmt = (
        select(Document.company, func.count())
        .where(Document.workspace_id == workspace_id)
        .group_by(Document.company)
    )
    doc_type_stmt = (
        select(Document.doc_type, func.count())
        .where(Document.workspace_id == workspace_id)
        .group_by(Document.doc_type)
    )
    time_scope_stmt = (
        select(Document.time_scope, func.count())
        .where(Document.workspace_id == workspace_id)
        .group_by(Document.time_scope)
    )
    recent_stmt = (
        select(
            Document.filename,
            Document.company,
            Document.doc_type,
            Document.time_scope,
            Document.created_at,
        )
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
        .limit(10)
    )

    company_counts = await session.execute(counts_stmt)
    doc_type_counts = await session.execute(doc_type_stmt)
    time_scope_counts = await session.execute(time_scope_stmt)
    recent_docs = await session.execute(recent_stmt)

    filtered_companies = [
        schemas.DiagnosticsCount(value=row[0], count=row[1])
        for row in company_counts
        if _is_real_value(row[0])
    ]
    filtered_doc_types = [
        schemas.DiagnosticsCount(value=row[0], count=row[1])
        for row in doc_type_counts
        if _is_real_value(row[0])
    ]
    filtered_time_scopes = [
        schemas.DiagnosticsCount(value=row[0], count=row[1])
        for row in time_scope_counts
        if _is_real_value(row[0])
    ]

    return schemas.DiagnosticsResponse(
        workspace_id=workspace_id,
        file_search_store_name=workspace.file_search_store_name,
        companies=filtered_companies,
        doc_types=filtered_doc_types,
        time_scopes=filtered_time_scopes,
        recent=[
            schemas.DiagnosticsRecent(
                filename=row[0],
                company=row[1],
                doc_type=row[2],
                time_scope=row[3],
                created_at=row[4],
            )
            for row in recent_docs
        ],
    )
