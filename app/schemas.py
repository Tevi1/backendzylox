"""Pydantic schemas for FastAPI endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class WorkspaceCreate(BaseModel):
    name: str


class WorkspaceOut(BaseModel):
    workspace_id: str
    file_search_store_name: str


class IngestedDocument(BaseModel):
    doc_id: str
    filename: str
    company: Optional[str] = None


class FolderUploadResponse(BaseModel):
    workspace_id: str
    file_search_store_name: str
    imported: List[IngestedDocument]
    companies: List[str]


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="User question to answer using File Search grounded context.",
    )
    metadata_filter: Optional[str] = Field(
        default=None,
        alias="filter",
        description='Gemini File Search metadata filter, e.g. company="highrise" AND doc_type="contract".',
    )

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"


class AskResponse(BaseModel):
    answer: str
    metadata_filter: Optional[str] = None
    grounding_metadata: Optional[dict] = None
    scope: Optional[str] = None
    suggested_filters: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    plan: Optional[dict] = None
    relaxed: Optional[bool] = None


class DiagnosticsCount(BaseModel):
    value: Optional[str]
    count: int


class DiagnosticsRecent(BaseModel):
    filename: str
    company: Optional[str] = None
    doc_type: Optional[str] = None
    time_scope: Optional[str] = None
    created_at: datetime


class DiagnosticsResponse(BaseModel):
    workspace_id: str
    file_search_store_name: str
    companies: List[DiagnosticsCount]
    doc_types: List[DiagnosticsCount]
    time_scopes: List[DiagnosticsCount]
    recent: List[DiagnosticsRecent]
