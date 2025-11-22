"""Pydantic schemas for FastAPI endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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


class TwinAgentRequest(BaseModel):
    """Request for Twin Agent (RAG + Gemini 3 reasoning)."""
    question: str = Field(..., min_length=1, description="User question")
    workspace_id: Optional[str] = Field(None, description="Workspace ID for document retrieval")
    metadata_filter: Optional[str] = Field(None, alias="filter", description="Metadata filter for retrieval")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for thought signature persistence")
    thinking_level: Optional[Literal["low", "high"]] = Field("high", description="Gemini 3 thinking level")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="Generation temperature")

    class Config:
        allow_population_by_field_name = True


class TwinAgentResponse(BaseModel):
    """Response from Twin Agent with Gemini 3 reasoning."""
    answer: str = Field(..., description="Final answer from Gemini 3")
    thought_signatures: List[str] = Field(default_factory=list, description="Thought signatures for persistence")
    retrieved_evidence: bool = Field(False, description="Whether evidence was retrieved from documents")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations from retrieved documents")
    raw: Dict[str, Any] = Field(default_factory=dict, description="Raw Gemini 3 response metadata")


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


class Gemini3MessagePart(BaseModel):
    text: Optional[str] = None
    thought_signature: Optional[str] = Field(default=None, alias="thoughtSignature")


class Gemini3Message(BaseModel):
    role: Literal["user", "assistant", "system", "model"]
    parts: List[Gemini3MessagePart] = Field(default_factory=list)
    text: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_parts(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        parts = data.get("parts")
        if isinstance(parts, str):
            data["parts"] = [{"text": parts}]
        elif parts is None:
            data["parts"] = []
        elif not isinstance(parts, list):
            raise ValueError("parts must be a list or string")
        return data

    @model_validator(mode="after")
    def _ensure_parts(self) -> "Gemini3Message":
        if self.parts:
            return self
        if self.text is None:
            raise ValueError("Each message must include either parts or text.")
        self.parts = [Gemini3MessagePart(text=self.text)]
        return self


class Gemini3Media(BaseModel):
    mime_type: str
    base64_data: str
    media_level: Optional[Literal["image", "pdf", "video_general", "video_text"]] = None


class Gemini3ChatRequest(BaseModel):
    messages: List[Gemini3Message]
    media: Optional[List[Gemini3Media]] = None
    thinking_level: Optional[Literal["low", "high"]] = "high"
    stream: bool = False
    temperature: Optional[float] = None
    conversation_id: Optional[str] = None
    thinking_budget: Optional[int] = Field(default=None, alias="thinkingBudget")


class Gemini3ChatResponse(BaseModel):
    text: str
    usage: Dict[str, Any]
    model_response: Dict[str, Any]
    thought_signatures: List[str]


class Gemini3StructuredToolConfig(BaseModel):
    google_search: bool = False
    url_context: bool = False
    code_execution: bool = False


class Gemini3StructuredRequest(BaseModel):
    prompt: str
    json_schema: Dict[str, Any] = Field(..., alias="jsonSchema")
    tools: Optional[Gemini3StructuredToolConfig] = Field(default_factory=Gemini3StructuredToolConfig)
    temperature: Optional[float] = None


class Gemini3StructuredResponse(BaseModel):
    data: Dict[str, Any]
    raw: Dict[str, Any]
