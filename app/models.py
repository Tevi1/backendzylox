"""SQLAlchemy ORM models for workspaces and documents."""
from __future__ import annotations

import datetime
from datetime import timezone

from sqlalchemy import JSON, Column, DateTime, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _utcnow() -> datetime.datetime:
    """Get current UTC timezone-aware datetime."""
    return datetime.datetime.now(timezone.utc)


class Workspace(Base):
    """Workspace model - each workspace maps to a Gemini File Search store."""

    __tablename__ = "workspaces"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    file_search_store_name = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class Document(Base):
    """Document model - encrypted file blobs with metadata for RAG filtering."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    workspace_id = Column(String, nullable=False, index=True)
    company = Column(String, index=True)  # Frequently filtered
    doc_type = Column(String, index=True)  # Frequently filtered
    time_scope = Column(String, index=True)  # Frequently filtered
    sensitivity = Column(String)
    summary = Column(String)
    folder_path = Column(String)
    filename = Column(String, nullable=False)
    metadata_json = Column(JSON)
    nonce_b64 = Column(String, nullable=False)
    blob_b64 = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class ConversationHistory(Base):
    """Conversation history for Gemini 3 Pro chat with thought signature persistence."""

    __tablename__ = "conversation_histories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, unique=True, nullable=False, index=True)
    messages = Column(JSON, nullable=False, default=list)
    thought_signatures = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)
