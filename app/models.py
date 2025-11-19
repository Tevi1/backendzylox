"""SQLAlchemy ORM models for workspaces and documents."""
from __future__ import annotations

import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    file_search_store_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    workspace_id = Column(String, nullable=False, index=True)
    company = Column(String)
    doc_type = Column(String)
    time_scope = Column(String)
    sensitivity = Column(String)
    summary = Column(String)
    folder_path = Column(String)
    filename = Column(String, nullable=False)
    metadata_json = Column(JSON)
    nonce_b64 = Column(String, nullable=False)
    blob_b64 = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
