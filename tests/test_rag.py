"""Basic tests for RAG endpoints (workspaces, ask validation)."""
import secrets

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def mock_gemini_client(monkeypatch):
    """Mock Gemini client for RAG operations to avoid real API calls."""
    fake_store_name = f"stores/{secrets.token_hex(8)}"

    def mock_create_store(display_name):
        return type("Store", (), {"name": fake_store_name})()

    def mock_ask(store_name, question, metadata_filter, **kwargs):
        return "Mocked answer based on documents.", {"retrieved_contexts": []}

    def mock_plan(question, distinct_values, **kwargs):
        return {"company": None, "doc_types": None, "time_scope": None, "temperature": 0.2}

    monkeypatch.setattr("app.gemini_client.create_file_search_store", mock_create_store)
    monkeypatch.setattr("app.gemini_client.ask_with_file_search", mock_ask)
    monkeypatch.setattr("app.gemini_client.plan_filters", mock_plan)
    return {"store_name": fake_store_name}


def test_create_workspace(client, mock_gemini_client):
    """Test workspace creation endpoint."""
    resp = client.post("/workspaces", json={"name": "test-workspace"})
    assert resp.status_code == 201
    data = resp.json()
    assert "workspace_id" in data
    assert "file_search_store_name" in data
    # Store name format: fileSearchStores/... or stores/...
    assert "store" in data["file_search_store_name"].lower()


def test_ask_endpoint_validation(client, mock_gemini_client):
    """Test ask endpoint validates workspace existence."""
    # Non-existent workspace
    resp = client.post(
        "/workspaces/nonexistent/ask",
        json={"question": "test"},
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert "not found" in detail["message"].lower()
