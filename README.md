# Zylox Backend - Gemini RAG + Chat API

FastAPI backend service providing two distinct Gemini-powered flows:
1. **RAG (Retrieval-Augmented Generation)**: Document-based Q&A using Gemini 2.5 Pro + File Search
2. **Gemini 3 Pro Chat**: Direct chat with multimodal support, structured outputs, and thought signatures

## Features

- **Dual Gemini Architecture**: Separate clients for RAG (2.5 Pro) and chat (3 Pro preview)
- **Workspace isolation**: Each workspace maps to its own Gemini File Search store
- **AES-256-GCM encryption**: File bytes encrypted at rest (no plaintext on disk)
- **Async SQLAlchemy**: Metadata stored in SQLite (swap `DATABASE_URL` for Postgres without code changes)
- **Metadata filtering**: Query documents by company, doc_type, time_scope, sensitivity
- **Thought signature persistence**: Gemini 3 conversations retain context across turns
- **Structured outputs**: JSON schema validation with optional tools (Google Search, URL Context, Code Execution)

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in the values:

```
APP_ENV=dev
GEMINI_API_KEY=your_gemini_key_here
FILE_ENC_KEY_B64=<openssl rand -base64 32>
DATABASE_URL=sqlite+aiosqlite:///./data.db
```

Notes:
- `GEMINI_API_KEY`: create in Google AI Studio (Gemini Developer API).
- `FILE_ENC_KEY_B64`: 32 random bytes (base64). Example: `openssl rand -base64 32`.
- `DATABASE_URL`: defaults to local SQLite; swap to Postgres (e.g. `postgresql+asyncpg://...`).

## 2. Run the API

```bash
uvicorn main:app --reload
```

CORS currently allows `http://localhost:5173` and Postman origins for dev use.

### Health check

```
GET http://localhost:8000/health
```

## 3. Architecture Overview

### 3.1 RAG Flow (Gemini 2.5 Pro)

**Purpose**: Document-based question answering with metadata filtering.

**Flow**:
1. Upload files → `app/services/ingestion_agent.py` encrypts, extracts metadata, uploads to File Search
2. Ask question → `app/services/retrieval.py` plans filters, calls `app/gemini_client.py` with File Search tool
3. Response → Grounded answer with citations from uploaded documents

**Endpoints**:
- `POST /workspaces` - Create workspace
- `POST /workspaces/{id}/files` - Upload documents
- `POST /workspaces/{id}/ask` - Query documents with RAG
- `GET /workspaces/{id}/diagnostics` - View metadata counts

### 3.2 Gemini 3 Pro Chat Flow

**Purpose**: Direct chat with multimodal support, structured outputs, conversation persistence.

**Flow**:
1. Send messages → `app/routers/gemini3.py` handles conversation state
2. Generate → `app/services/gemini3_client.py` calls Gemini 3 Pro API
3. Persist → Thought signatures and message history stored in `conversation_histories` table

**Endpoints**:
- `POST /api/v1/chat/gemini3` - Text + multimodal chat
- `POST /api/v1/chat/gemini3/structured` - JSON schema outputs with tools

## 4. API Workflows

### 4.1 RAG: Create workspace and upload files

```bash
POST /workspaces
Body: { "name": "secure-vc-deals" }

POST /workspaces/{workspace_id}/files
Body (form-data): files=<file1>, files=<file2>
```

Files are encrypted, metadata extracted (company, doc_type, time_scope), and uploaded to Gemini File Search.

### 4.2 RAG: Ask questions

```bash
POST /workspaces/{workspace_id}/ask
Body: {
  "question": "Summarize key commercial terms in all contracts",
  "filter": "doc_type=\"contract\""
}
```

Returns grounded answer with citations from uploaded documents.

### 4.3 Gemini 3: Chat

```bash
POST /api/v1/chat/gemini3
Body: {
  "messages": [{"role": "user", "parts": [{"text": "Hello"}]}],
  "conversation_id": "optional-id-for-persistence",
  "thinking_level": "high"
}
```

Supports text, images, PDFs, video with automatic media resolution handling.

### 4.4 Gemini 3: Structured outputs

```bash
POST /api/v1/chat/gemini3/structured
Body: {
  "prompt": "Extract key metrics",
  "jsonSchema": {"type": "object", "properties": {...}},
  "tools": {"google_search": true, "url_context": false}
}
```

## 5. Security & Extensibility

- **Encryption**: Files encrypted at rest (AES-256-GCM); only ciphertext + nonce in DB
- **No plaintext storage**: Temporary files are ephemeral for Gemini uploads only
- **Database flexibility**: SQLite by default; swap `DATABASE_URL` for Postgres without code changes
- **Workspace isolation**: Each workspace has its own File Search store and encrypted document blobs
- **Metadata filtering**: Query by `company`, `doc_type`, `time_scope`, `sensitivity` without re-uploading

## 6. Project Structure

```
main.py                    # FastAPI entrypoint with CORS
app/
  __init__.py              # Exports FastAPI app
  main.py                  # App factory, startup, health endpoint
  config.py                # Centralized config (Pydantic BaseSettings)
  db.py                    # Async SQLAlchemy engine + session factory
  models.py                # ORM models (Workspace, Document, ConversationHistory)
  schemas.py               # Pydantic request/response models
  gemini_client.py         # Gemini 2.5 Pro client (RAG/File Search)
  routers/
    workspaces.py          # RAG endpoints (create, upload, ask, diagnostics)
    gemini3.py             # Gemini 3 Pro endpoints (chat, structured)
  services/
    gemini3_client.py      # Gemini 3 Pro client wrapper
    retrieval.py           # RAG router with filter planning
    ingestion_agent.py     # File upload, encryption, File Search upload
    encryption.py          # AES-256-GCM helpers
  ingestion/
    heuristics.py          # Metadata extraction from filenames/paths
  prompts/
    system_prompt.py       # Classification prompts
    system_answer_style.py # RAG answer formatting
    retrieval_planner.py   # Filter planning prompt
```

**Key Design Decisions**:
- Two separate Gemini clients: RAG (2.5) vs Chat (3 Pro) - different APIs, different use cases
- Centralized config via `app/config.py` - no `os.getenv()` scattered across codebase
- Thin routers - business logic in services, routers handle HTTP layer only
- Consistent error handling - custom exceptions (GeminiRAGError, Gemini3Error) converted to HTTP responses
