# Secure Gemini RAG Backend

Backend-only FastAPI service that provisions Gemini File Search workspaces,
securely stores encrypted document blobs, and lets clients query them via
Retrieval-Augmented Generation with Gemini 2.5 Pro.

## Features

- **Workspace isolation**: each workspace maps to its own Gemini File Search store.
- **AES-256-GCM encryption**: file bytes are encrypted before persisting (no plaintext on disk).
- **Async SQLAlchemy**: metadata stored in SQLite (swap the `DATABASE_URL` for Postgres later).
- **Gemini 2.5 Pro + File Search**: upload documents once and chat over them with metadata filters.
- **Clean JSON APIs**: designed for Postman or a future frontend.

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

## 3. API Workflows

### 3.1 Create a workspace

```
POST /workspaces
Body: { "display_name": "secure-vc-deals" }
```

Response provides `workspace_id` and its Gemini `file_search_store_name`.

### 3.2 Upload files

```
POST /workspaces/{workspace_id}/files
Body (form-data): files=<file>, files=<file2>, ...
```

For each file the backend will:

1. Infer mime/doc types based on filename.
2. Encrypt bytes with AES-256-GCM and persist metadata only.
3. Upload plaintext once to Gemini File Search with chunking
   (`350` token chunks / `40` token overlap) and metadata tags
   (`workspace_id`, `doc_id`, `doc_type`).
4. Wait until indexing completes before returning.

Response lists imported documents (`doc_id`, `filename`, `doc_type`).

### 3.3 Ask questions

```
POST /workspaces/{workspace_id}/ask
Body:
{
  "question": "Summarize key commercial terms in all contracts",
  "metadata_filter": "doc_type=contract",
  "history": [
    {"role": "user", "text": "previous question"},
    {"role": "assistant", "text": "previous answer"}
  ]
}
```

Gemini responds with a grounded answer across the workspace's File Search store.

## 4. Security & Extensibility

- Files are encrypted at rest; only AES-256-GCM ciphertext + nonce stored in DB.
- No plaintext hits disk or logs (temporary files are ephemeral for Gemini uploads).
- Switching to a managed DB or adding auth only requires swapping the SQLAlchemy URL
  and adding middleware—API design already separates workspaces per tenant.
- Metadata filters (`doc_type=contract`, `doc_type=pitch_deck`, etc.) allow
  targeted retrieval without re-uploading documents.

## 5. Project Structure

```
main.py                # FastAPI entrypoint
app/
  __init__.py
  config.py            # Settings (env vars, AES key validation)
  crypto.py            # AES-256-GCM helpers
  db.py                # Async engine/session helpers
  gemini_client.py     # Gemini SDK wrappers
  models.py            # SQLAlchemy ORM models
  schemas.py           # Pydantic request/response models
  routers/
    workspaces.py      # Workspace creation + uploads
    chat.py            # RAG / ask endpoint
```

This layout keeps Gemini logic, crypto, DB access, and routers isolated so you can
easily extend the system (e.g., add auth, swap encryption keys per tenant, or move
to a message history store).
