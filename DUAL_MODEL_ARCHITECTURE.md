# Dual-Model Architecture: Gemini 2.5 + Gemini 3

## Overview

Zylox now uses a **dual-model architecture** where:
- **Gemini 2.5 Pro**: Handles all ingestion and retrieval (RAG)
- **Gemini 3 Pro**: Handles reasoning, personalized responses, and thought signatures

## Architecture Flow

### Document-Related Questions
```
User Question → Gemini 2.5 (Retrieve) → Gemini 3 (Reason) → Final Answer
```

1. **Retrieval (Gemini 2.5)**: 
   - Queries File Search store
   - Retrieves relevant document chunks
   - Extracts metadata and sources

2. **Reasoning (Gemini 3)**:
   - Receives retrieved evidence + question
   - Performs deep reasoning
   - Generates personalized answer
   - Maintains thought signatures

### General Questions
```
User Question → Gemini 3 (Reason) → Final Answer
```

- No retrieval needed
- Direct reasoning with Gemini 3
- Thought signatures maintained

## Endpoints

### 1. Twin Agent (Recommended)
**`POST /api/v1/twin/ask`**

Combined RAG + Reasoning pipeline. Automatically detects if retrieval is needed.

**Request:**
```json
{
  "question": "What are the key terms in the contract?",
  "workspace_id": "abc123",
  "conversation_id": "conv-xyz",
  "thinking_level": "high",
  "temperature": 1.0,
  "filter": "doc_type=\"contract\""
}
```

**Response:**
```json
{
  "answer": "Based on the retrieved evidence...",
  "thought_signatures": ["sig1", "sig2"],
  "retrieved_evidence": true,
  "sources": [
    {
      "file_name": "contract.pdf",
      "page": 5,
      "text_preview": "..."
    }
  ],
  "raw": {...}
}
```

### 2. Legacy RAG Endpoint (Backward Compatible)
**`POST /workspaces/{id}/ask`**

Still uses Gemini 2.5 only. Maintained for backward compatibility.

**Request:**
```json
{
  "question": "What are the key terms?",
  "filter": "doc_type=\"contract\""
}
```

**Response:**
```json
{
  "answer": "Answer from Gemini 2.5...",
  "scope": "...",
  "grounding_metadata": {...}
}
```

### 3. Direct Gemini 3 Chat
**`POST /api/v1/chat/gemini3`**

Pure Gemini 3 chat without retrieval. Use for general conversation.

## Question Routing Logic

The Twin Agent automatically determines if retrieval is needed based on:

1. **Document Keywords**: contract, memo, email, document, file, report, etc.
2. **Metadata References**: company, doc_type, time_scope, sensitivity
3. **Workspace Context**: If `workspace_id` is provided and workspace exists

**Examples:**
- ✅ "What are the key terms in the contract?" → **Retrieval + Reasoning**
- ✅ "Summarize all memos from Acme Corp" → **Retrieval + Reasoning**
- ❌ "Explain quantum computing" → **Direct Reasoning** (no retrieval)
- ❌ "What is the weather today?" → **Direct Reasoning** (no retrieval)

## Key Features

### 1. Gemini 2.5 (RAG)
- ✅ File Search ingestion
- ✅ Document chunking and indexing
- ✅ Metadata extraction
- ✅ Document classification
- ✅ Retrieval with metadata filters

### 2. Gemini 3 (Reasoning)
- ✅ Deep reasoning and analysis
- ✅ Personalized writing style
- ✅ Thought signature persistence
- ✅ Multi-step thinking
- ✅ Low hallucination (when grounded)

### 3. Combined Pipeline
- ✅ Automatic routing (retrieval vs direct)
- ✅ Evidence formatting for Gemini 3
- ✅ Source citations
- ✅ Conversation persistence
- ✅ Thought signature continuity

## Implementation Details

### Files Created
- `app/services/twin_agent.py` - Core pipeline logic
- `app/routers/twin.py` - Twin Agent endpoint
- `app/schemas.py` - Added `TwinAgentRequest` and `TwinAgentResponse`

### Files Modified
- `app/main.py` - Added twin router
- `app/routers/__init__.py` - Exported twin router
- `app/config.py` - Made RAG model configurable (optional)

### Files Unchanged (Backward Compatible)
- `app/gemini_client.py` - Still uses Gemini 2.5
- `app/services/retrieval.py` - Still uses Gemini 2.5
- `app/routers/workspaces.py` - Legacy endpoints unchanged
- `app/routers/gemini3.py` - Direct chat unchanged

## Migration Guide

### For Frontend Developers

**Old Way (Gemini 2.5 only):**
```javascript
POST /workspaces/{id}/ask
```

**New Way (Recommended - Twin Agent):**
```javascript
POST /api/v1/twin/ask
{
  "question": "...",
  "workspace_id": "...",
  "conversation_id": "..." // Optional, for persistence
}
```

**Benefits:**
- Deeper reasoning
- Personalized responses
- Thought signature persistence
- Better source citations
- Lower hallucination

### For Backend Developers

The architecture is **fully backward compatible**. Existing endpoints continue to work:

- ✅ `/workspaces/{id}/ask` - Still works (Gemini 2.5)
- ✅ `/api/v1/chat/gemini3` - Still works (Gemini 3 direct)
- ✅ `/workspaces` - Still works (ingestion with Gemini 2.5)

## Testing

Test the Twin Agent:
```bash
curl -X POST http://localhost:8000/api/v1/twin/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key terms in contracts?",
    "workspace_id": "your-workspace-id",
    "conversation_id": "test-conv-1",
    "thinking_level": "high"
  }'
```

## Acceptance Criteria ✅

- ✅ Gemini 2.5 handles ALL ingestion and retrieval
- ✅ Gemini 3 handles ALL final responses and chat reasoning
- ✅ Document-related queries flow: 2.5 → retrieve → 3.0 → answer
- ✅ General reasoning queries flow: Gemini 3 only
- ✅ Frontend always receives the final Gemini 3 output
- ✅ Thought signatures persist across turns
- ✅ Zero breaking changes to existing API routes

