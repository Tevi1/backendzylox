# Gemini Model Migration Guide

## Current Architecture

Your backend uses **two separate Gemini models** for different purposes:

### 1. RAG System (Document Q&A)
- **Model**: Gemini 2.5 Pro (default)
- **Purpose**: Answer questions based on indexed documents
- **Features**: File Search tool integration, metadata filtering
- **Endpoints**: `/workspaces/{id}/ask`

### 2. Chat System (Direct Conversation)
- **Model**: Gemini 3 Pro Preview
- **Purpose**: Direct chat, multimodal, structured outputs
- **Features**: Thought signatures, media support, JSON schema outputs
- **Endpoints**: `/api/v1/chat/gemini3`

## Why Two Models?

These serve **different use cases**:
- **RAG**: Query your uploaded documents with citations
- **Chat**: General conversation, image analysis, structured data extraction

## Option 1: Keep Both (Recommended)

**Best for**: Production systems that need both document Q&A and general chat.

**Pros**:
- Each model optimized for its use case
- Gemini 2.5 Pro is stable and proven for RAG
- Gemini 3 Pro offers latest features for chat

**Cons**:
- Two different models to maintain
- Slightly more complex architecture

## Option 2: Use Gemini 3 for RAG

**Best for**: If you want consistency and Gemini 3 supports File Search.

**To enable**:
1. Set environment variable:
   ```bash
   RAG_MODEL=gemini-3-pro-preview
   ```

2. **Important**: Verify that Gemini 3 Pro supports File Search tool. If it doesn't, RAG queries will fail.

**Pros**:
- Single model for all operations
- Latest model features

**Cons**:
- Gemini 3 is preview (may be less stable)
- File Search support may not be available yet
- May have different pricing/rate limits

## Option 3: Remove Gemini 3 Chat

**Best for**: If you only need document Q&A, not general chat.

**To remove**:
1. Remove `app/routers/gemini3.py` router from `app/main.py`
2. Delete `app/services/gemini3_client.py`
3. Remove Gemini 3 endpoints from frontend

**Pros**:
- Simpler architecture
- One model to maintain

**Cons**:
- Lose chat, multimodal, and structured output features
- No thought signature persistence

## Testing Gemini 3 with File Search

To test if Gemini 3 Pro supports File Search:

1. Set `RAG_MODEL=gemini-3-pro-preview` in your `.env`
2. Restart the backend
3. Try a RAG query: `POST /workspaces/{id}/ask`
4. If it works, Gemini 3 supports File Search!
5. If it fails with "tool not supported", stick with Gemini 2.5

## Recommendation

**Keep both models** unless you have a specific reason to change:
- RAG with Gemini 2.5 Pro (stable, proven)
- Chat with Gemini 3 Pro (latest features)

This gives you the best of both worlds without risking RAG functionality.

