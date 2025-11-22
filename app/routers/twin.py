"""Twin Agent endpoint: RAG (Gemini 2.5) + Reasoning (Gemini 3)."""
from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import ConversationHistory, Workspace
from ..prompts.twin_system_prompt import build_twin_system_prompt
from ..services.gemini3_client import Gemini3Client
from ..services.structured_retrieval import build_structured_context

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/twin", tags=["twin"])
router_v0 = APIRouter(prefix="/api/twin", tags=["twin"])  # Frontend compatibility


def normalize_parts(parts: List[any]) -> List[Dict[str, str]]:
    """
    Normalize message parts to valid Gemini format.
    
    Converts strings, dicts, or other objects into {"text": "..."} format.
    Ensures no unsupported payloads reach Gemini 3.
    
    Args:
        parts: List of parts (can be strings, dicts, or other objects)
    
    Returns:
        List of normalized dict parts with "text" key
    """
    normalized = []
    for part in parts:
        if isinstance(part, str):
            normalized.append({"text": part})
        elif isinstance(part, dict):
            # If it's already a dict, check if it has valid keys
            if "text" in part:
                normalized.append({"text": part["text"]})
            elif "thought_signature" in part:
                # Keep thought signatures as-is (they're handled separately)
                normalized.append({"thought_signature": part["thought_signature"]})
            else:
                # Convert unknown dict to text
                normalized.append({"text": str(part)})
        else:
            # Convert anything else to text
            normalized.append({"text": str(part)})
    return normalized


from pydantic import BaseModel, Field, model_validator
from typing import Literal


class TwinAskRequest(BaseModel):
    """Request model for Twin Agent. Accepts either 'question' (string) or 'messages' (array)."""
    workspace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    question: Optional[str] = None  # Frontend sends this
    messages: Optional[List[Dict[str, str]]] = None  # Alternative format
    thinking_level: str = "high"
    mode: Literal["docs_only", "advisor"] = "docs_only"  # Default to strict docs_only
    
    @model_validator(mode="after")
    def validate_question_or_messages(self):
        """Ensure either question or messages is provided."""
        if not self.question and not self.messages:
            raise ValueError("Either 'question' or 'messages' must be provided")
        return self


class TwinAskResponse(BaseModel):
    """Response model for Twin Agent."""
    text: str
    conversation_id: str
    used_retrieval: bool
    sources: Optional[List] = None
    usage: Optional[Dict] = None
    thought_signatures: Optional[List] = None


def get_gemini3_client() -> Gemini3Client:
    """Dependency to get Gemini 3 client."""
    return Gemini3Client()


@router.post("/ask", response_model=TwinAskResponse)
async def twin_ask(
    req: TwinAskRequest,
    db: AsyncSession = Depends(get_session),
    gemini3_client: Gemini3Client = Depends(get_gemini3_client),
):
    """
    Twin Agent endpoint: Combines Gemini 2.5 RAG retrieval with Gemini 3 reasoning.
    
    Request format:
    {
      "workspace_id": "string (optional)",
      "conversation_id": "string (optional)",
      "messages": [{"text": "user message"}],
      "thinking_level": "high|low (default: high)"
    }
    """
    LOGGER.info("=== Twin Ask Request Received ===")
    LOGGER.info("Request received at endpoint /ask")
    try:
        LOGGER.info("=== Twin Ask Request Started ===")
        LOGGER.info("Workspace ID: %s, Conversation ID: %s, Thinking Level: %s", 
                   req.workspace_id, req.conversation_id, req.thinking_level)
        
        # Create or reuse conversation
        conversation_id = req.conversation_id or str(uuid.uuid4())
        LOGGER.info("Using conversation_id: %s", conversation_id)
        
        # Extract user message - support both 'question' and 'messages' formats
        if req.question:
            user_msg = req.question.strip()
        elif req.messages and isinstance(req.messages, list) and len(req.messages) > 0:
            user_msg = req.messages[-1].get("text") if isinstance(req.messages[-1], dict) else str(req.messages[-1])
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "Either 'question' or 'messages' array is required", "code": "INVALID_REQUEST"},
            )
        
        if not user_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "Question or message text cannot be empty", "code": "INVALID_REQUEST"},
            )
        
        LOGGER.info("User message extracted (length=%d): %s", len(user_msg), user_msg[:100])

        # --- STEP 1: AUTO-DETECT IF WE NEED RAG -------------------------------
        LOGGER.info("STEP 1: Starting RAG retrieval check...")
        structured_context = None
        used_retrieval = False
        sources = None
        
        if req.workspace_id:
            LOGGER.info("Workspace ID provided, attempting RAG retrieval...")
            # Get workspace to access file_search_store_name
            workspace = await db.get(Workspace, req.workspace_id)
            if not workspace:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"message": "Workspace not found", "code": "WORKSPACE_NOT_FOUND"},
                )
            
            try:
                LOGGER.info("Calling build_structured_context...")
                # Build structured context from RAG retrieval (with doc_type boosting)
                # Add timeout to prevent hanging (2 minutes max for RAG)
                structured_context, used_retrieval = await asyncio.wait_for(
                    build_structured_context(
                        workspace,
                        user_msg,
                        None,  # metadata_filter
                        session=db,  # Pass session for diagnostics
                    ),
                    timeout=120.0  # 2 minutes
                )
                LOGGER.info("RAG retrieval completed: used_retrieval=%s", used_retrieval)
                
                if used_retrieval and structured_context:
                    sources = structured_context.sources
                    LOGGER.info(
                        "Structured context retrieved: msa=%d, security=%d, roadmap=%d, deck=%d, memos=%d, sources=%d",
                        len(structured_context.msa_clauses),
                        len(structured_context.security_excerpts),
                        len(structured_context.roadmap_items),
                        len(structured_context.deck_sections),
                        len(structured_context.memos_emails),
                        len(sources) if sources else 0,
                    )
            except asyncio.TimeoutError:
                LOGGER.warning("RAG retrieval timed out after 2 minutes, proceeding without evidence")
                # Continue without retrieval
            except Exception as exc:
                LOGGER.warning("RAG retrieval failed, proceeding without evidence: %s", exc, exc_info=True)
                # Continue without retrieval
        else:
            LOGGER.info("No workspace_id provided, skipping RAG retrieval")

        # --- STEP 2: BUILD FINAL PROMPT FOR GEMINI 3 ---------------------------
        LOGGER.info("STEP 2: Building prompt for Gemini 3...")
        LOGGER.info("Mode: %s", req.mode)
        
        # Check if user explicitly requests docs_only in question
        user_msg_lower = user_msg.lower()
        effective_mode = req.mode
        if "using only the provided documents" in user_msg_lower or "only the documents" in user_msg_lower:
            effective_mode = "docs_only"
            LOGGER.info("User explicitly requested docs_only, overriding mode")
        
        # Build system prompt with mode
        system_instruction = build_twin_system_prompt(effective_mode) if used_retrieval else None
        LOGGER.info("System instruction: %s (mode=%s)", "provided" if system_instruction else "none", effective_mode)
        
        # Build user message with structured context
        user_parts = []
        
        if used_retrieval and structured_context and structured_context.has_content():
            # Add structured sections
            context_sections = structured_context.to_prompt_sections()
            user_parts.append("\n".join(context_sections))
            # Add user question after context
            user_parts.append(f"\n## Section F: User Question\n{user_msg}")
        else:
            # No retrieval: just the question
            user_parts.append(user_msg)
        
        # Normalize parts to ensure valid format
        normalized_parts = normalize_parts(user_parts)
        
        gemini_messages = [
            {
                "role": "user",
                "parts": normalized_parts
            }
        ]
        
        # Fetch existing conversation history for thought signatures
        LOGGER.info("Fetching conversation history...")
        conversation = await db.execute(
            select(ConversationHistory).where(ConversationHistory.conversation_id == conversation_id)
        )
        existing_convo = conversation.scalars().first()
        stored_signatures = existing_convo.thought_signatures if existing_convo else []
        LOGGER.info("Found %d stored thought signatures", len(stored_signatures))
        
        # Inject thought signatures if available
        if stored_signatures:
            for sig in stored_signatures:
                gemini_messages[0]["parts"].append({"thought_signature": sig})  # Use underscore, not camelCase
            LOGGER.info("Injected %d thought signatures into messages", len(stored_signatures))

        # --- STEP 3: CALL GEMINI 3 --------------------------------------------
        LOGGER.info("STEP 3: Calling Gemini 3 API...")
        LOGGER.info("Gemini 3 params: messages=%d, thinking_level=%s, has_system_instruction=%s", 
                   len(gemini_messages), req.thinking_level, bool(system_instruction))
        
        try:
            # generate_text is synchronous, wrap in asyncio.to_thread with timeout
            def _call_gemini3():
                LOGGER.info("Gemini 3 API call starting (synchronous)...")
                try:
                    result = gemini3_client.generate_text(
                        messages=gemini_messages,
                        temperature=1.0,
                        thinking_level=req.thinking_level,
                        system_instruction=system_instruction,
                    )
                    LOGGER.info("Gemini 3 API call completed successfully")
                    return result
                except Exception as e:
                    LOGGER.error("Gemini 3 API call failed: %s", e, exc_info=True)
                    raise
            
            # Add timeout to prevent hanging (5 minutes max, but reduce for high thinking)
            timeout_seconds = 300.0  # 5 minutes default
            if req.thinking_level == "high":
                timeout_seconds = 180.0  # 3 minutes for high thinking (can be slow)
                LOGGER.info("Using reduced timeout (180s) for high thinking level")
            
            LOGGER.info("Starting Gemini 3 call with timeout=%ds...", timeout_seconds)
            start_time = time.time()
            
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(_call_gemini3),
                    timeout=timeout_seconds
                )
                elapsed = time.time() - start_time
                LOGGER.info("Gemini 3 response received in %.2fs, type=%s", elapsed, type(response).__name__)
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                LOGGER.error("Gemini 3 call timed out after %.2fs (timeout=%ds)", elapsed, timeout_seconds)
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={"message": f"Gemini 3 API call timed out after {timeout_seconds}s", "code": "TIMEOUT"},
                )
            
            # Extract text using the same pattern as gemini3 router
            answer_text = ""
            text = getattr(response, "text", None)
            if text:
                answer_text = text
                LOGGER.info("Extracted text from response.text (length=%d)", len(answer_text))
            else:
                # Try candidates
                collected: List[str] = []
                for candidate in getattr(response, "candidates", []) or []:
                    content = getattr(candidate, "content", None)
                    if not content:
                        continue
                    for part in getattr(content, "parts", []) or []:
                        part_text = getattr(part, "text", None)
                        if part_text:
                            collected.append(part_text)
                if collected:
                    answer_text = "\n".join(collected)
                    LOGGER.info("Extracted text from candidates (length=%d)", len(answer_text))
            
            # If still no text, log warning and set default
            if not answer_text:
                LOGGER.warning("No text extracted from Gemini 3 response. Response structure: %s", dir(response))
                answer_text = "I apologize, but I couldn't generate a response. Please try again."
            
            # Extract thought signatures using the same pattern as gemini3 router
            thought_signatures = []
            try:
                for candidate in getattr(response, "candidates", []) or []:
                    content = getattr(candidate, "content", None)
                    if not content:
                        continue
                    for part in getattr(content, "parts", []) or []:
                        signature = getattr(part, "thought_signature", None)
                        if signature:
                            if isinstance(signature, bytes):
                                signature = base64.b64encode(signature).decode("utf-8")
                            thought_signatures.append(str(signature))
            except Exception as sig_exc:
                LOGGER.warning("Failed to extract thought signatures: %s", sig_exc)
            
            if not thought_signatures:
                thought_signatures = ["context_engineering_is_the_way_to_go"]
            
            # Extract usage
            usage = {}
            try:
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    if hasattr(response.usage_metadata, "model_dump"):
                        usage = response.usage_metadata.model_dump()
                    elif hasattr(response.usage_metadata, "to_dict"):
                        usage = response.usage_metadata.to_dict()
                    else:
                        # Try to get common fields
                        usage = {
                            "prompt_token_count": getattr(response.usage_metadata, "prompt_token_count", 0),
                            "candidates_token_count": getattr(response.usage_metadata, "candidates_token_count", 0),
                        }
            except Exception as usage_exc:
                LOGGER.warning("Failed to extract usage metadata: %s", usage_exc)
            
            result = {
                "text": answer_text,
                "thought_signatures": thought_signatures,
                "usage": usage,
            }
            
        except Exception as exc:
            LOGGER.exception("Gemini 3 generation failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"message": f"Gemini 3 generation failed: {str(exc)}", "code": "GEMINI3_ERROR"},
            ) from exc

        # --- STEP 4: SAVE HISTORY ----------------------------------------------
        LOGGER.info("STEP 4: Saving conversation history...")
        try:
            history_messages = existing_convo.messages if existing_convo else []
            history_messages.append({"role": "user", "parts": [{"text": user_msg}]})
            history_messages.append({"role": "assistant", "parts": [{"text": result["text"]}]})
            
            if existing_convo:
                existing_convo.messages = history_messages
                existing_convo.thought_signatures = result["thought_signatures"]
            else:
                new_convo = ConversationHistory(
                    conversation_id=conversation_id,
                    messages=history_messages,
                    thought_signatures=result["thought_signatures"],
                )
                db.add(new_convo)
            
            await db.commit()
            LOGGER.info("Conversation history saved successfully")
        except Exception as exc:
            LOGGER.warning("Failed to save conversation history: %s", exc, exc_info=True)
            # Continue even if history save fails

        # --- STEP 5: CLEAN RESPONSE --------------------------------------------
        LOGGER.info("STEP 5: Preparing response...")
        # Ensure we have a valid response
        if not result.get("text"):
            LOGGER.error("Empty response text after Gemini 3 call")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"message": "Gemini 3 returned an empty response", "code": "EMPTY_RESPONSE"},
            )
        
        LOGGER.info("=== Twin Ask Request Completed Successfully ===")
        LOGGER.info("Response: text_length=%d, used_retrieval=%s, sources=%d", 
                   len(result["text"]), used_retrieval, len(sources) if sources else 0)
        
        return TwinAskResponse(
            text=result["text"],
            conversation_id=conversation_id,
            used_retrieval=used_retrieval,
            sources=sources,
            usage=result.get("usage"),
            thought_signatures=result.get("thought_signatures"),
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as exc:
        LOGGER.exception("Unexpected error in twin_ask endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Internal server error: {str(exc)}", "code": "INTERNAL_ERROR"},
        ) from exc


@router_v0.post("/ask", response_model=TwinAskResponse)
async def twin_ask_v0(
    req: TwinAskRequest,
    db: AsyncSession = Depends(get_session),
    gemini3_client: Gemini3Client = Depends(get_gemini3_client),
):
    """Twin Agent endpoint (v0 - for frontend compatibility at /api/twin/ask)."""
    return await twin_ask(req, db, gemini3_client)


