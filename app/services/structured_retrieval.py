"""Structured retrieval for Twin Agent - groups evidence by doc_type."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from ..gemini_client import ask_with_file_search
from ..models import Workspace
from .retrieval import get_boosted_doc_types, get_retrieval_strategy, is_comparison_question

LOGGER = logging.getLogger(__name__)


class StructuredContext:
    """Structured context grouped by doc_type for Gemini 3 reasoning."""

    def __init__(self):
        self.msa_clauses: List[str] = []
        self.security_excerpts: List[str] = []
        self.roadmap_items: List[str] = []
        self.deck_sections: List[str] = []
        self.memos_emails: List[str] = []
        self.sources: List[Dict[str, any]] = []

    def add_chunk(self, doc_type: Optional[str], text: str, source_meta: Dict[str, any]) -> None:
        """Add a retrieved chunk to the appropriate section."""
        if not text or not text.strip():
            return

        # Normalize doc_type
        doc_type_lower = (doc_type or "").lower()

        # Add to appropriate section
        if doc_type_lower == "contract":
            self.msa_clauses.append(text.strip())
        elif doc_type_lower == "security":
            self.security_excerpts.append(text.strip())
        elif doc_type_lower == "roadmap":
            self.roadmap_items.append(text.strip())
        elif doc_type_lower == "deck":
            self.deck_sections.append(text.strip())
        elif doc_type_lower in ("memo", "email"):
            self.memos_emails.append(text.strip())
        else:
            # Default: add to memos if unknown
            self.memos_emails.append(text.strip())

        # Add source metadata
        self.sources.append(source_meta)

    def to_prompt_sections(self) -> List[str]:
        """Convert structured context to prompt sections for Gemini 3."""
        sections = []

        if self.msa_clauses:
            sections.append("## Section A: MSA Clauses\n" + "\n\n".join(self.msa_clauses[:10]))  # Limit to top 10

        if self.security_excerpts:
            sections.append("## Section B: Security Architecture Excerpts\n" + "\n\n".join(self.security_excerpts[:10]))

        if self.roadmap_items:
            sections.append("## Section C: Roadmap & Risk Items\n" + "\n\n".join(self.roadmap_items[:10]))

        if self.deck_sections:
            sections.append("## Section D: Pitch Deck Sections\n" + "\n\n".join(self.deck_sections[:10]))

        if self.memos_emails:
            sections.append("## Section E: Memos & Investor Emails\n" + "\n\n".join(self.memos_emails[:10]))

        return sections

    def has_content(self) -> bool:
        """Check if any sections have content."""
        return bool(
            self.msa_clauses
            or self.security_excerpts
            or self.roadmap_items
            or self.deck_sections
            or self.memos_emails
        )


async def build_structured_context(
    workspace: Workspace,
    question: str,
    metadata_filter: Optional[str] = None,
    session: Optional[AsyncSession] = None,
) -> Tuple[StructuredContext, bool]:
    """
    Build structured context from RAG retrieval, grouped by doc_type.

    Args:
        workspace: Workspace containing indexed documents.
        question: User question.
        metadata_filter: Optional metadata filter.
        session: Optional database session for diagnostics.

    Returns:
        Tuple of (StructuredContext, used_retrieval_bool).
    """
    context = StructuredContext()
    used_retrieval = False

    LOGGER.info("build_structured_context: Starting for workspace=%s, question_length=%d", 
               workspace.id, len(question))
    
    try:
        # Enhance filter for complex/comparison questions
        enhanced_filter = metadata_filter
        LOGGER.info("build_structured_context: Initial filter=%s", enhanced_filter)
        if not enhanced_filter:
            # Use new retrieval strategy
            strategy = get_retrieval_strategy(question)
            boosted_doc_types = strategy.get("boost_doc_types", [])
            strategy_type = strategy.get("strategy")
            
            if boosted_doc_types or strategy_type == "multi_doc":
                # Get available doc_types from workspace if session provided
                if session:
                    try:
                        from .retrieval import _distinct_values
                        # Add timeout for diagnostics query (10 seconds)
                        diagnostics = await asyncio.wait_for(
                            _distinct_values(session, workspace.id),
                            timeout=10.0
                        )
                        available_doc_types = diagnostics.get("doc_type", [])
                        
                        # For multi_doc strategy (risk register, comparison), get all available doc_types
                        if strategy_type == "multi_doc":
                            # Force retrieval from all available doc_types
                            relevant_types = [dt for dt in boosted_doc_types if dt in available_doc_types]
                            if not relevant_types:
                                # Fallback: get all available doc_types
                                relevant_types = available_doc_types[:5]  # Limit to 5 to avoid overly complex filter
                            if relevant_types:
                                enhanced_filter = " OR ".join(f'doc_type="{dt}"' for dt in relevant_types)
                                LOGGER.info("build_structured_context: Using multi_doc strategy with types: %s", relevant_types)
                        elif boosted_doc_types:
                            # Use boosted doc_types
                            relevant_boosted = [dt for dt in boosted_doc_types if dt in available_doc_types]
                            if relevant_boosted:
                                enhanced_filter = " OR ".join(f'doc_type="{dt}"' for dt in relevant_boosted)
                                LOGGER.info("build_structured_context: Using boosted doc_types: %s", relevant_boosted)
                    except asyncio.TimeoutError:
                        LOGGER.warning("Diagnostics query timed out, proceeding without doc_type boost")
        
        # Call Gemini 2.5 RAG retrieval with timeout (90 seconds)
        LOGGER.info("build_structured_context: Calling ask_with_file_search (store=%s, filter=%s)", 
                   workspace.file_search_store_name, enhanced_filter)
        start_time = time.time()
        
        retrieved_text, grounding_metadata = await asyncio.wait_for(
            asyncio.to_thread(
                ask_with_file_search,
                workspace.file_search_store_name,
                question,
                enhanced_filter,
            ),
            timeout=90.0  # 90 seconds for RAG retrieval
        )
        
        elapsed = time.time() - start_time
        LOGGER.info("build_structured_context: RAG retrieval completed in %.2fs, has_text=%s, has_metadata=%s", 
                   elapsed, bool(retrieved_text), bool(grounding_metadata))

        if not retrieved_text or not grounding_metadata:
            return context, False

        used_retrieval = True

        # Extract chunks from grounding metadata
        retrieved_contexts = grounding_metadata.get("retrieved_contexts", [])
        if not retrieved_contexts:
            # Fallback: use retrieved_text as a single chunk
            context.add_chunk(None, retrieved_text, {"file_name": "Unknown", "text_preview": retrieved_text[:200]})
            return context, used_retrieval

        # Process each retrieved chunk
        for ctx in retrieved_contexts[:15]:  # Limit to top 15 chunks
            chunk_data = ctx.get("chunk", {})
            text = chunk_data.get("text", "")
            if not text:
                continue

            # Extract metadata
            file_name = chunk_data.get("file_name", "Unknown")
            page = chunk_data.get("page")
            doc_type = None

            # Try to extract doc_type from chunk metadata
            chunk_metadata = chunk_data.get("custom_metadata", [])
            if isinstance(chunk_metadata, list):
                for meta in chunk_metadata:
                    if isinstance(meta, dict) and meta.get("key") == "doc_type":
                        doc_type = meta.get("string_value")
                        break
            elif isinstance(chunk_metadata, dict):
                # Handle dict format
                doc_type = chunk_metadata.get("doc_type")
            
            # Fallback: try to infer from file name if not found
            if not doc_type and file_name:
                from ..ingestion.heuristics import guess_doc_type_from_filename
                doc_type = guess_doc_type_from_filename(file_name)

            # Build source metadata
            source_meta = {
                "doc_type": doc_type,
                "file_name": file_name,
                "page": page,
                "text_preview": text[:200] if text else None,
            }

            # Add chunk to structured context
            context.add_chunk(doc_type, text, source_meta)

        LOGGER.info(
            "Structured context built: msa=%d, security=%d, roadmap=%d, deck=%d, memos=%d",
            len(context.msa_clauses),
            len(context.security_excerpts),
            len(context.roadmap_items),
            len(context.deck_sections),
            len(context.memos_emails),
        )

    except asyncio.TimeoutError:
        LOGGER.warning("Structured context retrieval timed out")
        used_retrieval = False
    except Exception as exc:
        LOGGER.warning("Structured context retrieval failed: %s", exc)
        # Return empty context, but mark as attempted
        used_retrieval = False

    return context, used_retrieval

