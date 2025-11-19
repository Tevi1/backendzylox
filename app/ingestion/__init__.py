"""Ingestion utilities (heuristics, parsers, etc.)."""

from .heuristics import derive_metadata, guess_doc_type_from_filename

__all__ = ["derive_metadata", "guess_doc_type_from_filename"]

