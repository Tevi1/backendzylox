"""Filename/path heuristics to derive metadata during ingestion."""
from __future__ import annotations

import re
from typing import Optional, Tuple

CONTRACT_HINTS = [
    r"\bmsa\b",
    r"master\s+services?\s+agreement",
    r"\bagreement\b",
    r"\bcontract\b",
    r"\bterms?\b",
    r"\bsla\b",
    r"service\s+levels?",
]

DECK_HINTS = [r"\bdeck\b", r"slides?", r"presentation", r"pitch"]
SECURITY_HINTS = [r"security", r"encryption", r"controls?", r"iso\s*27001", r"soc\s*2", r"rbac", r"egress"]
ROADMAP_HINTS = [r"roadmap", r"plan", r"risks?", r"mitigation", r"strategy", r"q[1-4]", r"fy\d+"]
MEMO_HINTS = [r"memo", r"internal\s+note", r"minutes", r"retros?", r"email"]


def _matches(hints: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in hints)


def guess_doc_type_from_filename(name: str) -> Optional[str]:
    """Guess doc_type from filename patterns."""
    low = name.lower()
    if _matches(CONTRACT_HINTS, low):
        return "contract"
    if _matches(DECK_HINTS, low):
        return "deck"
    if _matches(SECURITY_HINTS, low):
        return "security"
    if _matches(ROADMAP_HINTS, low):
        return "roadmap"
    if _matches(MEMO_HINTS, low):
        return "memo"
    return None


def chunk_preset_for_doc_type(doc_type: Optional[str]) -> Tuple[int, int]:
    """
    Return (chunk_tokens, overlap_tokens) for a given doc_type.
    
    Optimized for clause-level and section-level retrieval:
    - contract: Small chunks (140) for clause-level granularity
    - security: Medium chunks (180) for section coherence
    - roadmap: Medium chunks (200) for narrative balance
    - memo: Medium chunks (220) for complete thoughts
    - deck: Larger chunks (300) per slide group/section
    """
    presets = {
        "contract": (140, 50),  # Clause-level granularity for MSA/legal text
        "security": (180, 40),  # Section-level coherence for architecture docs
        "roadmap": (200, 40),   # Narrative balance for roadmap & risk docs
        "memo": (220, 40),      # Complete thoughts for memos/emails
        "deck": (300, 30),      # Slide group/section for pitch decks
    }
    return presets.get(doc_type, (200, 40))  # Default: medium chunk with overlap


def derive_metadata(path: str, filename: str, company_from_path: Optional[str]) -> dict:
    doc_type = guess_doc_type_from_filename(filename)
    time_scope = "current"
    combined = f"{path}/{filename}".lower()
    if any(term in combined for term in ["roadmap", "plan", "q1", "q2", "q3", "q4", "fy"]):
        time_scope = "future_plan"
    if any(term in combined for term in ["retro", "postmortem", "histor", "archive"]):
        time_scope = "historic"

    return {
        "company": company_from_path,
        "doc_type": doc_type,
        "time_scope": time_scope,
        "sensitivity": "confidential",
    }

