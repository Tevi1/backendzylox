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
MEMO_HINTS = [r"memo", r"internal\s+note", r"minutes", r"retros?"]


def _matches(hints: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in hints)


def guess_doc_type(name: str) -> Optional[str]:
    low = name.lower()
    if _matches(CONTRACT_HINTS, low):
        return "contract"
    if _matches(DECK_HINTS, low):
        return "deck"
    if _matches(SECURITY_HINTS, low):
        return "security"
    if _matches(MEMO_HINTS, low):
        return "memo"
    return None


def chunk_preset_for_doc_type(doc_type: Optional[str]) -> Tuple[int, int]:
    presets = {
        "contract": (150, 50),
        "deck": (300, 30),
        "security": (220, 30),
        "memo": (220, 30),
    }
    return presets.get(doc_type, (200, 20))


def derive_metadata(path: str, filename: str, company_from_path: Optional[str]) -> dict:
    doc_type = guess_doc_type(filename)
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

