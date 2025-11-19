"""Lightweight intent detection helpers for adaptive responses."""
from __future__ import annotations

import re

CLAUSE_HINTS = [
    r"\bmsa\b",
    r"\bclause\b",
    r"\buptime\b|\bsla\b|\bp[12]\b",
    r"liabilit",
    r"indemn",
    r"terminate|termination",
]

COMPARE_HINTS = [
    r"compare",
    r"\bvs\.\b",
    r"\bversus\b",
    r"which (company|vendor)",
    r"across",
    r"between",
]

TABLE_HINTS = [
    r"\btable\b",
    r"\bgrid\b",
    r"\bmatrix\b",
    r"\bkpi\b",
    r"\bmetrics\b",
]

SUMMARY_HINTS = [
    r"summarise",
    r"summarize",
    r"\btl;dr\b",
    r"executive summary",
]

PLAN_HINTS = [
    r"\bplan\b",
    r"roadmap",
    r"next\s+(30|60|90)",
    r"actions?",
    r"checklist",
]


def _matches(hints: list[str], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in hints)


def detect_intent(question: str) -> str:
    """Return a coarse intent label for the given user question."""

    lowered = (question or "").lower()
    if _matches(CLAUSE_HINTS, lowered):
        return "clauses"
    if _matches(COMPARE_HINTS, lowered):
        return "compare"
    if _matches(TABLE_HINTS, lowered):
        return "table"
    if _matches(SUMMARY_HINTS, lowered):
        return "summary"
    if _matches(PLAN_HINTS, lowered):
        return "plan"
    return "chat"

