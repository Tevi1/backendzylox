"""Simple keyword-based router agent for metadata scoping."""
from __future__ import annotations

from typing import Dict, List, Optional


def route_query(question: str, companies: List[str]) -> Dict[str, object]:
    q = (question or "").lower()

    matched = [c for c in companies if c and c.lower() in q]

    if len(matched) == 1:
        return {"company": matched[0]}
    if len(matched) > 1:
        return {"companies": matched}

    portfolio_keywords = ["portfolio", "all companies", "compare", "across"]
    if any(keyword in q for keyword in portfolio_keywords):
        return {"company": None}

    return {"company": None}

