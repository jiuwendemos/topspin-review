"""Lexical retrieval over past reports to add relevant context.

A dependency-free stand-in for a vector store: rank prior reports by token
overlap with the current query/profile and feed the best few into the report
prompt. Swap :func:`relevant` for an embedding-backed retriever later without
changing callers.
"""

from __future__ import annotations

from topspin_review.domain import report as report_schema


def _tokens(text: str) -> set[str]:
    return {t for t in (text or "").lower().split() if len(t) > 2}


def _report_text(item: dict) -> str:
    parts = list(report_schema.issue_texts(item))
    parts += [str(s) for s in (item.get("strengths") or [])]
    parts.append(str(item.get("focus", "")))
    return " ".join(parts)


def score(item: dict, query_tokens: set[str]) -> int:
    return len(_tokens(_report_text(item)) & query_tokens)


def relevant(reports: list[dict], query: str, k: int = 3) -> list[dict]:
    """The top-``k`` prior reports that overlap ``query`` (most relevant first)."""
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    scored = [(score(r, query_tokens), r) for r in reports]
    picked = [(s, r) for s, r in scored if s > 0]
    picked.sort(key=lambda pair: pair[0], reverse=True)
    return [r for _, r in picked[:k]]


def context_text(reports: list[dict], query: str, k: int = 3) -> str:
    """Render the relevant past sessions as a compact context block (or '')."""
    picks = relevant(reports, query, k)
    if not picks:
        return ""
    lines = []
    for item in picks:
        issues = "; ".join(report_schema.issue_texts(item)[:4]) or "-"
        lines.append(f"- {item.get('source', '?')}: focus='{item.get('focus', '')}'; issues: {issues}")
    return "\n".join(lines)
