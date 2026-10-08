"""The agent's tools — thin wrappers over :mod:`topspin_review.storage.store`.

Plain callables; the backend agent builder decorates them (auto-extracting the
name, description and input schema from the function/docstring/signature).
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from topspin_review.domain import report as report_schema
from topspin_review.storage import store


def _parse(raw: Any) -> Any:
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        return json.loads(text)
    raise ValueError(f"unsupported payload type: {type(raw)!r}")


def get_profile() -> str:
    """Return the player profile as JSON (sport, level, dominant_hand, goal)."""
    return json.dumps(store.get_profile(), ensure_ascii=False)


def recent_reports(n: int = 5) -> str:
    """Return the most recent saved coaching reports as a JSON array.

    Args:
        n: How many recent reports to return (default 5).
    """
    return json.dumps(store.recent_reports(int(n or 5)), ensure_ascii=False)


def save_report(report_json: str) -> str:
    """Save the coaching report.

    Args:
        report_json: JSON object string with keys date, sport, summary, strengths,
            issues (array of {issue, evidence_times, confidence}), drills, focus,
            progress, limitations.
    """
    raw = _parse(report_json)
    if not isinstance(raw, dict):
        return "report must be a JSON object"
    raw.setdefault("date", date.today().isoformat())
    saved = report_schema.normalize(raw)
    store.add_report(saved)
    problems = report_schema.validate(saved)
    return "report saved" + (f" (warnings: {'; '.join(problems)})" if problems else "")


ALL_TOOLS = [get_profile, recent_reports, save_report]
