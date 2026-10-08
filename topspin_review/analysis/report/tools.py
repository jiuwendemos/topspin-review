"""The agent's tools — thin wrappers over :mod:`topspin_review.storage.store`.

Plain callables plus a :class:`~topspin_review.backend.ToolSpec`; the backend agents
file decorates them into openjiuwen tools.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from topspin_review.backend import ToolSpec
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
    """Return the most recent saved coaching reports as a JSON array (for tracking progress)."""
    return json.dumps(store.recent_reports(int(n or 5)), ensure_ascii=False)


def save_report(report_json: str) -> str:
    """Save the coaching report (JSON object: date, sport, summary, strengths,
    issues[{issue, evidence_times, confidence}], drills, focus, progress, limitations)."""
    raw = _parse(report_json)
    if not isinstance(raw, dict):
        return "report must be a JSON object"
    raw.setdefault("date", date.today().isoformat())
    saved = report_schema.normalize(raw)
    store.add_report(saved)
    problems = report_schema.validate(saved)
    return "report saved" + (f" (warnings: {'; '.join(problems)})" if problems else "")


ALL_TOOLS = [
    ToolSpec(
        get_profile,
        name="get_profile",
        description="Return the player profile as JSON (sport, level, dominant_hand, goal).",
        input_params={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        recent_reports,
        name="recent_reports",
        description="Return the most recent saved coaching reports as a JSON array (for tracking progress).",
        input_params={
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "How many recent reports (default 5)."}},
            "required": [],
        },
    ),
    ToolSpec(
        save_report,
        name="save_report",
        description=(
            "Save the coaching report. report_json: an object with keys "
            "date (string), sport (string), summary (string), strengths (array of strings), "
            "issues (array of {issue: string, evidence_times: [number], confidence: 'high'|'medium'|'low'}), "
            "drills (array of strings), focus (string), progress (string), limitations (array of strings)."
        ),
        input_params={
            "type": "object",
            "properties": {"report_json": {"type": "string", "description": "JSON object string."}},
            "required": ["report_json"],
        },
    ),
]
