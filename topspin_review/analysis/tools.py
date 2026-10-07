"""The agent's tools — thin wrappers over :mod:`topspin_review.store`."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from openjiuwen.core.foundation.tool import tool

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


@tool(
    name="get_profile",
    description="Return the player profile as JSON (sport, level, dominant_hand, goal).",
    input_params={"type": "object", "properties": {}, "required": []},
)
def get_profile() -> str:
    return json.dumps(store.get_profile(), ensure_ascii=False)


@tool(
    name="recent_reports",
    description="Return the most recent saved coaching reports as a JSON array (for tracking progress).",
    input_params={
        "type": "object",
        "properties": {"n": {"type": "integer", "description": "How many recent reports (default 5)."}},
        "required": [],
    },
)
def recent_reports(n: int = 5) -> str:
    return json.dumps(store.recent_reports(int(n or 5)), ensure_ascii=False)


@tool(
    name="save_report",
    description=(
        "Save the coaching report. report_json: an object with keys "
        "date (string), sport (string), summary (string), strengths (array of strings), "
        "issues (array of strings), drills (array of strings), focus (string)."
    ),
    input_params={
        "type": "object",
        "properties": {"report_json": {"type": "string", "description": "JSON object string."}},
        "required": ["report_json"],
    },
)
def save_report(report_json: str) -> str:
    raw = _parse(report_json)
    if not isinstance(raw, dict):
        return "report must be a JSON object"
    raw.setdefault("date", date.today().isoformat())
    saved = report_schema.normalize(raw)
    store.add_report(saved)
    problems = report_schema.validate(saved)
    return "report saved" + (f" (warnings: {'; '.join(problems)})" if problems else "")


ALL_TOOLS = [get_profile, recent_reports, save_report]
