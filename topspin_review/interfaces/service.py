"""Embedding surface: run the analysis programmatically.

``analyze_video`` (async) imports the pipeline lazily so importing this module does
not pull in the heavy backend. ``describe`` is the single source of truth for
the ``analyze_sport_video`` tool contract (used by the MCP server).
"""

from __future__ import annotations

import json
from typing import Any

from topspin_review.bootstrap import run as run_async
from topspin_review.storage import store

QA_SYSTEM = (
    "You are Topspin Review. Answer questions about the player's coaching report using ONLY the "
    "report and its measurements. Be concise and practical. Never invent spin, ball speed, or "
    "exact angles. If the report does not cover it, say so."
)

TOOL_NAME = "analyze_sport_video"
TOOL_DESCRIPTION = (
    "Analyze a racket-sport session video and return a coaching report "
    "(strengths, issues with evidence timestamps and confidence, drills, focus)."
)
TOOL_INPUT_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string", "description": "Path to a video file."},
        "region_box": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Optional normalized player region [left, top, right, bottom] in 0..1.",
        },
    },
    "required": ["video_path"],
}


def parse_region_box(value: Any) -> tuple[float, float, float, float] | None:
    """Coerce a region-box argument into a 4-tuple, or ``None`` if invalid."""
    if not value:
        return None
    try:
        box = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return box if len(box) == 4 else None


async def analyze_video(
    video_path: str, region_box: tuple[float, float, float, float] | None = None
) -> dict[str, Any]:
    from topspin_review.analysis import strategies

    strategy = strategies.get_strategy(strategies.DETERMINISTIC)
    return await strategy.analyze(strategies.Params(video_path=video_path, region_box=region_box))


def analyze_video_sync(
    video_path: str, region_box: tuple[float, float, float, float] | None = None
) -> dict[str, Any]:
    return run_async(analyze_video(video_path, region_box=region_box))


async def ask(stem: str, question: str) -> str:
    """Answer a follow-up question grounded in the stored report."""
    report = store.find_report(stem) if stem else None
    if report is None:
        reports = store.get_reports()
        report = reports[-1] if reports else None
    if not report:
        return "I don't have a report to answer from yet."

    context = {
        "summary": report.get("summary"),
        "strengths": report.get("strengths"),
        "issues": report.get("issues"),
        "drills": report.get("drills"),
        "focus": report.get("focus"),
        "metrics": report.get("metrics"),
        "limitations": report.get("limitations"),
    }
    prompt = f"Report:\n{json.dumps(context, ensure_ascii=False)}\n\nQuestion: {question}"
    try:
        from topspin_review.backend import run_text

        return await run_text(QA_SYSTEM, prompt)
    except Exception as exc:  # noqa: BLE001
        return f"Could not answer: {exc}"


def ask_sync(stem: str, question: str) -> str:
    return run_async(ask(stem, question))


def describe() -> dict[str, Any]:
    return {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_params": TOOL_INPUT_PARAMS}
