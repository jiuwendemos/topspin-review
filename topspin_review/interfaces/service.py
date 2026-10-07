"""Embedding surface: run the analysis programmatically.

``analyze_video`` (async) imports the pipeline lazily so importing this module does
not pull in the openjiuwen harness. ``describe`` is the single source of truth for
the ``analyze_sport_video`` tool contract (used by the MCP tool and server).
"""

from __future__ import annotations

from typing import Any

from topspin_review.bootstrap import run as run_async

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
    from topspin_review.analysis import pipeline

    return await pipeline.analyze(video_path, region_box=region_box)


def analyze_video_sync(
    video_path: str, region_box: tuple[float, float, float, float] | None = None
) -> dict[str, Any]:
    return run_async(analyze_video(video_path, region_box=region_box))


def describe() -> dict[str, Any]:
    return {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_params": TOOL_INPUT_PARAMS}
