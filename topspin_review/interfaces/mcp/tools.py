"""Expose the analysis as an openjiuwen ``@tool`` so other agents can call it."""

from __future__ import annotations

import json

from openjiuwen.core.foundation.tool import tool

from topspin_review.interfaces.service import analyze_video_sync


@tool(
    name="analyze_sport_video",
    description=(
        "Analyze a racket-sport session video and return a coaching report "
        "(strengths, issues with evidence timestamps and confidence, drills, focus)."
    ),
    input_params={
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
    },
)
def analyze_sport_video(video_path: str, region_box: list[float] | None = None) -> str:
    box = tuple(region_box) if region_box and len(region_box) == 4 else None
    result = analyze_video_sync(video_path, region_box=box)
    report = result.get("report") or {}
    return json.dumps(
        {"report_path": result.get("report_path"), "report": report},
        ensure_ascii=False,
    )


ALL_TOOLS = [analyze_sport_video]
