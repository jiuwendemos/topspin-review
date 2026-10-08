"""Expose the analysis as an openjiuwen tool so other agents can call it.

A plain callable + the ``service.describe()`` contract; the backend decorates it.
"""

from __future__ import annotations

import json

from topspin_review.backend import make_tool
from topspin_review.interfaces import service

_SPEC = service.describe()


def analyze_sport_video(video_path: str, region_box: list[float] | None = None) -> str:
    """Analyze a racket-sport session video and return a coaching report.

    Args:
        video_path: Path to a video file.
        region_box: Optional normalized player region [left, top, right, bottom] in 0..1.
    """
    result = service.analyze_video_sync(video_path, region_box=service.parse_region_box(region_box))
    report = result.get("report") or {}
    return json.dumps({"report_path": result.get("report_path"), "report": report}, ensure_ascii=False)


ALL_TOOLS = [
    make_tool(
        analyze_sport_video,
        name=_SPEC["name"],
        description=_SPEC["description"],
        input_params=_SPEC["input_params"],
    )
]
