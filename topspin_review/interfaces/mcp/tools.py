"""Expose the analysis as an openjiuwen tool so other agents can call it.

A plain callable + :class:`~topspin_review.backend.ToolSpec`; the backend decorates it.
"""

from __future__ import annotations

import json

from topspin_review.backend import ToolSpec, decorate
from topspin_review.interfaces import service

_SPEC = service.describe()


def analyze_sport_video(video_path: str, region_box: list[float] | None = None) -> str:
    result = service.analyze_video_sync(video_path, region_box=service.parse_region_box(region_box))
    report = result.get("report") or {}
    return json.dumps({"report_path": result.get("report_path"), "report": report}, ensure_ascii=False)


ALL_TOOLS = [decorate(ToolSpec(analyze_sport_video, _SPEC["name"], _SPEC["description"], _SPEC["input_params"]))]
