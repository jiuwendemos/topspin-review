"""Expose the analysis as an openjiuwen ``@tool`` so other agents can call it."""

from __future__ import annotations

import json

from topspin_review.backend import tool
from topspin_review.interfaces import service

_SPEC = service.describe()


@tool(name=_SPEC["name"], description=_SPEC["description"], input_params=_SPEC["input_params"])
def analyze_sport_video(video_path: str, region_box: list[float] | None = None) -> str:
    result = service.analyze_video_sync(video_path, region_box=service.parse_region_box(region_box))
    report = result.get("report") or {}
    return json.dumps({"report_path": result.get("report_path"), "report": report}, ensure_ascii=False)


ALL_TOOLS = [analyze_sport_video]
