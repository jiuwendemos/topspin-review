"""Embedding surface: run the analysis programmatically.

``analyze_video`` (async) imports the agent lazily so importing this module does
not pull in the openjiuwen harness. ``describe`` exposes a tool schema for agents
or an API wrapper.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def analyze_video(video_path: str, region_box: tuple[float, float, float, float] | None = None) -> dict[str, Any]:
    from topspin_review.analysis import agent

    return await agent.analyze(video_path, region_box=region_box)


def analyze_video_sync(video_path: str, region_box: tuple[float, float, float, float] | None = None) -> dict[str, Any]:
    return asyncio.run(analyze_video(video_path, region_box=region_box))


def describe() -> dict[str, Any]:
    return {
        "name": "analyze_sport_video",
        "description": (
            "Analyze a racket-sport session video and return a coaching report "
            "(strengths, issues with evidence timestamps and confidence, drills, focus)."
        ),
        "input_schema": {
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
    }
