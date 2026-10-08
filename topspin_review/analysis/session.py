"""Shared machinery for one analysis run.

Both strategies — :mod:`topspin_review.analysis.strategies.deterministic` (the
default pipeline) and :mod:`topspin_review.analysis.strategies.agentic` (the
opt-in model-driven mode) — build on this so the openjiuwen Runner lifecycle and
the run recorder live in exactly one place. The vision agent is built here too
(from the backend agent builder).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from topspin_review.analysis import prompts
from topspin_review.backend import VisionParams, build, run_agent
from topspin_review.storage import runtime

__all__ = ["RunSession", "run_agent", "start_session"]


@dataclass
class RunSession:
    """The run recorder, the vision agent and the media dir for a single run."""

    video_path: str
    recorder: Any
    vision_agent: Any
    media_dir: Path

    def usage_summary(self) -> dict[str, Any]:
        """Combine vision + text usage into the report's ``usage`` block."""
        vision_usage = self.recorder.vision.summary() if self.recorder else {}
        vision_calls = self.recorder.vision.records() if self.recorder else []
        text = self.recorder.text.summary() if self.recorder else {}
        text_calls = self.recorder.text.records() if self.recorder else []
        usage: dict[str, Any] = {
            "vision": vision_usage,
            "vision_calls": vision_calls,
            "text": text,
            "text_calls": text_calls,
            "calls": int(vision_usage.get("calls", 0)) + int(text.get("calls", 0)),
            "total_tokens": int(vision_usage.get("total_tokens", 0)) + int(text.get("total_tokens", 0)),
            "models": self.recorder.models() if self.recorder else {},
        }
        if self.recorder is not None and self.recorder.callback is not None:
            usage["callback_trace"] = self.recorder.callback.summary()
        return usage

    def save_details(self) -> str | None:
        """Persist model calls + tool calls to ``<stem>_observability.json`` (best-effort)."""
        if self.recorder is None:
            return None
        try:
            stem = Path(self.video_path).stem
            path = runtime.ARTIFACTS_DIR / f"{stem}_observability.json"
            path.write_text(
                json.dumps(
                    {"calls": self.recorder.calls.calls, "tools": self.recorder.tools.records()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return str(path)
        except Exception:
            return None


def start_session(video_path: str) -> RunSession:
    """Build the vision agent (and run recorder) for ``video_path``."""
    media_dir = runtime.ARTIFACTS_DIR / f"{Path(video_path).stem}_media"
    result = build(
        VisionParams(
            system_prompt=prompts.VISION_AGENT_SYSTEM,
            workspace=str(runtime.ARTIFACTS_DIR),
            media_dir=str(media_dir),
        )
    )
    return RunSession(
        video_path=video_path,
        recorder=result.recorder,
        vision_agent=result.agent,
        media_dir=media_dir,
    )
