"""Shared machinery for one analysis run.

Both strategies — :mod:`topspin_review.analysis.strategies.deterministic` (the
default pipeline) and :mod:`topspin_review.analysis.strategies.agentic` (the
opt-in model-driven mode) — build on this so the openjiuwen Runner lifecycle,
model/tool tracing, usage accounting and artifact writing live in exactly one
place. The vision agent is built here too (from the backend agents file).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from topspin_review.analysis import prompts
from topspin_review.backend import create_vision_agent, observability, runner
from topspin_review.backend import settings as backend_settings
from topspin_review.storage import runtime


async def ensure_runner() -> None:
    """Start openjiuwen's Runner once per process (best-effort, never raises)."""
    await runner.start()


async def run_agent(agent: Any, query: str) -> Any:
    """Run a DeepAgent query on the shared Runner."""
    return await runner.run_agent(agent, query)


@dataclass
class RunSession:
    """Tracing, the vision agent, and usage for a single analysis run."""

    video_path: str
    call_trace: observability.CallTrace
    tool_trace: observability.ToolTrace
    vision_agent: Any
    vision_usage: observability.UsageCollector
    media_dir: Path

    def usage_summary(self, text_usage: observability.UsageCollector, callback_trace: Any = None) -> dict[str, Any]:
        """Combine vision + text usage into the report's ``usage`` block."""
        vision_usage = self.vision_usage.summary()
        vision_calls = self.vision_usage.records()
        text = text_usage.summary()
        usage: dict[str, Any] = {
            "vision": vision_usage,
            "vision_calls": vision_calls,
            "text": text,
            "text_calls": text_usage.records(),
            "calls": int(vision_usage.get("calls", 0)) + int(text.get("calls", 0)),
            "total_tokens": int(vision_usage.get("total_tokens", 0)) + int(text.get("total_tokens", 0)),
            "models": {
                "text": backend_settings.text_model_name(),
                "vision": backend_settings.vision_model_name(),
                "provider": backend_settings.model_provider(),
                "api_base": backend_settings.api_base(),
            },
        }
        if callback_trace is not None:
            usage["callback_trace"] = callback_trace.summary()
        return usage

    def save_details(self) -> str | None:
        """Persist model calls + tool calls to ``<stem>_observability.json`` (best-effort)."""
        try:
            stem = Path(self.video_path).stem
            path = runtime.ARTIFACTS_DIR / f"{stem}_observability.json"
            path.write_text(
                json.dumps(
                    {"calls": self.call_trace.calls, "tools": self.tool_trace.records()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return str(path)
        except Exception:
            return None


def start_session(video_path: str) -> RunSession:
    """Install model/tool traces and build the vision agent for ``video_path``."""
    media_dir = runtime.ARTIFACTS_DIR / f"{Path(video_path).stem}_media"
    call_trace = observability.CallTrace(
        record_io=backend_settings.save_call_io(),
        media_dir=media_dir,
    )
    tool_trace = observability.ToolTrace()
    tool_trace.install()
    vision_usage = observability.UsageCollector()
    vision_agent = create_vision_agent(
        system_prompt=prompts.VISION_AGENT_SYSTEM,
        workspace=str(runtime.ARTIFACTS_DIR),
        usage=vision_usage,
        trace=call_trace,
    )
    return RunSession(
        video_path=video_path,
        call_trace=call_trace,
        tool_trace=tool_trace,
        vision_agent=vision_agent,
        vision_usage=vision_usage,
        media_dir=media_dir,
    )
