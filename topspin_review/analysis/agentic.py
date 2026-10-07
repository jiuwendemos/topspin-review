"""Model-driven analysis mode (opt-in).

Instead of the fixed coarse→zoom pipeline, the agent is given tools to inspect
the clip itself (``inspect_window``) and writes the report from what it chooses
to look at. Enable with ``AGENTIC_MODE=true`` or ``analyze --agentic``. The
deterministic pipeline in :mod:`topspin_review.analysis.pipeline` remains the
default.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from topspin_review import config, observability, reporting
from topspin_review.analysis import prompts, vision
from topspin_review.analysis import rails as rails_mod
from topspin_review.analysis import tools as report_tools
from topspin_review.analysis.progress import Progress, tick
from topspin_review.bootstrap import setup
from topspin_review.domain import progress as domain_progress
from topspin_review.perception import ball, metrics, sampling
from topspin_review.providers import get_backend
from topspin_review.storage import runtime, store

try:  # openjiuwen is provided by the host environment
    from openjiuwen.core.foundation.tool import tool
except Exception:  # pragma: no cover

    def tool(**_kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        return _wrap


AGENTIC_SYSTEM = """You are Topspin Review. You analyze a racket-sport session video
yourself and write a coaching report.

Tools:
- get_measurements(): overall measured motion/mechanics for the clip.
- inspect_window(start, end, frame_count): sample frames in a time window, look at
  them with the vision model, and return observations for that window.
- get_profile(), recent_reports(n), save_report(report_json): the report tools.

Procedure:
1. Call get_measurements() to see where motion peaks.
2. Call inspect_window() for 1-3 windows you want to examine (e.g. around the peak).
3. Save the report with save_report, EXACTLY this shape:
   {"date": "...", "sport": "...", "summary": "...",
    "strengths": ["..."],
    "issues": [{"issue": "...", "evidence_times": [2.9], "confidence": "high|medium|low"}],
    "drills": ["..."], "focus": "...", "progress": "...", "limitations": ["..."]}
   Every issue MUST cite evidence_times (seconds). Never claim spin, ball speed, or
   exact angles. If the clip is unclear, say so in limitations.
4. Reply with a short plain-text summary.
"""

_STATE: dict[str, Any] = {}


@tool(
    name="get_measurements",
    description="Return the overall measured motion/mechanics for the current clip.",
    input_params={"type": "object", "properties": {}, "required": []},
)
def get_measurements() -> str:
    return prompts.metrics_text(_STATE.get("metrics") or {})


@tool(
    name="inspect_window",
    description="Sample frames between start and end seconds, analyze them with the vision model, and return observations.",
    input_params={
        "type": "object",
        "properties": {
            "start": {"type": "number", "description": "Window start in seconds."},
            "end": {"type": "number", "description": "Window end in seconds."},
            "frame_count": {"type": "integer", "description": "How many frames to sample (default 4)."},
        },
        "required": ["start", "end"],
    },
)
async def inspect_window(start: float, end: float, frame_count: int = 4) -> str:
    state = _STATE
    if not state:
        return "No clip loaded."
    frames, times = sampling.sample_window(state["video"], float(start), float(end), int(frame_count or 4))
    if not frames:
        return "No frames in that window."
    window_metrics = metrics.analyze(frames, times)
    out = await vision.fine(
        state["frames"],
        state["timestamps"],
        state["metrics"],
        [{"start": float(start), "end": float(end)}],
        frames,
        times,
        state["profile"],
        backend=state["backend"],
    )
    return vision.observations_text({"overall": ""}, out) + "\n" + prompts.metrics_text(window_metrics)


AGENTIC_TOOLS = [get_measurements, inspect_window]

_runner_started = False


async def _ensure_runner() -> None:
    global _runner_started
    if not _runner_started:
        from openjiuwen.core.runner import Runner

        try:
            await Runner.start()
        except Exception:
            pass
        _runner_started = True


async def analyze(
    video_path: str,
    region_box: tuple[float, float, float, float] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Model-driven analysis: the agent inspects the clip and writes the report."""
    from openjiuwen.core.runner import Runner
    from openjiuwen.harness import create_deep_agent

    tick(progress, "preparing", 2)
    setup()
    config.validate()
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    await _ensure_runner()
    store.set_current_video(video_path)
    profile = store.get_profile()

    tick(progress, "sampling frames", 8)
    _meta, frames, timestamps = sampling.sample_frames(video_path, config.max_frames(), config.use_cache())
    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    tick(progress, "measuring motion", 22)
    steps = metrics.activity(frames, timestamps)
    ball_info = ball.detect(frames, timestamps, steps)
    measured = metrics.analyze(frames, timestamps, ball=ball_info, region_box=region_box)

    call_trace = observability.CallTrace()
    tool_trace = observability.ToolTrace()
    tool_trace.install()
    backend = get_backend(trace=call_trace)
    _STATE.clear()
    _STATE.update(
        {
            "video": video_path,
            "frames": frames,
            "timestamps": timestamps,
            "metrics": measured,
            "profile": profile,
            "backend": backend,
        }
    )

    report_rails = rails_mod.build_rails() if config.rails_enabled() else []
    text_usage = observability.UsageCollector()
    agent = create_deep_agent(
        model=observability.attach(config.make_model(), text_usage, call_trace),
        system_prompt=AGENTIC_SYSTEM,
        tools=[*AGENTIC_TOOLS, *report_tools.ALL_TOOLS],
        rails=report_rails,
        enable_task_loop=False,
        max_iterations=25,
        workspace=runtime.workspace(),
    )

    prev = domain_progress.summarize(store.get_reports())
    history = prev.get("text", "") if prev else "No previous report."
    query = (
        f"Today is {date.today().isoformat()}. "
        f"Analyze my {profile.get('sport', 'table tennis')} session video at {video_path} "
        f"(level {profile.get('level', 'unknown')}, {profile.get('dominant_hand', 'right')}-handed, "
        f"working on {profile.get('goal', 'improve')}).\n"
        f"History: {history}\n"
        "Start with get_measurements, inspect the interesting windows, then save the report."
    )
    tick(progress, "agent: analyzing", 45)
    result = await Runner.run_agent(agent, {"query": query})
    tick(progress, "saving report", 92)

    vision_usage = backend.usage_summary() if hasattr(backend, "usage_summary") else {}
    vision_calls = backend.usage_calls() if hasattr(backend, "usage_calls") else []
    text = text_usage.summary()
    usage = {
        "vision": vision_usage,
        "vision_calls": vision_calls,
        "text": text,
        "text_calls": text_usage.records(),
        "calls": int(vision_usage.get("calls", 0)) + int(text.get("calls", 0)),
        "total_tokens": int(vision_usage.get("total_tokens", 0)) + int(text.get("total_tokens", 0)),
        "models": {
            "text": config.text_model_name(),
            "vision": config.vision_model_name(),
            "provider": config.model_provider(),
            "api_base": config.api_base(),
        },
    }
    artifacts: dict = {}
    try:
        stem = Path(video_path).stem
        calls_path = runtime.ARTIFACTS_DIR / f"{stem}_calls.json"
        calls_path.write_text(json.dumps(call_trace.calls, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["calls"] = str(calls_path)
        tools_path = runtime.ARTIFACTS_DIR / f"{stem}_tools.json"
        tools_path.write_text(json.dumps(tool_trace.records(), ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["tools"] = str(tools_path)
    except Exception:
        pass

    report = store.patch_last_report(
        {
            "metrics": measured,
            "source": str(video_path),
            "frames": len(frames),
            "frame_times": [round(t, 2) for t in timestamps],
            "usage": usage,
            "artifacts": artifacts,
            "region_box": list(region_box) if region_box else None,
        }
    )
    if report:
        tick(progress, "exporting", 97)
        trend = domain_progress.summarize(store.get_reports())
        try:
            exports = reporting.write(report)
        except Exception:
            exports = {}
        report = store.patch_last_report({"progress_trend": trend, "exports": exports})

    tick(progress, "done", 100)

    return {
        "result": result,
        "report": report or {},
        "report_path": str(store.report_path(video_path)),
        "frames": len(frames),
        "observations": "",
        "meta": {},
    }
