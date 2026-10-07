"""Model-driven analysis mode (opt-in).

Instead of the fixed coarse→zoom pipeline, the agent is given tools to inspect
the clip itself (``inspect_window``) and writes the report from what it chooses
to look at. Enable with ``AGENTIC_MODE=true`` or ``analyze --agentic``. The
deterministic pipeline in :mod:`topspin_review.analysis.pipeline` remains the
default.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from topspin_review import config, reporting
from topspin_review.analysis import prompts, rails as rails_mod
from topspin_review.analysis import tools as report_tools
from topspin_review.analysis import vision
from topspin_review.bootstrap import setup
from topspin_review.domain import progress
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


async def analyze(video_path: str, region_box: tuple[float, float, float, float] | None = None) -> dict[str, Any]:
    """Model-driven analysis: the agent inspects the clip and writes the report."""
    from openjiuwen.core.runner import Runner
    from openjiuwen.harness import create_deep_agent

    setup()
    config.validate()
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    await _ensure_runner()
    store.set_current_video(video_path)
    profile = store.get_profile()

    _meta, frames, timestamps = sampling.sample_frames(video_path, config.max_frames(), config.use_cache())
    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    steps = metrics.activity(frames, timestamps)
    ball_info = ball.detect(frames, timestamps, steps)
    measured = metrics.analyze(frames, timestamps, ball=ball_info, region_box=region_box)

    backend = get_backend()
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
    agent = create_deep_agent(
        model=config.make_model(),
        system_prompt=AGENTIC_SYSTEM,
        tools=[*AGENTIC_TOOLS, *report_tools.ALL_TOOLS],
        rails=report_rails,
        enable_task_loop=False,
        max_iterations=25,
        workspace=runtime.workspace(),
    )

    prev = progress.summarize(store.get_reports())
    history = prev.get("text", "") if prev else "No previous report."
    query = (
        f"Today is {date.today().isoformat()}. "
        f"Analyze my {profile.get('sport', 'table tennis')} session video at {video_path} "
        f"(level {profile.get('level', 'unknown')}, {profile.get('dominant_hand', 'right')}-handed, "
        f"working on {profile.get('goal', 'improve')}).\n"
        f"History: {history}\n"
        "Start with get_measurements, inspect the interesting windows, then save the report."
    )
    result = await Runner.run_agent(agent, {"query": query})

    vision_usage = backend.usage_summary() if hasattr(backend, "usage_summary") else {}
    report = store.patch_last_report(
        {
            "metrics": measured,
            "source": str(video_path),
            "frames": len(frames),
            "frame_times": [round(t, 2) for t in timestamps],
            "usage": {"vision": vision_usage, "text": {"calls": 0, "total_tokens": 0}},
            "region_box": list(region_box) if region_box else None,
        }
    )
    if report:
        trend = progress.summarize(store.get_reports())
        try:
            exports = reporting.write(report)
        except Exception:
            exports = {}
        report = store.patch_last_report({"progress_trend": trend, "exports": exports})

    return {
        "result": result,
        "report": report or {},
        "report_path": str(store.report_path(video_path)),
        "frames": len(frames),
        "observations": "",
        "meta": {},
    }
