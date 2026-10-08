"""Model-driven analysis strategy (opt-in).

Instead of the fixed coarse→zoom pipeline, the agent is given tools to inspect
the clip itself (``inspect_window``) and writes the report from what it chooses
to look at. Enable with ``AGENTIC_MODE=true`` or ``analyze --agentic``. The
deterministic strategy in :mod:`topspin_review.analysis.strategies.deterministic`
remains the default.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from topspin_review import config, reporting
from topspin_review.analysis import prompts, vision
from topspin_review.analysis.coaching import build_agent, report_tools
from topspin_review.analysis.progress import Progress, tick
from topspin_review.analysis.run_session import start_session
from topspin_review.backend import run_agent
from topspin_review.bootstrap import setup
from topspin_review.domain import progress as domain_progress
from topspin_review.perception import ball, metrics, sampling
from topspin_review.storage import store

_STATE: dict[str, Any] = {}


def get_measurements() -> str:
    """Return the overall measured motion/mechanics for the current clip."""
    return prompts.metrics_text(_STATE.get("metrics") or {})


async def inspect_window(start: float, end: float, frame_count: int = 4) -> str:
    """Sample frames in a time window, analyze them, and return observations.

    Args:
        start: Window start in seconds.
        end: Window end in seconds.
        frame_count: How many frames to sample (default 4).
    """
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
        agent=state["agent"],
        media_dir=state["media_dir"],
    )
    return vision.observations_text({"overall": ""}, out) + "\n" + prompts.metrics_text(window_metrics)


AGENTIC_TOOLS = [get_measurements, inspect_window]


async def analyze(
    video_path: str,
    region_box: tuple[float, float, float, float] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Model-driven analysis: the agent inspects the clip and writes the report."""
    tick(progress, "preparing", 2)
    setup()
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

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

    session = start_session(video_path)
    _STATE.clear()
    _STATE.update(
        {
            "video": video_path,
            "frames": frames,
            "timestamps": timestamps,
            "metrics": measured,
            "profile": profile,
            "agent": session.vision_agent,
            "media_dir": session.media_dir,
        }
    )

    agent = build_agent(
        rails=config.rails(),
        recorder=session.recorder,
        system_prompt=prompts.AGENTIC_SYSTEM,
        tools=[*AGENTIC_TOOLS, *report_tools.ALL_TOOLS],
        max_iterations=25,
    ).agent

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
    result = await run_agent(agent, query)
    tick(progress, "saving report", 92)

    usage = session.usage_summary()
    artifacts: dict = {}
    if path := session.save_details():
        artifacts["observability"] = path

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
