"""Model-driven analysis strategy (opt-in).

Instead of the fixed coarse→zoom pipeline, the agent is given tools to inspect
the clip itself (``inspect_window``) and writes the report from what it chooses
to look at. Enable with ``AGENTIC_MODE=true`` or ``analyze --agentic``. The
deterministic strategy remains the default.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from topspin_review.analysis.pipeline.params import Params
from topspin_review.analysis.pipeline.progress import Progress
from topspin_review.analysis.pipeline.strategies.base import Strategy
from topspin_review.analysis.stages import observe
from topspin_review.analysis.stages.coach import report_tools
from topspin_review.perception import metrics, sampling

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


def get_measurements() -> str:
    """Return the overall measured motion/mechanics for the current clip."""
    return metrics.text(_STATE.get("metrics") or {})


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
    out = await observe.fine(
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
    return observe.observations_text({"overall": ""}, out) + "\n" + metrics.text(window_metrics)


AGENTIC_TOOLS = [get_measurements, inspect_window]


class AgenticStrategy(Strategy):
    """The opt-in mode: the agent inspects the clip itself and writes the report."""

    name = "agentic"
    description = "Model-driven: the agent inspects the clip itself and writes the report."

    async def _run(self, params: Params, progress: Progress | None) -> tuple[Any, dict]:
        _STATE.clear()
        _STATE.update(
            {
                "video": params.video_path,
                "frames": params.frames,
                "timestamps": params.timestamps,
                "metrics": params.measured,
                "profile": params.profile,
                "agent": params.session.vision_agent,
                "media_dir": params.session.media_dir,
            }
        )

        agent = self._build_report_agent(
            params,
            system_prompt=AGENTIC_SYSTEM,
            tools=[*AGENTIC_TOOLS, *report_tools.ALL_TOOLS],
            max_iterations=25,
        )

        prev = self._previous_summary(params)
        history = prev.get("text", "") if prev else "No previous report."
        query = (
            f"Today is {date.today().isoformat()}. "
            f"Analyze my {params.profile.get('sport', 'table tennis')} session video at {params.video_path} "
            f"({self._profile_line(params)}).\n"
            f"History: {history}\n"
            "Start with get_measurements, inspect the interesting windows, then save the report."
        )
        result = await self._run_report_agent(params, agent, query, progress, stage="agent: analyzing", pct=45)
        return result, {}
