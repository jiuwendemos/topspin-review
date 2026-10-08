"""The analysis-strategy base class and shared run machinery.

A ``Strategy`` is a template: ``analyze(params)`` prepares the clip (sample frames
+ measure), runs the mode-specific ``_run``, then finalizes (usage, report patch,
export). It also owns the steps both modes share — building the report agent,
running it, the profile line and previous-report context — so the concrete
strategies only implement their real difference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topspin_review import config, reporting
from topspin_review.analysis.pipeline.params import Params
from topspin_review.analysis.pipeline.run_session import start_session
from topspin_review.analysis.progress import Progress, tick
from topspin_review.analysis.stages.coach import build_agent
from topspin_review.analysis.stages.measure import measure
from topspin_review.backend import run_agent
from topspin_review.bootstrap import setup
from topspin_review.domain import progress as domain_progress
from topspin_review.storage import store


class Strategy:
    """Base analysis strategy (template method)."""

    name: str = ""
    description: str = ""

    async def analyze(self, params: Params) -> dict[str, Any]:
        self._prepare(params)
        result, extras = await self._run(params, params.progress)
        return await self._finalize(params, result, extras, params.progress)

    # -- template steps (override only _run / _after_report) -------------- #
    async def _run(self, params: Params, progress: Progress | None) -> tuple[Any, dict]:
        """Run the mode-specific analysis; return (agent result, first-patch extras)."""
        raise NotImplementedError

    async def _after_report(self, params: Params, report: dict, progress: Progress | None) -> dict:
        """Optional extras for the post-export patch. Default: none."""
        return {}

    # -- shared prologue -------------------------------------------------- #
    def _prepare(self, params: Params) -> None:
        progress = params.progress
        tick(progress, "preparing", 2)
        setup()
        if not Path(params.video_path).exists():
            raise FileNotFoundError(f"Video not found: {params.video_path}")

        store.set_current_video(params.video_path)
        params.profile = store.get_profile()

        measured = measure(params.video_path, params.region_box, progress)
        params.meta = measured.meta
        params.frames = measured.frames
        params.timestamps = measured.timestamps
        params.measured = measured.measured
        params.is_still = measured.is_still

        params.session = start_session(params.video_path)

    # -- shared report steps ---------------------------------------------- #
    def _build_report_agent(
        self,
        params: Params,
        *,
        system_prompt: str | None = None,
        tools: list | None = None,
        max_iterations: int = 15,
    ) -> Any:
        """Build the report-writing agent with the run's recorder."""
        return build_agent(
            rails=config.rails(),
            recorder=params.session.recorder,
            system_prompt=system_prompt,
            tools=tools,
            max_iterations=max_iterations,
        ).agent

    async def _run_report_agent(
        self,
        params: Params,
        agent: Any,
        query: str,
        progress: Progress | None,
        *,
        stage: str,
        pct: float,
    ) -> Any:
        """Run the report agent, bracketed by the stage / saving-report ticks."""
        tick(progress, stage, pct)
        result = await run_agent(agent, query)
        tick(progress, "saving report", 92)
        return result

    def _profile_line(self, params: Params) -> str:
        profile = params.profile
        return (
            f"level {profile.get('level', 'unknown')}, {profile.get('dominant_hand', 'right')}-handed, "
            f"working on {profile.get('goal', 'improve')}"
        )

    def _previous_summary(self, params: Params) -> dict:
        return domain_progress.summarize(store.get_reports()) or {}

    # -- shared epilogue -------------------------------------------------- #
    async def _finalize(
        self,
        params: Params,
        result: Any,
        extras: dict,
        progress: Progress | None,
    ) -> dict[str, Any]:
        usage = params.session.usage_summary()
        if path := params.session.save_details():
            params.artifact_paths["observability"] = path

        report = store.patch_last_report(
            {
                **extras,
                "metrics": params.measured,
                "source": params.video_path,
                "frames": len(params.frames),
                "frame_times": [round(t, 2) for t in params.timestamps],
                "usage": usage,
                "artifacts": params.artifact_paths,
                "region_box": list(params.region_box) if params.region_box else None,
            }
        )

        if report:
            tick(progress, "exporting", 97)
            post = await self._after_report(params, report, progress)
            trend = domain_progress.summarize(store.get_reports())
            try:
                exports = reporting.write(report)
            except Exception:
                exports = {}
            report = store.patch_last_report({**post, "progress_trend": trend, "exports": exports})

        tick(progress, "done", 100)

        timings: list = []
        if progress is not None:
            try:
                timings = progress.snapshot().get("timings") or []
            except Exception:
                timings = []
        if report:
            report = store.patch_last_report({"timings": timings}) or report

        return {
            "result": result,
            "report": report or {},
            "report_path": str(store.report_path(params.video_path)),
            "observations": extras.get("observations", ""),
            "frames": len(params.frames),
            "meta": params.meta,
            "timings": timings,
        }
