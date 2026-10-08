"""Deterministic analysis strategy (the default).

Measure motion, params two fixed vision passes, then have the report agent write the
report. The alternative model-driven strategy is
:mod:`topspin_review.analysis.strategies.agentic`.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from topspin_review import config, reporting
from topspin_review.analysis import vision
from topspin_review.analysis.coaching import retrieve_reports, verify_issues
from topspin_review.analysis.coaching.prompts import rubric_text
from topspin_review.analysis.strategies.base import Strategy
from topspin_review.analysis.strategies.params import Params
from topspin_review.analysis.strategies.progress import Progress, tick
from topspin_review.domain import report as report_schema
from topspin_review.perception import imaging, metrics, pose, quality, sampling
from topspin_review.storage import runtime, store


def _clean_windows(raw: Any, timestamps: list[float], cap: int) -> list[dict]:
    """Validate the model's requested windows into sorted [start, end] pairs."""
    end_limit = round(max(timestamps), 2) if timestamps else 0.0
    windows: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, float(item.get("start", 0)))
            end = min(end_limit, float(item.get("end", 0)))
        except (TypeError, ValueError):
            continue
        if end - start < 0.2:
            continue
        windows.append({"start": round(start, 2), "end": round(end, 2), "why": str(item.get("why", ""))})
    return windows[:cap]


def _zoom_frames(video_path: str, windows: list[dict], budget: int) -> tuple[list, list]:
    """Sample a burst of frames inside each window, capped to ``budget`` total."""
    if not windows or budget <= 0:
        return [], []
    per = max(1, budget // len(windows))
    frames: list = []
    times: list[float] = []
    for window in windows:
        f, t = sampling.sample_burst(video_path, window["start"], window["end"], config.zoom_fps(), cap=per)
        frames.extend(f)
        times.extend(t)
    if len(frames) > budget:
        step = len(frames) / budget
        keep = sorted({int(i * step) for i in range(budget)})
        frames = [frames[i] for i in keep if i < len(frames)]
        times = [times[i] for i in keep if i < len(times)]
    return frames, times


def _save_artifacts(video_path: str, frames, timestamps, measured: dict, region_box) -> dict:
    """Save the contact sheet, motion map, pose overlay and metrics; return their paths."""
    stem = Path(video_path).stem
    paths: dict[str, str] = {}

    sheet = imaging.contact_sheet(frames, timestamps, cols=min(4, len(frames)))
    sheet_path = runtime.ARTIFACTS_DIR / f"{stem}_frames.png"
    imaging.save_png(sheet, sheet_path)
    paths["frames"] = str(sheet_path)

    mask = metrics.region_from_box(region_box) if region_box else metrics.subject_region(frames)
    motion_map = metrics.motion_map(frames, mask=mask)
    if motion_map is not None:
        motion_path = runtime.ARTIFACTS_DIR / f"{stem}_motion.png"
        imaging.save_png(motion_map, motion_path)
        paths["motion"] = str(motion_path)

    if measured.get("pose"):
        annotated = pose.overlay(frames, measured["pose"])
        if annotated:
            pose_sheet = imaging.contact_sheet(annotated, timestamps, cols=min(4, len(annotated)))
            pose_path = runtime.ARTIFACTS_DIR / f"{stem}_pose.png"
            imaging.save_png(pose_sheet, pose_path)
            paths["pose"] = str(pose_path)

    metrics_path = runtime.ARTIFACTS_DIR / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(measured, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["metrics"] = str(metrics_path)
    return paths


class DeterministicStrategy(Strategy):
    """The default: fixed measure → two-pass vision → report pipeline."""

    name = "deterministic"
    description = "Fixed pipeline: measure motion, two vision passes, then write the report."

    async def _run(self, params: Params, progress: Progress | None) -> tuple[Any, dict]:
        frames, timestamps, measured, profile = params.frames, params.timestamps, params.measured, params.profile
        clip_quality = quality.evaluate(params.meta, frames, measured)
        params.state["clip_quality"] = clip_quality

        tick(progress, "writing artifacts", 30)
        params.artifact_paths.update(_save_artifacts(params.video_path, frames, timestamps, measured, params.region_box))

        agent, media_dir = params.session.vision_agent, params.session.media_dir
        if params.is_still:
            tick(progress, "vision: reviewing image", 55)
            coarse_out = {"overall": "", "attentive_windows": [], "limitations": []}
            windows: list[dict] = []
            zoom_imgs: list = []
            zoom_times: list[float] = []
            fine_out = await vision.analyze_still(frames[0], profile, agent=agent, media_dir=media_dir)
        else:
            tick(progress, "vision: overview", 38)
            coarse_out = await vision.coarse(params.meta, frames, timestamps, measured, profile, agent=agent, media_dir=media_dir)
            windows = _clean_windows(coarse_out.get("attentive_windows"), timestamps, config.max_windows())
            tick(progress, "vision: zoom", 52)
            zoom_imgs, zoom_times = _zoom_frames(params.video_path, windows, config.zoom_frames())
            tick(progress, "vision: detail", 60)
            fine_out = await vision.fine(
                frames, timestamps, measured, windows, zoom_imgs, zoom_times, profile, agent=agent, media_dir=media_dir
            )
        observations = vision.observations_text(coarse_out, fine_out)
        params.state["observations"] = observations

        all_reports = store.get_reports()
        prev = self._previous_summary(params)
        progress_note = prev.get("text", "") if prev else "No previous report."
        repeated = ", ".join(prev.get("repeated_themes", [])) if prev else ""
        related = ""
        if config.retrieval_enabled():
            related = retrieve_reports.context_text(all_reports, f"{profile.get('goal', '')} {observations}")

        window_label = ", ".join(f"{w['start']}-{w['end']}s" for w in windows) or "none"
        rubric = rubric_text(profile.get("sport", "table tennis"))
        baseline = ""
        if all_reports:
            prev_mech = (all_reports[-1].get("metrics") or {}).get("mechanics") or {}
            now_mech = measured.get("mechanics") or {}
            if prev_mech or now_mech:
                deltas = {
                    k: round(float(now_mech.get(k, 0)) - float(prev_mech.get(k, 0)), 3)
                    for k in set(prev_mech) | set(now_mech)
                }
                baseline = f"Previous mechanics: {prev_mech}\nNow: {now_mech}\nChange: {deltas}"
        quality_note = "; ".join(clip_quality.get("warnings") or []) or "ok"

        query = (
            f"Today is {date.today().isoformat()}. "
            f"Write my coaching report for a {profile.get('sport', 'table tennis')} session.\n"
            f"Player: {self._profile_line(params)}.\n\n"
            f"History: {progress_note}" + (f" Recurring themes: {repeated}." if repeated else "") + "\n"
            + (f"Related past sessions:\n{related}\n" if related else "")
            + "\n"
            f"Analysis ({len(frames)} sampled frames, windows examined: {window_label}):\n"
            f"{observations}\n\n"
            f"Measured motion/mechanics:\n{metrics.text(measured)}\n\n"
            f"Footage quality: {quality_note}.\n\n"
            + (f"Baseline (previous session):\n{baseline}\n\n" if baseline else "")
            + f"Check the player against this technique rubric (issues should map to it):\n{rubric}\n\n"
            "Save the report with save_report (schema with evidence_times and confidence), "
            "then give me a short summary."
        )
        result = await self._run_report_agent(
            params, self._build_report_agent(params), query, progress, stage="agent writing report", pct=80
        )
        return result, {
            "observations": observations,
            "windows": windows,
            "signals": fine_out.get("signals") or [],
        }

    async def _after_report(self, params: Params, report: dict, progress: Progress | None) -> dict:
        clip_quality = params.state["clip_quality"]
        tick(progress, "verifying", 95)
        report_schema.calibrate(report, clip_quality)
        if config.verify_reports():
            try:
                await verify_issues.verify(report, params.state["observations"], metrics.text(params.measured))
            except Exception:
                pass

        clips: dict = {}
        if config.clips_enabled() and not params.is_still:
            clip_times: list[float] = []
            for item in (report.get("issues") or [])[:3]:
                if isinstance(item, dict):
                    clip_times.extend(item.get("evidence_times") or [])
            try:
                clips = reporting.make_clips(params.video_path, clip_times, Path(params.video_path).stem)
            except Exception:
                clips = {}

        limitations = list(report.get("limitations") or []) + list(clip_quality.get("warnings") or [])
        return {"quality": clip_quality, "clips": clips, "limitations": limitations}
