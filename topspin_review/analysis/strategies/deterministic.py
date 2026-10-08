"""Deterministic analysis pipeline (the default strategy).

Measure motion, run two vision passes, then have the report agent write the
report. The alternative model-driven strategy is
:mod:`topspin_review.analysis.strategies.agentic`.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from topspin_review import config, reporting
from topspin_review.analysis import prompts, vision
from topspin_review.analysis.coaching import build_agent, retrieve_reports, verify_issues
from topspin_review.analysis.progress import Progress, tick
from topspin_review.analysis.run_session import start_session
from topspin_review.backend import run_agent
from topspin_review.bootstrap import setup
from topspin_review.domain import progress as domain_progress
from topspin_review.domain import report as report_schema
from topspin_review.perception import ball, imaging, metrics, pose, quality, sampling
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
    if not windows or budget <= 0:
        return [], []
    per = max(1, budget // len(windows))
    frames: list = []
    times: list[float] = []
    for w in windows:
        f, t = sampling.sample_burst(video_path, w["start"], w["end"], config.zoom_fps(), cap=per)
        frames.extend(f)
        times.extend(t)
    if len(frames) > budget:
        step = len(frames) / budget
        keep = sorted({int(i * step) for i in range(budget)})
        frames = [frames[i] for i in keep if i < len(frames)]
        times = [times[i] for i in keep if i < len(times)]
    return frames, times


def _save_artifacts(
    video_path: str,
    frames,
    timestamps,
    metrics_data: dict,
    region_box: tuple[float, float, float, float] | None = None,
) -> dict:
    stem = Path(video_path).stem
    paths: dict[str, str] = {}
    sheet = imaging.contact_sheet(frames, timestamps, cols=min(4, len(frames)))
    sheet_path = runtime.ARTIFACTS_DIR / f"{stem}_frames.png"
    imaging.save_png(sheet, sheet_path)
    paths["frames"] = str(sheet_path)

    mask = metrics.region_from_box(region_box) if region_box else metrics.subject_region(frames)
    mm = metrics.motion_map(frames, mask=mask)
    if mm is not None:
        motion_path = runtime.ARTIFACTS_DIR / f"{stem}_motion.png"
        imaging.save_png(mm, motion_path)
        paths["motion"] = str(motion_path)

    if metrics_data.get("pose"):
        annotated = pose.overlay(frames, metrics_data["pose"])
        if annotated:
            pose_sheet = imaging.contact_sheet(annotated, timestamps, cols=min(4, len(annotated)))
            pose_path = runtime.ARTIFACTS_DIR / f"{stem}_pose.png"
            imaging.save_png(pose_sheet, pose_path)
            paths["pose"] = str(pose_path)

    metrics_path = runtime.ARTIFACTS_DIR / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_data, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["metrics"] = str(metrics_path)
    return paths


async def analyze(
    video_path: str,
    region_box: tuple[float, float, float, float] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Measure motion, run two vision passes, then have the agent write the report.

    ``region_box`` (normalized l,t,r,b) restricts analysis to a user-picked area.
    ``progress`` (optional) receives stage/percent updates.
    """
    tick(progress, "preparing", 2)
    setup()
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    store.set_current_video(video_path)
    profile = store.get_profile()

    tick(progress, "sampling frames", 8)
    is_still = sampling.is_image(video_path)
    if is_still:
        meta, frames, timestamps = sampling.load_image(video_path)
    else:
        meta, frames, timestamps = sampling.sample_frames(video_path, config.max_frames(), config.use_cache())
    if not frames:
        raise ValueError("No frames could be extracted from the input.")

    tick(progress, "measuring motion", 22)
    steps = metrics.activity(frames, timestamps)
    ball_info = ball.detect(frames, timestamps, steps)
    measured = metrics.analyze(frames, timestamps, ball=ball_info, region_box=region_box)
    clip_quality = quality.evaluate(meta, frames, measured)

    tick(progress, "writing artifacts", 30)
    artifact_paths = _save_artifacts(video_path, frames, timestamps, measured, region_box)

    session = start_session(video_path)
    agent = session.vision_agent
    media_dir = session.media_dir
    if is_still:
        tick(progress, "vision: reviewing image", 55)
        coarse_out = {"overall": "", "attentive_windows": [], "limitations": []}
        windows: list[dict] = []
        zoom_frames: list = []
        zoom_times: list[float] = []
        fine_out = await vision.analyze_still(frames[0], profile, agent=agent, media_dir=media_dir)
    else:
        tick(progress, "vision: overview", 38)
        coarse_out = await vision.coarse(
            meta, frames, timestamps, measured, profile, agent=agent, media_dir=media_dir
        )
        windows = _clean_windows(coarse_out.get("attentive_windows"), timestamps, config.max_windows())
        tick(progress, "vision: zoom", 52)
        zoom_frames, zoom_times = _zoom_frames(video_path, windows, config.zoom_frames())
        tick(progress, "vision: detail", 60)
        fine_out = await vision.fine(
            frames, timestamps, measured, windows, zoom_frames, zoom_times, profile, agent=agent, media_dir=media_dir
        )
    observations = vision.observations_text(coarse_out, fine_out)

    all_reports = store.get_reports()
    prev_progress = domain_progress.summarize(all_reports)
    progress_note = prev_progress.get("text", "") if prev_progress else "No previous report."
    repeated = ", ".join(prev_progress.get("repeated_themes", [])) if prev_progress else ""

    related = ""
    if config.retrieval_enabled():
        related = retrieve_reports.context_text(all_reports, f"{profile.get('goal', '')} {observations}")

    window_label = ", ".join(f"{w['start']}-{w['end']}s" for w in windows) or "none"

    rubric = prompts.rubric_text(profile.get("sport", "table tennis"))
    baseline = ""
    if all_reports:
        prev_mech = (all_reports[-1].get("metrics") or {}).get("mechanics") or {}
        now_mech = measured.get("mechanics") or {}
        if prev_mech or now_mech:
            deltas = {k: round(float(now_mech.get(k, 0)) - float(prev_mech.get(k, 0)), 3) for k in set(prev_mech) | set(now_mech)}
            baseline = f"Previous mechanics: {prev_mech}\nNow: {now_mech}\nChange: {deltas}"
    quality_note = "; ".join(clip_quality.get("warnings") or []) or "ok"

    tick(progress, "preparing report", 72)
    agent = build_agent(rails=config.rails(), recorder=session.recorder).agent
    query = (
        f"Today is {date.today().isoformat()}. "
        f"Write my coaching report for a {profile.get('sport', 'table tennis')} session.\n"
        f"Player: level {profile.get('level', 'unknown')}, "
        f"{profile.get('dominant_hand', 'right')}-handed, working on: "
        f"{profile.get('goal', 'improve')}.\n\n"
        f"History: {progress_note}" + (f" Recurring themes: {repeated}." if repeated else "") + "\n"
        + (f"Related past sessions:\n{related}\n" if related else "")
        + "\n"
        f"Analysis ({len(frames)} sampled frames, windows examined: {window_label}):\n"
        f"{observations}\n\n"
        f"Measured motion/mechanics:\n{prompts.metrics_text(measured)}\n\n"
        f"Footage quality: {quality_note}.\n\n"
        + (f"Baseline (previous session):\n{baseline}\n\n" if baseline else "")
        + f"Check the player against this technique rubric (issues should map to it):\n{rubric}\n\n"
        "Save the report with save_report (schema with evidence_times and confidence), "
        "then give me a short summary."
    )
    tick(progress, "agent writing report", 80)
    result = await run_agent(agent, query)
    tick(progress, "saving report", 92)

    usage = session.usage_summary()
    if path := session.save_details():
        artifact_paths["observability"] = path

    report = store.patch_last_report(
        {
            "observations": observations,
            "metrics": measured,
            "windows": windows,
            "signals": fine_out.get("signals") or [],
            "source": str(video_path),
            "frames": len(frames),
            "frame_times": [round(t, 2) for t in timestamps],
            "artifacts": artifact_paths,
            "usage": usage,
            "region_box": list(region_box) if region_box else None,
        }
    )

    if report:
        tick(progress, "verifying", 95)
        report_schema.calibrate(report, clip_quality)
        if config.verify_reports():
            try:
                await verify_issues.verify(report, observations, prompts.metrics_text(measured))
            except Exception:
                pass

        clips: dict = {}
        if config.clips_enabled() and not is_still:
            clip_times: list[float] = []
            for item in (report.get("issues") or [])[:3]:
                if isinstance(item, dict):
                    clip_times.extend(item.get("evidence_times") or [])
            try:
                clips = reporting.make_clips(str(video_path), clip_times, Path(video_path).stem)
            except Exception:
                clips = {}

        tick(progress, "exporting", 97)
        trend = domain_progress.summarize(store.get_reports())
        limitations = list(report.get("limitations") or []) + list(clip_quality.get("warnings") or [])
        try:
            exports = reporting.write(report)
        except Exception:
            exports = {}
        report = store.patch_last_report(
            {
                "progress_trend": trend,
                "exports": exports,
                "quality": clip_quality,
                "clips": clips,
                "limitations": limitations,
            }
        )

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
        "report_path": str(store.report_path(video_path)),
        "observations": observations,
        "frames": len(frames),
        "meta": meta,
        "timings": timings,
    }
