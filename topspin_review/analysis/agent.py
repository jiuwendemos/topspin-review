"""Analyze a session video: measure motion, two-pass vision, then write the report."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from topspin_review import config
from topspin_review.analysis import tools, vision
from topspin_review.analysis import usage as usage_mod
from topspin_review.domain import progress
from topspin_review.perception import ball, motion, pose, sampling, video
from topspin_review.providers.backends import get_backend
from topspin_review.reporting import export
from topspin_review.storage import runtime, store

SYSTEM_PROMPT = """You are Topspin Review. You turn objective, measured observations
from a time-ordered analysis of a racket-sport session into a practical coaching report.

You have these tools:
- get_profile(): the player's sport, level, dominant hand, and goal.
- recent_reports(n): earlier reports, to note progress.
- save_report(report_json): save the report.

When asked to write a report from observations:
1. Call get_profile and recent_reports first.
2. Save the report with save_report using EXACTLY this JSON shape:
   {"date": "...", "sport": "...", "summary": "...",
    "strengths": ["...", "..."],
    "issues": [{"issue": "...", "evidence_times": [2.9, 3.8], "confidence": "high|medium|low"}],
    "drills": ["...", "..."],
    "focus": "one thing to work on next session",
    "progress": "one sentence comparing with the previous report, or '' if none",
    "limitations": ["...", "..."]}
3. Put the most impactful issues first. Every issue MUST cite the evidence_times
   (seconds) it is based on. Use the measured motion metrics and the motion map as
   evidence; do NOT invent spin, ball speed, or exact angles.
4. If the observations say the clip is unclear or the frames near-duplicate, say so
   in "limitations" and keep confidence low.
5. Reply with a short plain-text summary (do not paste the JSON back).
"""


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


def build_agent(model=None):
    from openjiuwen.harness import create_deep_agent

    runtime.setup()

    return create_deep_agent(
        model=model or config.make_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=tools.ALL_TOOLS,
        enable_task_loop=False,
        max_iterations=15,
        workspace=runtime.workspace(),
    )


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
        f, t = sampling.sample_window(video_path, w["start"], w["end"], per)
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
    metrics: dict,
    region_box: tuple[float, float, float, float] | None = None,
) -> dict:
    stem = Path(video_path).stem
    paths: dict[str, str] = {}
    sheet = video.contact_sheet(frames, timestamps, cols=min(4, len(frames)))
    sheet_path = runtime.ARTIFACTS_DIR / f"{stem}_frames.png"
    video.save_png(sheet, sheet_path)
    paths["frames"] = str(sheet_path)

    mask = motion.region_from_box(region_box) if region_box else motion.subject_region(frames)
    mm = motion.motion_map(frames, mask=mask)
    if mm is not None:
        motion_path = runtime.ARTIFACTS_DIR / f"{stem}_motion.png"
        video.save_png(mm, motion_path)
        paths["motion"] = str(motion_path)

    if metrics.get("pose"):
        annotated = pose.overlay(frames, metrics["pose"])
        if annotated:
            pose_sheet = video.contact_sheet(annotated, timestamps, cols=min(4, len(annotated)))
            pose_path = runtime.ARTIFACTS_DIR / f"{stem}_pose.png"
            video.save_png(pose_sheet, pose_path)
            paths["pose"] = str(pose_path)

    metrics_path = runtime.ARTIFACTS_DIR / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["metrics"] = str(metrics_path)
    return paths


async def analyze(
    video_path: str,
    region_box: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Measure motion, run two vision passes, then have the agent write the report.

    ``region_box`` (normalized l,t,r,b) restricts analysis to a user-picked area.
    """
    from openjiuwen.core.runner import Runner

    runtime.setup()
    config.validate()
    if not Path(video_path).exists():
        raise SystemExit(f"Video not found: {video_path}")

    await _ensure_runner()
    store.set_current_video(video_path)
    profile = store.get_profile()

    meta, frames, timestamps = sampling.sample_frames(video_path, config.max_frames(), config.use_cache())
    if not frames:
        raise SystemExit("No frames could be extracted from the video.")

    steps = motion.activity(frames, timestamps)
    ball_info = ball.detect(frames, timestamps, steps)
    metrics = motion.analyze(frames, timestamps, ball=ball_info, region_box=region_box)
    artifact_paths = _save_artifacts(video_path, frames, timestamps, metrics, region_box)

    backend = get_backend()
    coarse_out = await vision.coarse(meta, frames, timestamps, metrics, profile, backend=backend)
    windows = _clean_windows(coarse_out.get("attentive_windows"), timestamps, config.max_windows())
    zoom_frames, zoom_times = _zoom_frames(video_path, windows, config.zoom_frames())
    fine_out = await vision.fine(
        frames, timestamps, metrics, windows, zoom_frames, zoom_times, profile, backend=backend
    )
    observations = vision.observations_text(coarse_out, fine_out)

    prev_progress = progress.summarize(store.get_reports())
    progress_note = prev_progress.get("text", "") if prev_progress else "No previous report."
    repeated = ", ".join(prev_progress.get("repeated_themes", [])) if prev_progress else ""

    window_label = ", ".join(f"{w['start']}-{w['end']}s" for w in windows) or "none"

    text_usage = usage_mod.UsageCollector()
    agent = build_agent(model=usage_mod.attach(config.make_model(), text_usage))
    query = (
        f"Today is {date.today().isoformat()}. "
        f"Write my coaching report for a {profile.get('sport', 'table tennis')} session.\n"
        f"Player: level {profile.get('level', 'unknown')}, "
        f"{profile.get('dominant_hand', 'right')}-handed, working on: "
        f"{profile.get('goal', 'improve')}.\n\n"
        f"History: {progress_note}" + (f" Recurring themes: {repeated}." if repeated else "") + "\n\n"
        f"Analysis ({len(frames)} sampled frames, windows examined: {window_label}):\n"
        f"{observations}\n\n"
        f"Measured motion/mechanics:\n{motion.metrics_text(metrics)}\n\n"
        "Save the report with save_report (schema with evidence_times and confidence), "
        "then give me a short summary."
    )
    result = await Runner.run_agent(agent, {"query": query})

    vision_usage = backend.usage_summary() if hasattr(backend, "usage_summary") else {}
    text = text_usage.summary()
    usage = {
        "vision": vision_usage,
        "text": text,
        "calls": int(vision_usage.get("calls", 0)) + int(text.get("calls", 0)),
        "total_tokens": int(vision_usage.get("total_tokens", 0)) + int(text.get("total_tokens", 0)),
    }

    report = store.patch_last_report(
        {
            "observations": observations,
            "metrics": metrics,
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
        trend = progress.summarize(store.get_reports())
        try:
            exports = export.write(report)
        except Exception:
            exports = {}
        report = store.patch_last_report({"progress_trend": trend, "exports": exports})

    return {
        "result": result,
        "report": report or {},
        "report_path": str(store.report_path(video_path)),
        "observations": observations,
        "frames": len(frames),
        "meta": meta,
    }
