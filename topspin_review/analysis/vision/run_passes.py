"""The vision analysis passes: overview, detail and still.

Each pass composes a prompt over the rendered images (``vision.render_images``), asks the
vision agent (``vision.ask_agent``) and returns structured JSON. Prompt text lives in
:mod:`topspin_review.analysis.vision.prompts`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from topspin_review.analysis.vision import prompts, render_images
from topspin_review.analysis.vision.ask_agent import ask
from topspin_review.perception import metrics


def _duration(meta: dict, timestamps: list[float]) -> float:
    value = meta.get("duration")
    if value:
        return float(value)
    return round(max(timestamps), 2) if timestamps else 0.0


async def coarse(
    meta: dict,
    frames: list[Image.Image],
    timestamps: list[float],
    measured: dict,
    profile: dict,
    *,
    agent: Any,
    media_dir: Path,
) -> dict:
    """Overview pass: summarize the clip and propose windows to zoom into."""
    if not frames:
        return {"overall": "No frames could be extracted.", "attentive_windows": [], "limitations": []}

    sheet_path = render_images.overview_sheet(frames, timestamps, media_dir)
    prompt = prompts.COARSE_PROMPT.format(
        n=len(frames),
        sport=profile.get("sport", "table tennis"),
        level=profile.get("level", "unknown"),
        hand=profile.get("dominant_hand", "right"),
        goal=profile.get("goal", "improve"),
        metrics=metrics.text(measured),
        duration=_duration(meta, timestamps),
    )
    parsed = await ask(agent, f"{prompt}\n\nFirst call read_file on: {sheet_path}")
    parsed.setdefault("overall", "")
    parsed.setdefault("attentive_windows", [])
    parsed.setdefault("limitations", [])
    return parsed


async def fine(
    frames: list[Image.Image],
    timestamps: list[float],
    measured: dict,
    windows: list[dict],
    zoom_frames: list[Image.Image],
    zoom_times: list[float],
    profile: dict,
    *,
    agent: Any,
    media_dir: Path,
) -> dict:
    """Detail pass over the proposed windows; returns structured observations."""
    if not frames and not zoom_frames:
        return {"observations": "", "signals": [], "limitations": []}

    paths = render_images.detail_images(frames, timestamps, measured, zoom_frames, media_dir)
    rendered = ", ".join(f"[{w.get('start')}s-{w.get('end')}s]" for w in windows) or "none"
    prompt = prompts.FINE_PROMPT.format(
        z=len(zoom_frames),
        level=profile.get("level", "unknown"),
        hand=profile.get("dominant_hand", "right"),
        goal=profile.get("goal", "improve"),
        metrics=metrics.text(measured),
        windows=rendered,
    )
    files = "\n".join(f"read_file: {p}" for p in paths)
    parsed = await ask(agent, f"{prompt}\n\nRead these images:\n{files}")
    parsed.setdefault("observations", "")
    parsed.setdefault("signals", [])
    parsed.setdefault("limitations", [])
    return parsed


async def analyze_still(frame: Image.Image, profile: dict, *, agent: Any, media_dir: Path) -> dict:
    """Single-image pass: describe the static posture only."""
    still_path = render_images.still_image(frame, media_dir)
    prompt = prompts.STILL_PROMPT.format(
        sport=profile.get("sport", "table tennis"),
        level=profile.get("level", "unknown"),
        hand=profile.get("dominant_hand", "right"),
        goal=profile.get("goal", "improve"),
    )
    parsed = await ask(agent, f"{prompt}\n\nFirst call read_file on: {still_path}")
    parsed.setdefault("observations", "")
    parsed.setdefault("signals", [])
    parsed.setdefault("limitations", [])
    return parsed


def observations_text(coarse_out: dict, fine_out: dict) -> str:
    """Flatten both passes into one block for the report-writing agent."""
    lines = []
    if coarse_out.get("overall"):
        lines.append(f"Overview: {coarse_out['overall']}")
    if fine_out.get("observations"):
        lines.append(f"Detailed observations: {fine_out['observations']}")
    signals = fine_out.get("signals") or []
    if signals:
        lines.append("Signals (with evidence):")
        for s in signals:
            times = ", ".join(str(t) for t in (s.get("evidence_times") or []))
            lines.append(f"  - {s.get('signal')} (t={times}; confidence {s.get('confidence')})")
    limitations = list(coarse_out.get("limitations") or []) + list(fine_out.get("limitations") or [])
    if limitations:
        lines.append("Limitations:")
        for item in limitations:
            lines.append(f"  - {item}")
    return "\n".join(lines)
