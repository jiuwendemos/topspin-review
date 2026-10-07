"""Two-pass vision analysis: a coarse overview, then a zoom into busy windows.

The vision model receives a compact contact sheet, the raw motion/posture metrics
and a motion map. First it proposes the time windows worth a closer look; the
agent re-samples those windows and the model returns structured observations with
evidence timestamps and confidence. It is told never to invent measurements.
Prompt text lives in :mod:`topspin_review.analysis.prompts`.
"""

from __future__ import annotations

import json
import re

from PIL import Image

from topspin_review import config
from topspin_review.analysis import prompts
from topspin_review.perception import imaging, metrics, pose
from topspin_review.providers import get_backend


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a model reply."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


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
    backend=None,
) -> dict:
    """Overview pass: summarize the clip and propose windows to zoom into."""
    if not frames:
        return {"overall": "No frames could be extracted.", "attentive_windows": [], "limitations": []}

    backend = backend or get_backend()
    sheet = imaging.contact_sheet(frames, timestamps, cols=min(4, len(frames)))
    prompt = prompts.COARSE_PROMPT.format(
        n=len(frames),
        sport=profile.get("sport", "table tennis"),
        level=profile.get("level", "unknown"),
        hand=profile.get("dominant_hand", "right"),
        goal=profile.get("goal", "improve"),
        metrics=prompts.metrics_text(measured),
        duration=_duration(meta, timestamps),
    )
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": imaging.to_data_url(sheet, max_width=config.max_width())}},
    ]
    text = await backend.complete(
        [{"role": "system", "content": prompts.VISION_SYSTEM}, {"role": "user", "content": content}]
    )
    parsed = extract_json(text)
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
    backend=None,
) -> dict:
    """Detail pass over the proposed windows; returns structured observations."""
    if not frames and not zoom_frames:
        return {"observations": "", "signals": [], "limitations": []}

    backend = backend or get_backend()
    sheet = imaging.contact_sheet(frames, timestamps, cols=min(4, len(frames))) if frames else None
    rendered = ", ".join(f"[{w.get('start')}s-{w.get('end')}s]" for w in windows) or "none"
    prompt = prompts.FINE_PROMPT.format(
        z=len(zoom_frames),
        level=profile.get("level", "unknown"),
        hand=profile.get("dominant_hand", "right"),
        goal=profile.get("goal", "improve"),
        metrics=prompts.metrics_text(measured),
        windows=rendered,
    )

    content: list[dict] = [{"type": "text", "text": prompt}]
    if sheet is not None:
        content.append(
            {"type": "image_url", "image_url": {"url": imaging.to_data_url(sheet, max_width=config.max_width())}}
        )
    mm = metrics.motion_map(frames) if len(frames) >= 2 else None
    if mm is not None:
        content.append({"type": "text", "text": "Motion map (red = movement across the clip):"})
        content.append({"type": "image_url", "image_url": {"url": imaging.to_data_url(mm, max_width=config.max_width())}})
    if measured.get("pose"):
        annotated = pose.overlay(frames, measured["pose"])
        if annotated:
            content.append({"type": "text", "text": "Pose skeleton overlay:"})
            sheet_pose = imaging.contact_sheet(annotated, timestamps, cols=min(4, len(annotated)))
            content.append(
                {"type": "image_url", "image_url": {"url": imaging.to_data_url(sheet_pose, max_width=config.max_width())}}
            )
    if zoom_frames:
        content.append(
            {"type": "text", "text": f"Zoomed frames from the windows ({', '.join(f'{t:.2f}s' for t in zoom_times)}):"}
        )
        for url in imaging.frames_to_data_urls(zoom_frames, max_width=config.max_width()):
            content.append({"type": "image_url", "image_url": {"url": url}})

    text = await backend.complete(
        [{"role": "system", "content": prompts.VISION_SYSTEM}, {"role": "user", "content": content}]
    )
    parsed = extract_json(text)
    parsed.setdefault("observations", text or "")
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
