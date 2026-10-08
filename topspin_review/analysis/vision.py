"""Two-pass vision analysis via a vision DeepAgent.

The openjiuwen vision agent reads the rendered images (contact sheet, motion map,
pose overlay, zoom frames) with its ``read_file`` tool (native multimodal) and
returns structured JSON. Prompt text lives in
:mod:`topspin_review.analysis.prompts`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from topspin_review.analysis import prompts
from topspin_review.analysis.session import run_agent
from topspin_review.perception import imaging, metrics, pose


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


def _output_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, dict):
        for key in ("output", "content", "answer", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        return str(result)
    return getattr(result, "content", None) or str(result)


async def _ask(agent: Any, prompt: str) -> dict:
    """Run the vision agent and parse its JSON reply."""
    result = await run_agent(agent, prompt)
    return extract_json(_output_text(result))


def _save(image: Image.Image, media_dir: Path, name: str) -> str:
    path = media_dir / name
    imaging.save_png(image, path)
    return str(path)


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

    sheet = imaging.contact_sheet(frames, timestamps, cols=min(4, len(frames)))
    sheet_path = _save(sheet, media_dir, "overview_sheet.png")
    prompt = prompts.COARSE_PROMPT.format(
        n=len(frames),
        sport=profile.get("sport", "table tennis"),
        level=profile.get("level", "unknown"),
        hand=profile.get("dominant_hand", "right"),
        goal=profile.get("goal", "improve"),
        metrics=prompts.metrics_text(measured),
        duration=_duration(meta, timestamps),
    )
    parsed = await _ask(agent, f"{prompt}\n\nFirst call read_file on: {sheet_path}")
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

    paths: list[str] = []
    if frames:
        sheet = imaging.contact_sheet(frames, timestamps, cols=min(4, len(frames)))
        paths.append(_save(sheet, media_dir, "detail_sheet.png"))
    if len(frames) >= 2:
        mm = metrics.motion_map(frames)
        if mm is not None:
            paths.append(_save(mm, media_dir, "motion_map.png"))
    if measured.get("pose") and frames:
        annotated = pose.overlay(frames, measured["pose"])
        if annotated:
            pose_sheet = imaging.contact_sheet(annotated, timestamps, cols=min(4, len(annotated)))
            paths.append(_save(pose_sheet, media_dir, "pose_sheet.png"))
    for i, zoom in enumerate(zoom_frames):
        paths.append(_save(zoom, media_dir, f"zoom_{i}.png"))

    rendered = ", ".join(f"[{w.get('start')}s-{w.get('end')}s]" for w in windows) or "none"
    prompt = prompts.FINE_PROMPT.format(
        z=len(zoom_frames),
        level=profile.get("level", "unknown"),
        hand=profile.get("dominant_hand", "right"),
        goal=profile.get("goal", "improve"),
        metrics=prompts.metrics_text(measured),
        windows=rendered,
    )
    files = "\n".join(f"read_file: {p}" for p in paths)
    parsed = await _ask(agent, f"{prompt}\n\nRead these images:\n{files}")
    parsed.setdefault("observations", "")
    parsed.setdefault("signals", [])
    parsed.setdefault("limitations", [])
    return parsed


async def analyze_still(frame: Image.Image, profile: dict, *, agent: Any, media_dir: Path) -> dict:
    """Single-image pass: describe the static posture only."""
    still_path = _save(frame, media_dir, "still.png")
    prompt = prompts.STILL_PROMPT.format(
        sport=profile.get("sport", "table tennis"),
        level=profile.get("level", "unknown"),
        hand=profile.get("dominant_hand", "right"),
        goal=profile.get("goal", "improve"),
    )
    parsed = await _ask(agent, f"{prompt}\n\nFirst call read_file on: {still_path}")
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
