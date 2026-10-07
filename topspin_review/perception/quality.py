"""Footage-quality gate: is this clip good enough to coach from?

Turns video metadata + measured signals into a score and plain warnings, so the
pipeline can tell the user (and the model) when the footage limits the feedback.
"""

from __future__ import annotations


def evaluate(meta: dict, frames: list, metrics: dict) -> dict:
    """Return ``{score, usable, warnings}`` for the analyzed clip."""
    warnings: list[str] = []
    score = 100

    try:
        fps = float(meta.get("fps") or 0)
    except (TypeError, ValueError):
        fps = 0.0
    size = meta.get("size") or (0, 0)
    try:
        duration = float(meta.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0

    still = len(frames) == 1 and not fps and not duration
    if isinstance(size, (list, tuple)) and len(size) == 2 and size[1] and size[1] < 480:
        warnings.append("low resolution")
        score -= 20

    if still:
        warnings.append("single still image — only static posture can be judged (no movement/footwork)")
        score -= 15
    else:
        if fps and fps < 25:
            warnings.append(f"low frame rate ({fps:.0f} fps) — fast footwork may be missed")
            score -= 25
        if duration and duration < 3:
            warnings.append("very short clip (a few rallies are better)")
            score -= 15
        if len(frames) < 8:
            warnings.append("few frames sampled")
            score -= 10
        if not metrics.get("mean_energy"):
            warnings.append("little visible movement detected")
            score -= 30

    if metrics.get("subject_coverage") is None and not still:
        warnings.append("could not isolate the player from the background")
        score -= 20

    score = max(0, score)
    return {"score": score, "usable": score >= 40, "warnings": warnings}
