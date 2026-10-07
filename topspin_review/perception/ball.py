"""Heuristic ball detection, tracking and rally/contact segmentation.

No trained detector is available, so the ball is found as a small, bright,
warm/white blob and linked frame-to-frame with a greedy nearest-neighbour
tracker. Direction changes along the track are reported as candidate
contacts/bounces. Everything is labelled heuristic and low confidence.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from topspin_review.perception import cvutil

_SIZE = (160, 90)
_MIN_AREA = 2
_MAX_AREA = 200
_MAX_STEP = 0.16  # normalized per-frame link distance
_MAX_MISSED = 2


def _rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize(_SIZE), dtype=np.float32) / 255.0


def _color_mask(img: Image.Image) -> np.ndarray:
    hsv = cvutil.rgb_to_hsv(_rgb(img))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm = (h > 0.02) & (h < 0.17) & (s > 0.35) & (v > 0.65)
    white = (s < 0.25) & (v > 0.85)
    return (warm | white) & (v > 0.6)


def candidates(frame: Image.Image) -> list[dict]:
    """Roundish, bright, small blobs that could be the ball."""
    out: list[dict] = []
    for comp in cvutil.components(_color_mask(frame)):
        area = comp["area"]
        if area < _MIN_AREA or area > _MAX_AREA:
            continue
        x0, y0, x1, y1 = comp["bbox"]
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        aspect = bw / max(bh, 1)
        fill = area / max(bw * bh, 1)
        if 0.4 <= aspect <= 2.5 and fill >= 0.4:
            out.append({"cx": comp["cx"], "cy": comp["cy"], "area": area})
    return out


def track(frames: list[Image.Image], timestamps: list[float]) -> list[dict]:
    """Greedy nearest-neighbour tracking; returns tracks as time-ordered points."""
    tracks: list[dict] = []
    for i, frame in enumerate(frames):
        cands = candidates(frame)
        for tr in tracks:
            tr["matched"] = False
        for cand in cands:
            best = None
            best_d = _MAX_STEP
            for tr in tracks:
                if tr["matched"]:
                    continue
                d = ((tr["last"][0] - cand["cx"]) ** 2 + (tr["last"][1] - cand["cy"]) ** 2) ** 0.5
                if d < best_d:
                    best_d, best = d, tr
            if best is not None:
                best["points"].append({"t": round(float(timestamps[i]), 2), "cx": cand["cx"], "cy": cand["cy"]})
                best["last"] = (cand["cx"], cand["cy"])
                best["matched"] = True
                best["missed"] = 0
            else:
                tracks.append(
                    {
                        "points": [{"t": round(float(timestamps[i]), 2), "cx": cand["cx"], "cy": cand["cy"]}],
                        "last": (cand["cx"], cand["cy"]),
                        "matched": True,
                        "missed": 0,
                    }
                )
        for tr in tracks:
            if not tr["matched"]:
                tr["missed"] += 1
        tracks = [tr for tr in tracks if tr["missed"] <= _MAX_MISSED]
    tracks = [tr for tr in tracks if len(tr["points"]) >= 3]
    tracks.sort(key=lambda tr: len(tr["points"]), reverse=True)
    return tracks


def contacts(main_track: dict) -> list[float]:
    """Times where the tracked ball reverses direction (hit/bounce proxy)."""
    pts = main_track["points"]
    events: list[float] = []
    for i in range(2, len(pts)):
        vx1 = pts[i - 1]["cx"] - pts[i - 2]["cx"]
        vy1 = pts[i - 1]["cy"] - pts[i - 2]["cy"]
        vx2 = pts[i]["cx"] - pts[i - 1]["cx"]
        vy2 = pts[i]["cy"] - pts[i - 1]["cy"]
        speed = (vx2 * vx2 + vy2 * vy2) ** 0.5
        reverse_x = vx1 * vx2 < 0 and abs(vx1) > 0.005 and abs(vx2) > 0.005
        reverse_y = vy1 * vy2 < 0 and abs(vy1) > 0.005 and abs(vy2) > 0.005
        if (reverse_x or reverse_y) and speed > 0.01:
            events.append(pts[i]["t"])
    return events


def rally_segments(activity: list[dict], timestamps: list[float]) -> list[dict]:
    """Contiguous spans where movement energy is clearly above the pause level."""
    energies = [s["energy"] for s in activity]
    if not energies:
        return []
    baseline = float(np.median(energies)) or 0.0
    threshold = max(baseline * 1.3, baseline + 0.5)
    segments: list[dict] = []
    start = None
    for s in activity:
        active = s["energy"] >= threshold
        if active and start is None:
            start = s["t"]
        elif not active and start is not None:
            segments.append({"start": start, "end": s["t"]})
            start = None
    if start is not None:
        end = round(float(timestamps[-1]), 2) if timestamps else start
        segments.append({"start": start, "end": end})
    return [seg for seg in segments if seg["end"] - seg["start"] >= 0.4]


def detect(frames: list[Image.Image], timestamps: list[float], activity: list[dict]) -> dict:
    """Aggregate heuristic ball/track/contact/rally information."""
    tracks = track(frames, timestamps)
    main = tracks[0] if tracks else None
    contact_times = contacts(main) if main else []
    rallies = rally_segments(activity, timestamps)
    visible = len({p["t"] for p in (main["points"] if main else [])})

    if not main and not rallies:
        summary = "no ball-like object or rally segments detected (heuristic, low confidence)"
    else:
        rally_txt = ", ".join(f"{r['start']}-{r['end']}s" for r in rallies) or "none"
        bits = [f"rally segment(s) {rally_txt}"]
        if main:
            bits.append(f"ball tracked in {visible} frames")
        if contact_times:
            bits.append(f"~{len(contact_times)} direction change(s) at {', '.join(f'{t}s' for t in contact_times[:6])}")
        summary = "; ".join(bits) + " (heuristic, low confidence)"

    return {
        "summary": summary,
        "tracks": [{"points": t["points"]} for t in tracks[:3]],
        "contacts": contact_times,
        "rallies": rallies,
    }
