"""Dependency-free motion, subject and coarse-mechanics analysis.

Turns raw frames into objective numbers restricted to the moving subject:
movement energy, global shift, subject box/track, upper/lower body region
motion (a footwork proxy) and a crouch trend. The vision model then reasons
over measurements instead of eyeballing isolated photos.

If ``mediapipe`` is installed, :func:`pose_estimate` returns real joints; without
it the region/box metrics below are the pose proxy.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from topspin_review.perception import cvutil, pose

_ANALYSIS_SIZE = (320, 180)


def _gray(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(img.convert("L").resize(size), dtype=np.float32)


def subject_region(frames: list[Image.Image], size: tuple[int, int] = _ANALYSIS_SIZE) -> np.ndarray | None:
    """Foreground region of the **primary** subject.

    Background = per-pixel median across the sampled frames. The union of the
    foreground is reduced to its largest connected component, so other people
    elsewhere in the hall are dropped from the motion map and metrics.
    """
    if len(frames) < 2:
        return None
    grays = np.stack([_gray(f, size) for f in frames])
    background = np.median(grays, axis=0)
    diff = np.abs(grays - background)
    mask = diff.max(axis=0) > 18
    if not mask.any():
        return None
    primary = cvutil.largest(mask, min_area=max(20, int(mask.sum() * 0.05)))
    return primary if primary is not None else mask


def motion_map(
    frames: list[Image.Image], size: tuple[int, int] = (640, 360), mask: np.ndarray | None = None
) -> Image.Image | None:
    """Heatmap of where the image changed, restricted to the subject region."""
    if len(frames) < 2:
        return None
    grays = [_gray(f, size) for f in frames]
    acc = np.zeros_like(grays[0])
    for prev, cur in zip(grays, grays[1:]):
        acc += np.abs(cur - prev)
    if mask is not None:
        m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(size)) > 127
        acc = np.where(m, acc, 0.0)
    peak = float(acc.max())
    if peak <= 0:
        return None
    acc /= peak

    base = np.asarray(frames[0].convert("RGB").resize(size), dtype=np.float32)
    out = base.copy()
    out[..., 0] = np.clip(base[..., 0] + acc * 200.0, 0, 255)
    out[..., 1] = np.clip(base[..., 1] * (1.0 - 0.6 * acc), 0, 255)
    out[..., 2] = np.clip(base[..., 2] * (1.0 - 0.6 * acc), 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def phase_shift(prev: Image.Image, cur: Image.Image, size: tuple[int, int] = _ANALYSIS_SIZE) -> tuple[float, float]:
    """Global translation (dx, dy) from ``prev`` to ``cur`` (phase correlation)."""
    a = _gray(prev, size)
    b = _gray(cur, size)
    a = a - a.mean()
    b = b - b.mean()
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-6)
    corr = np.fft.irfft2(cross, s=a.shape)
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return float(dx), float(dy)


def activity(frames: list[Image.Image], timestamps: list[float], mask: np.ndarray | None = None) -> list[dict]:
    """Per-step motion energy and shift; energy is averaged over the subject region."""
    size = _ANALYSIS_SIZE
    m = None
    if mask is not None:
        m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(size)) > 127
    steps: list[dict] = []
    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        a = _gray(prev, size)
        b = _gray(cur, size)
        diff = np.abs(b - a)
        energy = float(diff[m].mean()) if m is not None and m.any() else float(diff.mean())
        dx, dy = phase_shift(prev, cur)
        steps.append(
            {"t": round(float(timestamps[i]), 2), "energy": round(energy, 2), "dx": round(dx, 1), "dy": round(dy, 1)}
        )
    return steps


def subject_track(frames: list[Image.Image], region: np.ndarray | None = None) -> list[dict | None]:
    """Coarse subject position each frame (normalized centre, box, coverage)."""
    if len(frames) < 2:
        return [None for _ in frames]
    w, h = _ANALYSIS_SIZE
    grays = np.stack([_gray(f, _ANALYSIS_SIZE) for f in frames])
    background = np.median(grays, axis=0)

    track: list[dict | None] = []
    for g in grays:
        mask = np.abs(g - background) > 18
        if region is not None:
            mask = mask & region
        ys, xs = np.where(mask)
        if xs.size < 20:
            track.append(None)
            continue
        track.append(
            {
                "cx": round(float(xs.mean() / w), 3),
                "cy": round(float(ys.mean() / h), 3),
                "w": round(float((xs.max() - xs.min()) / w), 3),
                "h": round(float((ys.max() - ys.min()) / h), 3),
                "coverage": round(float(mask.mean()), 3),
            }
        )
    return track


def regions(frames: list[Image.Image], region: np.ndarray | None) -> list[dict | None]:
    """Per-frame centroid/width of the upper, middle and lower body thirds.

    Foreground is recomputed **per frame** and restricted to the primary-subject
    region, so the lower-third horizontal position tracks footwork over time (the
    pooled subject mask would give a constant centroid).
    """
    if len(frames) < 2 or region is None:
        return [None for _ in frames]
    w, h = _ANALYSIS_SIZE
    band_edges = [(0, h // 3), (h // 3, 2 * h // 3), (2 * h // 3, h)]
    grays = np.stack([_gray(f, _ANALYSIS_SIZE) for f in frames])
    background = np.median(grays, axis=0)

    out: list[dict | None] = []
    for g in grays:
        fg = (np.abs(g - background) > 18) & region
        rows = {}
        for name, (top, bottom) in zip(("upper", "middle", "lower"), band_edges):
            band = fg[top:bottom, :]
            ys, xs = np.where(band)
            if xs.size < 10:
                rows[name] = None
                continue
            rows[name] = {
                "cx": round(float(xs.mean() / w), 3),
                "cy": round(float((top + ys.mean()) / h), 3),
                "w": round(float((xs.max() - xs.min()) / w), 3),
            }
        out.append(rows)
    return out


def mechanics(region_track: list[dict | None]) -> dict:
    """Derive footwork/crouch proxies from the region track."""
    lowers = [r["lower"] for r in region_track if r and r.get("lower")]
    uppers = [r["upper"] for r in region_track if r and r.get("upper")]
    mids = [r["middle"] for r in region_track if r and r.get("middle")]
    if not lowers:
        return {
            "lower_lateral_range": 0.0,
            "lower_vertical_range": 0.0,
            "upper_lateral_range": 0.0,
            "lower_direction_changes": 0,
            "crouch_box_range": 0.0,
        }
    lower_x = [b["cx"] for b in lowers]
    lower_y = [b["cy"] for b in lowers]
    upper_x = [b["cx"] for b in uppers] or lower_x
    # direction changes in lower-body horizontal position ~ steps / shuffles
    deltas = [lower_x[i] - lower_x[i - 1] for i in range(1, len(lower_x))]
    changes = sum(1 for i in range(1, len(deltas)) if deltas[i] * deltas[i - 1] < 0 and abs(deltas[i]) > 0.005)
    return {
        "lower_lateral_range": round(max(lower_x) - min(lower_x), 3),
        "lower_vertical_range": round(max(lower_y) - min(lower_y), 3),
        "upper_lateral_range": round(max(upper_x) - min(upper_x), 3),
        "lower_direction_changes": changes,
        "crouch_box_range": round(max(b["w"] for b in mids) - min(b["w"] for b in mids), 3) if mids else 0.0,
    }


def posture(track: list[dict | None]) -> dict:
    boxes = [t for t in track if t]
    if not boxes:
        return {}
    cxs = [b["cx"] for b in boxes]
    widths = [b["w"] for b in boxes]
    heights = [b["h"] for b in boxes]
    return {
        "box_height_mean": round(float(np.mean(heights)), 3),
        "box_width_mean": round(float(np.mean(widths)), 3),
        "aspect_mean": round(float(np.mean(heights) / max(np.mean(widths), 1e-6)), 2),
        "lateral_range": round(float(max(cxs) - min(cxs)), 3),
        "coverage_mean": round(float(np.mean([b["coverage"] for b in boxes])), 3),
    }


def pose_estimate(frames: list[Image.Image]) -> list | None:
    """Optional real pose (joints). Returns ``None`` unless ``mediapipe`` is installed."""
    return pose.estimate(frames)


def region_from_box(box: tuple[float, float, float, float], size: tuple[int, int] = _ANALYSIS_SIZE) -> np.ndarray:
    """Boolean mask for a normalized (left, top, right, bottom) box in 0..1."""
    left, top, right, bottom = box
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    mask[int(top * h) : int(bottom * h), int(left * w) : int(right * w)] = True
    return mask


def analyze(
    frames: list[Image.Image],
    timestamps: list[float],
    ball: dict | None = None,
    region_box: tuple[float, float, float, float] | None = None,
) -> dict:
    """Aggregate all motion/mechanics metrics for one sampled frame sequence.

    ``region_box`` (normalized l,t,r,b) overrides the auto primary-subject mask.
    """
    mask = region_from_box(region_box) if region_box else subject_region(frames)
    steps = activity(frames, timestamps, mask)
    track = subject_track(frames, mask)
    region_track = regions(frames, mask)
    pose_seq = pose_estimate(frames)
    energies = [s["energy"] for s in steps]
    return {
        "activity": steps,
        "track": track,
        "posture": posture(track),
        "mechanics": mechanics(region_track),
        "pose": pose_seq,
        "pose_summary": pose.summarize(pose_seq),
        "subject_coverage": round(float(mask.mean()), 3) if mask is not None else None,
        "ball": ball or {},
        "peak_motion_time": max(steps, key=lambda s: s["energy"])["t"] if steps else None,
        "mean_energy": round(float(np.mean(energies)), 2) if energies else 0.0,
        "max_energy": round(float(np.max(energies)), 2) if energies else 0.0,
        "net_shift": [round(sum(s["dx"] for s in steps), 1), round(sum(s["dy"] for s in steps), 1)]
        if steps
        else [0.0, 0.0],
    }


def text(metrics: dict) -> str:
    """Render a metrics dict as compact text for prompts."""
    net = metrics.get("net_shift", [0, 0])
    lines = [
        f"- overall motion: mean energy {metrics.get('mean_energy')}, "
        f"peak {metrics.get('max_energy')} at t={metrics.get('peak_motion_time')}s, "
        f"net shift (dx,dy)=({net[0]}, {net[1]}) px"
    ]
    steps = metrics.get("activity") or []
    if steps:
        rendered = "; ".join(f"t={s['t']}s E={s['energy']} shift=({s['dx']},{s['dy']})" for s in steps)
        lines.append(f"- activity timeline: {rendered}")
    post = metrics.get("posture") or {}
    if post:
        lines.append(
            "- subject box: mean height {box_height_mean} width {box_width_mean} (aspect {aspect_mean}), "
            "lateral drift {lateral_range}".format(**post)
        )
    mech = metrics.get("mechanics") or {}
    if mech:
        lines.append(
            "- mechanics: lower-body lateral range {lower_lateral_range}, direction changes "
            "(step proxy) {lower_direction_changes}, upper-body lateral range {upper_lateral_range}, "
            "crouch change {crouch_box_range}".format(**mech)
        )
    ball = metrics.get("ball") or {}
    if ball:
        lines.append(f"- ball (heuristic): {ball.get('summary', 'n/a')}")
    pose_summary = metrics.get("pose_summary") or {}
    if pose_summary:
        rendered = ", ".join(f"{k}={v}" for k, v in pose_summary.items())
        lines.append(f"- pose joints (degrees/normalized): {rendered}")
    return "\n".join(lines)
