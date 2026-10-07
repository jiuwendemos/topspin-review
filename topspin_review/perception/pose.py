"""Optional real pose estimation via ``mediapipe``.

Installed via ``requirements.txt`` (``mediapipe``). When available, :func:`estimate`
returns per-frame joint metrics (knee/elbow angles, stance width); otherwise the
caller falls back to the dependency-free region proxy in
:mod:`topspin_review.motion`.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

# mediapipe Pose landmark indices.
_L = {
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow": 13,
    "r_elbow": 14,
    "l_wrist": 15,
    "r_wrist": 16,
    "l_hip": 23,
    "r_hip": 24,
    "l_knee": 25,
    "r_knee": 26,
    "l_ankle": 27,
    "r_ankle": 28,
}

# Skeleton edges drawn by :func:`overlay` (subset of mediapipe's connections).
_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


def available() -> bool:
    try:
        import mediapipe  # noqa: F401
    except Exception:
        return False
    return True


def _angle(a, b, c) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return round(math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2))))), 1)


def estimate(frames: list[Image.Image]) -> list[dict] | None:
    """Per-frame joint metrics, or ``None`` when mediapipe is unavailable."""
    if not available():
        return None

    import mediapipe as mp

    pose = mp.solutions.pose
    out: list[dict] = []
    with pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as estimator:
        for frame in frames:
            result = estimator.process(np.asarray(frame.convert("RGB")))
            if not result.pose_landmarks:
                out.append({})
                continue
            lm = result.pose_landmarks.landmark
            pt = {name: (lm[idx].x, lm[idx].y) for name, idx in _L.items()}
            out.append(
                {
                    "l_knee": _angle(pt["l_hip"], pt["l_knee"], pt["l_ankle"]),
                    "r_knee": _angle(pt["r_hip"], pt["r_knee"], pt["r_ankle"]),
                    "l_elbow": _angle(pt["l_shoulder"], pt["l_elbow"], pt["l_wrist"]),
                    "r_elbow": _angle(pt["r_shoulder"], pt["r_elbow"], pt["r_wrist"]),
                    "stance_width": round(abs(pt["l_ankle"][0] - pt["r_ankle"][0]), 3),
                    "crouch": round(
                        (pt["l_hip"][1] + pt["r_hip"][1]) / 2 - (pt["l_ankle"][1] + pt["r_ankle"][1]) / 2, 3
                    ),
                    "landmarks": [(p.x, p.y, p.visibility) for p in lm],
                }
            )
    return out


def overlay(frames: list[Image.Image], pose: list | None) -> list[Image.Image]:
    """Draw the skeleton on each frame (no-op without landmarks)."""
    if not pose:
        return []
    out: list[Image.Image] = []
    for frame, p in zip(frames, pose):
        img = frame.copy()
        landmarks = (p or {}).get("landmarks")
        if landmarks:
            w, h = img.size
            draw = ImageDraw.Draw(img)
            for a, b in _CONNECTIONS:
                if a >= len(landmarks) or b >= len(landmarks):
                    continue
                xa, ya, va = landmarks[a]
                xb, yb, vb = landmarks[b]
                if va < 0.4 or vb < 0.4:
                    continue
                draw.line([(xa * w, ya * h), (xb * w, yb * h)], fill=(0, 255, 60), width=max(2, w // 320))
            for x, y, vis in landmarks:
                if vis < 0.4:
                    continue
                r = max(2, w // 400)
                draw.ellipse([x * w - r, y * h - r, x * w + r, y * h + r], fill=(255, 90, 0))
        out.append(img)
    return out


def summarize(pose: list[dict] | None) -> dict:
    if not pose:
        return {}
    valid = [p for p in pose if p]
    if not valid:
        return {}
    keys = ("l_knee", "r_knee", "l_elbow", "r_elbow", "stance_width", "crouch")
    summary = {}
    for k in keys:
        values = [p[k] for p in valid if k in p]
        if values:
            summary[f"{k}_mean"] = round(sum(values) / len(values), 2)
            summary[f"{k}_range"] = round(max(values) - min(values), 2)
    return summary
