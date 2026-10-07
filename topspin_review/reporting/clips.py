"""Cut short highlight clips around flagged moments using ffmpeg.

Best-effort: returns ``{timestamp: clip_path}`` for the clips it manages to make.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from topspin_review.storage import runtime


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def make_clip(video: str, start: float, end: float, out_path: Path) -> bool:
    start = max(0.0, float(start))
    end = max(start + 0.5, float(end))
    try:
        exe = _ffmpeg()
        # Fast path: stream copy (works when cuts land on keyframes).
        subprocess.run(
            [exe, "-y", "-ss", f"{start}", "-to", f"{end}", "-i", str(video), "-c", "copy", str(out_path)],
            capture_output=True,
        )
        if out_path.exists() and out_path.stat().st_size > 0:
            return True
        # Fallback: re-encode (accurate but slower).
        subprocess.run(
            [
                exe, "-y", "-ss", f"{start}", "-i", str(video), "-t", f"{end - start}",
                "-c:v", "libx264", "-preset", "veryfast", "-an", str(out_path),
            ],
            capture_output=True,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def make_clips(video: str, times: list[float], stem: str, before: float = 1.0, after: float = 2.0, limit: int = 3) -> dict:
    """Create up to ``limit`` short clips centred on ``times``."""
    clips: dict[str, str] = {}
    if not times:
        return clips
    out_dir = runtime.ARTIFACTS_DIR / f"{stem}_clips"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return clips
    for i, t in enumerate(sorted(set(times))[:limit]):
        out = out_dir / f"moment_{i}_{round(float(t), 1)}s.mp4"
        if make_clip(video, float(t) - before, float(t) + after, out):
            clips[str(round(float(t), 1))] = str(out)
    return clips
