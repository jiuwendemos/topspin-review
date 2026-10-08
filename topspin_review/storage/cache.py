"""On-disk cache for sampled frames, keyed by file identity and sampling params.

Kept separate from :mod:`topspin_review.analysis.video.sampling` (which only decides
*which* frames to read); this module owns the filesystem cache under
``runtime/data/cache/<video-stem>/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from topspin_review import config
from topspin_review.storage import runtime

_CACHE_VERSION = "v2"


def _signature(path: Path, max_frames: int) -> str:
    st = path.stat()
    raw = f"{_CACHE_VERSION}:{path.resolve()}:{st.st_mtime_ns}:{st.st_size}:{max_frames}:{config.max_seconds()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_dir(path: Path) -> Path:
    return runtime.CACHE_DIR / path.stem


def _jsonable(meta: dict) -> dict:
    out = {}
    for key, value in meta.items():
        try:
            json.dumps(value)
            out[key] = value
        except TypeError:
            out[key] = str(value)
    return out


def load(path: Path, max_frames: int) -> tuple[dict, list[Image.Image], list[float]] | None:
    """Return cached ``(meta, frames, timestamps)`` or ``None`` on miss/stale."""
    directory = _cache_dir(path)
    sig_file = directory / "signature.txt"
    if not sig_file.exists() or sig_file.read_text(encoding="utf-8") != _signature(path, max_frames):
        return None
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        timestamps = json.loads((directory / "timestamps.json").read_text(encoding="utf-8"))
        frames = [Image.open(p).convert("RGB") for p in sorted(directory.glob("frame_*.jpg"))]
    except Exception:
        return None
    if not frames:
        return None
    return meta, frames, timestamps


def save(path: Path, max_frames: int, meta: dict, frames: list[Image.Image], timestamps: list[float]) -> None:
    directory = _cache_dir(path)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("frame_*.jpg"):
        old.unlink()
    for i, frame in enumerate(frames):
        frame.save(directory / f"frame_{i:03d}.jpg", quality=90)
    (directory / "meta.json").write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "timestamps.json").write_text(json.dumps(timestamps), encoding="utf-8")
    (directory / "signature.txt").write_text(_signature(path, max_frames), encoding="utf-8")
