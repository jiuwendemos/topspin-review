"""Adaptive, motion-weighted frame sampling with an on-disk cache.

Even sampling wastes frames on dead time and skips short bursts of action. This
probes the clip cheaply, then spends the frame budget where the image is actually
changing (weighted by motion, with a baseline everywhere), so footwork and swing
moments are more likely to be captured.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

from topspin_review import config
from topspin_review.storage import runtime

_CACHE_VERSION = "v2"


def _scaled_reader(path: str, width: int = 160, height: int = 90):
    """Reader that decodes frames downscaled (cheap probing for long clips)."""
    try:
        return imageio.get_reader(path, format="ffmpeg", output_params=["-vf", f"scale={width}:{height}"])
    except Exception:
        return imageio.get_reader(path, format="ffmpeg")


def _probe_indices(total: int, budget: int) -> list[int]:
    if total <= budget:
        return list(range(total))
    return sorted({int(round(i * (total - 1) / (budget - 1))) for i in range(budget)})


def _allocate(count: int, energies: list[float]) -> list[int]:
    """Split ``count`` samples across intervals, weighted by motion energy."""
    k = len(energies)
    if k == 0 or count <= 0:
        return [0] * k
    peak = max(energies) or 1.0
    weights = [0.3 + (e / peak) for e in energies]  # 0.3 baseline so no interval starves
    total = sum(weights)
    raw = [count * w / total for w in weights]
    counts = [int(r) for r in raw]
    remainder = count - sum(counts)
    order = sorted(range(k), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % k]] += 1
    return counts


def _select_indices(probe: list[int], energies: list[float], max_frames: int, total: int) -> list[int]:
    if total <= max_frames:
        return list(range(total))

    # one motion value per interval between consecutive probes
    interval_energy = [max(0.0, e) for e in energies]
    counts = _allocate(max_frames, interval_energy)

    chosen: set[int] = set()
    for i, c in enumerate(counts):
        start, end = probe[i], probe[i + 1]
        if c <= 0 or end <= start:
            continue
        for j in range(c):
            idx = int(round(start + (j + 1) * (end - start) / (c + 1)))
            chosen.add(min(idx, total - 1))
    chosen.add(0)
    chosen.add(total - 1)
    return sorted(chosen)


def _signature(path: Path, max_frames: int) -> str:
    st = path.stat()
    raw = f"{_CACHE_VERSION}:{path.resolve()}:{st.st_mtime_ns}:{st.st_size}:{max_frames}:{config.max_seconds()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_dir(path: Path) -> Path:
    return runtime.CACHE_DIR / path.stem


def _load_cache(path: Path, max_frames: int) -> tuple[dict, list[Image.Image], list[float]] | None:
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


def _save_cache(path: Path, max_frames: int, meta: dict, frames: list[Image.Image], timestamps: list[float]) -> None:
    directory = _cache_dir(path)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("frame_*.jpg"):
        old.unlink()
    for i, frame in enumerate(frames):
        frame.save(directory / f"frame_{i:03d}.jpg", quality=90)
    (directory / "meta.json").write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "timestamps.json").write_text(json.dumps(timestamps), encoding="utf-8")
    (directory / "signature.txt").write_text(_signature(path, max_frames), encoding="utf-8")


def _jsonable(meta: dict) -> dict:
    out = {}
    for key, value in meta.items():
        try:
            json.dumps(value)
            out[key] = value
        except TypeError:
            out[key] = str(value)
    return out


def sample_window(
    video_path: str | Path, start: float, end: float, count: int = 4
) -> tuple[list[Image.Image], list[float]]:
    """Sample ``count`` frames evenly inside ``[start, end]`` seconds."""
    path = Path(video_path)
    if not path.exists() or count <= 0:
        return [], []

    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        try:
            meta = dict(reader.get_meta_data())
        except Exception:
            meta = {}
        fps = float(meta.get("fps") or 30)
        duration = meta.get("duration")
        if duration:
            total = max(1, int(float(duration) * fps))
        else:
            try:
                total = int(reader.count_frames())
            except Exception:
                total = 0

        lo = max(0, int(round(float(start) * fps)))
        hi = int(round(float(end) * fps))
        if hi <= lo:
            hi = lo + count
        if total:
            lo = min(lo, total - 1)
            hi = min(max(hi, lo + 1), total)

        if count == 1:
            indices = [lo]
        else:
            indices = sorted({int(round(lo + i * (hi - lo) / (count - 1))) for i in range(count)})

        frames: list[Image.Image] = []
        times: list[float] = []
        for idx in indices:
            try:
                frames.append(Image.fromarray(reader.get_data(idx)).convert("RGB"))
                times.append(idx / fps)
            except Exception:
                continue
        return frames, times
    finally:
        reader.close()


def sample_frames(
    video_path: str | Path, max_frames: int = 12, use_cache: bool = True
) -> tuple[dict, list[Image.Image], list[float]]:
    """Return ``(metadata, frames, timestamps)`` sampled where motion happens."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    if use_cache:
        cached = _load_cache(path, max_frames)
        if cached is not None:
            return cached

    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        try:
            meta = dict(reader.get_meta_data())
        except Exception:
            meta = {}

        fps = float(meta.get("fps") or 30)
        duration = meta.get("duration")
        if duration:
            total = max(1, int(float(duration) * fps))
        else:
            try:
                total = int(reader.count_frames())
            except Exception:
                total = 0

        cap_seconds = config.max_seconds()
        if cap_seconds > 0:
            total = min(total, max(1, int(cap_seconds * fps)))

        if total and total > 0:
            probes = _probe_indices(total, min(config.probe_budget(), max(2, total)))
            probe_reader = _scaled_reader(str(path), 160, 90)
            try:
                probe_gray = []
                for idx in probes:
                    frame = Image.fromarray(probe_reader.get_data(idx)).convert("L")
                    if frame.size != (160, 90):
                        frame = frame.resize((160, 90))
                    probe_gray.append(np.asarray(frame, dtype=np.float32))
            except Exception:
                probe_gray = [
                    np.asarray(Image.fromarray(reader.get_data(idx)).convert("L").resize((160, 90)), dtype=np.float32)
                    for idx in probes
                ]
            finally:
                try:
                    probe_reader.close()
                except Exception:
                    pass
            energies = [float(np.abs(probe_gray[i] - probe_gray[i - 1]).mean()) for i in range(1, len(probe_gray))]
            indices = _select_indices(probes, energies, max_frames, total)
            frames = [Image.fromarray(reader.get_data(idx)).convert("RGB") for idx in indices]
            timestamps = [idx / fps for idx in indices]
        else:
            frames = []
            timestamps = []
            for i, arr in enumerate(reader):
                if i % 30 == 0:
                    frames.append(Image.fromarray(arr).convert("RGB"))
                    timestamps.append(i / fps)
                if len(frames) >= max_frames:
                    break
    finally:
        reader.close()

    if use_cache and frames:
        try:
            _save_cache(path, max_frames, meta, frames, timestamps)
        except Exception:
            pass
    return meta, frames, timestamps
