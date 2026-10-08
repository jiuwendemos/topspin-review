"""Stage helper: measure the clip (sample frames + compute metrics).

Turns a video (or a single still) into the frames and numbers the rest of the
pipeline reasons over: motion energy/shift, subject track, coarse mechanics,
optional pose and heuristic ball signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from topspin_review import config
from topspin_review.analysis.progress import Progress, tick
from topspin_review.analysis.stages.measure import ball, sampling
from topspin_review.analysis.video import metrics


@dataclass
class Measurement:
    """The result of measuring a clip."""

    meta: dict
    frames: list
    timestamps: list
    measured: dict
    is_still: bool = False


def measure(
    video_path: str,
    region_box: tuple[float, float, float, float] | None = None,
    progress: Progress | None = None,
) -> Measurement:
    """Sample the clip and measure it."""
    tick(progress, "sampling frames", 8)
    is_still = sampling.is_image(video_path)
    if is_still:
        meta, frames, timestamps = sampling.load_image(video_path)
    else:
        meta, frames, timestamps = sampling.sample_frames(video_path, config.max_frames(), config.use_cache())
    if not frames:
        raise ValueError("No frames could be extracted from the input.")

    tick(progress, "measuring motion", 22)
    steps = metrics.activity(frames, timestamps)
    ball_info = ball.detect(frames, timestamps, steps)
    measured = metrics.analyze(frames, timestamps, ball=ball_info, region_box=region_box)

    return Measurement(
        meta=meta,
        frames=frames,
        timestamps=timestamps,
        measured=measured,
        is_still=is_still,
    )


__all__ = ["Measurement", "measure"]
