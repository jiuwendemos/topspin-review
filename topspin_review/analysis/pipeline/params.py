"""Parameters for running an analysis strategy.

Callers pass a single ``Params`` to ``Strategy.analyze``. It carries the request
(``video_path``/``region_box``/``progress``) and, during the run, is populated with
the sampled frames, measurements and session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from topspin_review.analysis.progress import Progress


@dataclass
class Params:
    """One analysis request + the run state filled in by ``Strategy.analyze``."""

    video_path: str
    region_box: tuple[float, float, float, float] | None = None
    progress: Progress | None = None

    # populated during the run
    profile: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    measured: dict = field(default_factory=dict)
    is_still: bool = False
    session: Any = None
    artifact_paths: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
