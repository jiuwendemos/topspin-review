"""Thread-safe progress reporting for long-running analysis runs.

The analysis updates a :class:`Progress` from the worker thread; a UI (e.g. the
Streamlit app) polls :meth:`Progress.snapshot` from the main thread and renders a
progress bar/stage. ``tick`` is a no-op when no reporter is attached.
"""

from __future__ import annotations

import threading


class Progress:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stage = "starting"
        self._pct = 0.0
        self._detail = ""
        self._done = False
        self._error: str | None = None
        self._stem: str | None = None

    def update(self, stage: str, pct: float, detail: str = "") -> None:
        with self._lock:
            self._stage = stage
            self._pct = float(pct)
            self._detail = detail

    def finish(self, stem: str | None = None) -> None:
        with self._lock:
            self._stage = "done"
            self._pct = 100.0
            self._done = True
            self._stem = stem

    def fail(self, message: str) -> None:
        with self._lock:
            self._done = True
            self._error = message

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "stage": self._stage,
                "pct": self._pct,
                "detail": self._detail,
                "done": self._done,
                "error": self._error,
                "stem": self._stem,
            }


def tick(progress: Progress | None, stage: str, pct: float, detail: str = "") -> None:
    """Update ``progress`` if present; never raises."""
    if progress is not None:
        try:
            progress.update(stage, pct, detail)
        except Exception:
            pass
