"""Application settings, read from the environment (.env).

Only behaviour that shapes *what the app does* lives here (sampling, feature
toggles, budgets). Anything about reaching model/embedding endpoints — provider,
keys, URLs, model names, timeouts — lives in :mod:`topspin_review.backend.settings`.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# --- vision sampling ------------------------------------------------------- #
def max_frames() -> int:
    return int(os.getenv("VISION_MAX_FRAMES", "12"))


def max_width() -> int:
    return int(os.getenv("VISION_MAX_WIDTH", "768"))


def zoom_frames() -> int:
    """Frames sampled inside the windows the first pass asks to inspect."""
    return int(os.getenv("VISION_ZOOM_FRAMES", "6"))


def max_windows() -> int:
    return int(os.getenv("VISION_MAX_WINDOWS", "3"))


def zoom_fps() -> float:
    """Target frames-per-second when sampling a zoom burst inside a window."""
    return float(os.getenv("VISION_ZOOM_FPS", "8"))


def use_cache() -> bool:
    return _bool("VISION_CACHE", "true")


def max_seconds() -> float:
    """Cap analysis to the first N seconds of a clip (0 = whole clip)."""
    return float(os.getenv("VISION_MAX_SECONDS", "0"))


def probe_budget() -> int:
    """How many cheap frames to read when locating motion-heavy windows."""
    return int(os.getenv("VISION_PROBE_BUDGET", "64"))


# --- analysis behaviour ---------------------------------------------------- #
def agentic_mode() -> bool:
    """Run the model-driven analysis strategy instead of the fixed one."""
    return _bool("AGENTIC_MODE", "false")


def verify_reports() -> bool:
    """Run a verification pass that drops unsupported issues."""
    return _bool("VERIFY_REPORTS", "true")


def clips_enabled() -> bool:
    """Cut short highlight clips around flagged moments."""
    return _bool("HIGHLIGHT_CLIPS", "true")


def retrieval_enabled() -> bool:
    """Add lexically-retrieved past sessions to the report context."""
    return _bool("RETRIEVAL", "true")
