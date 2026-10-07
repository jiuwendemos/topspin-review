"""Local state: player profile and one report file per analyzed video.

Each analyzed video is saved as ``runtime/data/<video-stem>_report.json`` (for
example ``video11.mp4`` -> ``video11_report.json``), so a report is always
identified by the video file name it came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topspin_review.storage import json_store, runtime

DATA_DIR = runtime.DATA_DIR
PROFILE_PATH = DATA_DIR / "profile.json"
REPORT_SUFFIX = "_report.json"

_current_video: str | None = None


def set_current_video(video_path: str | Path) -> str:
    """Remember which video the next saved report belongs to."""
    global _current_video
    _current_video = Path(video_path).stem
    return _current_video


def report_path(video_path: str | Path) -> Path:
    """Report file path for ``video_path``: ``<stem>_report.json``."""
    return DATA_DIR / f"{Path(video_path).stem}{REPORT_SUFFIX}"


def _current_report_path() -> Path | None:
    if not _current_video:
        return None
    return DATA_DIR / f"{_current_video}{REPORT_SUFFIX}"


def _read(path: Path, default: Any) -> Any:
    return json_store.read_json(path, default)


def _write(path: Path, data: Any) -> None:
    json_store.write_json(path, data)


def get_profile() -> dict[str, Any]:
    return _read(PROFILE_PATH, {})


def set_profile(profile: dict[str, Any]) -> dict[str, Any]:
    _write(PROFILE_PATH, profile)
    return profile


def get_reports() -> list[dict[str, Any]]:
    """All saved reports, oldest first — one file per analyzed video."""
    reports: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob(f"*{REPORT_SUFFIX}")):
        report = _read(path, {})
        if isinstance(report, dict) and report:
            reports.append(report)
    return reports


def add_report(report: dict[str, Any]) -> dict[str, Any]:
    """Save the report for the current video to ``<stem>_report.json``."""
    path = _current_report_path()
    if path is None:
        raise RuntimeError("no current video set; call set_current_video() first")
    _write(path, report)
    return report


def recent_reports(n: int = 5) -> list[dict[str, Any]]:
    return get_reports()[-n:]


def patch_last_report(patch: dict[str, Any]) -> dict[str, Any] | None:
    """Merge ``patch`` into the current video's report file."""
    path = _current_report_path()
    if path is None or not path.exists():
        return None
    report = _read(path, {})
    if not isinstance(report, dict):
        return None
    report.update(patch)
    _write(path, report)
    return report


def reset() -> None:
    for path in DATA_DIR.glob(f"*{REPORT_SUFFIX}"):
        path.unlink()
