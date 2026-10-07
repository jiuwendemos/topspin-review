"""Single implementation of report text rendering shared by CLI, export and UI."""

from __future__ import annotations

from pathlib import Path


def video_name(report: dict) -> str:
    """The analyzed video's file name, used to identify a report."""
    source = report.get("source")
    return Path(source).name if source else "(unknown video)"


def issue_line(issue: object) -> str:
    """One-line rendering of an issue, with evidence times and confidence."""
    if isinstance(issue, dict):
        times = ", ".join(str(t) for t in (issue.get("evidence_times") or []))
        confidence = issue.get("confidence") or "?"
        suffix = f"  [t={times}; confidence {confidence}]" if times else f"  [confidence {confidence}]"
        return f"{issue.get('issue', issue)}{suffix}"
    return str(issue)
