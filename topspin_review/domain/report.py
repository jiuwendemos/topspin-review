"""Report schema: normalize whatever the model returns into a stable shape.

Every claim should be backed by evidence, so an "issue" is a dict with the
timestamps that support it and a confidence. ``normalize`` tolerates the model
returning bare strings (older behaviour) and coerces everything to the schema.
"""

from __future__ import annotations

from typing import Any

CONFIDENCE = {"high", "medium", "low"}


_CONF_ORDER = {"low": 0, "medium": 1, "high": 2}


def calibrate(report: dict, quality: dict | None) -> dict:
    """Cap issue confidence according to footage quality (measurement, not vibes)."""
    score = int((quality or {}).get("score", 100))
    cap = "low" if score < 40 else "medium" if score < 70 else "high"
    for issue in report.get("issues") or []:
        if isinstance(issue, dict):
            conf = issue.get("confidence", "low")
            if _CONF_ORDER.get(conf, 0) > _CONF_ORDER[cap]:
                issue["confidence"] = cap
    return report


def issues(report: dict) -> list[Any]:
    """Raw issue entries (dicts or legacy strings) from a report."""
    return list(report.get("issues") or [])


def issue_texts(report: dict) -> list[str]:
    """Issue texts only, tolerating both dict and legacy string issues."""
    out: list[str] = []
    for item in issues(report):
        out.append(str(item.get("issue", "")) if isinstance(item, dict) else str(item))
    return out


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _as_issue(value: Any) -> dict | None:
    if isinstance(value, str):
        text = value.strip()
        return {"issue": text, "evidence_times": [], "confidence": "low"} if text else None
    if not isinstance(value, dict):
        return None
    times: list[float] = []
    for item in value.get("evidence_times") or []:
        try:
            times.append(round(float(item), 2))
        except (TypeError, ValueError):
            continue
    confidence = str(value.get("confidence", "low")).lower()
    if confidence not in CONFIDENCE:
        confidence = "low"
    return {
        "issue": str(value.get("issue", "")).strip(),
        "evidence_times": times,
        "confidence": confidence,
    }


def normalize(raw: Any) -> dict:
    """Coerce a model report into the schema (unknown keys are dropped)."""
    raw = raw if isinstance(raw, dict) else {}
    issues = [issue for issue in (_as_issue(i) for i in (raw.get("issues") or [])) if issue]
    return {
        "date": str(raw.get("date", "")),
        "sport": str(raw.get("sport", "")),
        "summary": str(raw.get("summary", "")),
        "strengths": _as_str_list(raw.get("strengths")),
        "issues": issues,
        "drills": _as_str_list(raw.get("drills")),
        "focus": str(raw.get("focus", "")),
        "progress": str(raw.get("progress", "")),
        "limitations": _as_str_list(raw.get("limitations")),
    }


def validate(report: dict) -> list[str]:
    """Return a list of schema problems (empty means OK)."""
    problems: list[str] = []
    for field in ("date", "sport", "summary", "focus"):
        if not report.get(field):
            problems.append(f"missing {field}")
    if not report.get("issues"):
        problems.append("no issues reported")
    for issue in report.get("issues") or []:
        if not issue.get("evidence_times"):
            problems.append(f"issue without evidence_times: {issue.get('issue', '')[:40]}")
        if issue.get("confidence") not in CONFIDENCE:
            problems.append(f"bad confidence: {issue.get('confidence')}")
    return problems
