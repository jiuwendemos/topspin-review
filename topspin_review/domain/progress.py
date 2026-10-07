"""Aggregate saved reports into a computed progress summary and theme trend.

The model writes a one-line ``progress``; this module computes the real trend:
which coaching themes keep recurring across sessions, how the issue count moves,
and the focus history. It is deterministic, so the dashboard and the report can
both cite it.
"""

from __future__ import annotations

from collections import Counter, defaultdict

THEMES: dict[str, list[str]] = {
    "footwork": ["foot", "step", "split", "stance", "balance", "weight", "lunge", "shuffle", "step"],
    "stroke": ["swing", "stroke", "forehand", "backhand", "racket", "paddle", "follow-through", "follow through"],
    "preparation": ["preparation", "prep", "backswing", "ready", "set up"],
    "recovery": ["recovery", "recover", "reset", "return", "neutral"],
    "serve": ["serve", "toss", "service"],
}


def _issues(report: dict) -> list[str]:
    out = []
    for item in report.get("issues") or []:
        if isinstance(item, dict):
            out.append(str(item.get("issue", "")))
        else:
            out.append(str(item))
    return out


def classify(text: str) -> set[str]:
    lowered = (text or "").lower()
    return {theme for theme, words in THEMES.items() if any(w in lowered for w in words)}


def theme_counts(report: dict) -> Counter:
    counts: Counter = Counter()
    for issue in _issues(report):
        for theme in classify(issue):
            counts[theme] += 1
    return counts


def summarize(reports: list[dict]) -> dict:
    """Compute theme trend, issue counts and a progress sentence."""
    if not reports:
        return {}

    per_report = [theme_counts(r) for r in reports]
    trend: dict[str, list[int]] = defaultdict(list)
    for counts in per_report:
        for theme in THEMES:
            trend[theme].append(counts.get(theme, 0))

    latest = reports[-1]
    latest_themes = classify(" ".join(_issues(latest)))
    previous = reports[-2] if len(reports) > 1 else None
    previous_themes = classify(" ".join(_issues(previous))) if previous else set()
    strength_themes = classify(" ".join(latest.get("strengths") or []))

    issue_counts = [len(_issues(r)) for r in reports]
    repeated = sorted({t for t, series in trend.items() if sum(1 for v in series if v > 0) >= 2})

    # Did last session's focus get addressed?
    focus_themes = classify(previous.get("focus", "")) if previous else set()
    focus_status = "n/a"
    focus_achieved = False
    if focus_themes:
        still_open = focus_themes & latest_themes
        shown_strong = focus_themes & strength_themes
        if not still_open and (shown_strong or not latest_themes):
            focus_status = "achieved"
            focus_achieved = True
        elif still_open:
            focus_status = "still open"
        else:
            focus_status = "no longer flagged"

    if previous is None:
        text = f"first session recorded ({issue_counts[-1]} issue(s) noted)."
    else:
        fixed = sorted(previous_themes - latest_themes)
        still = sorted(previous_themes & latest_themes)
        parts = []
        if still:
            parts.append("still present: " + ", ".join(still))
        if fixed:
            parts.append("no longer flagged: " + ", ".join(fixed))
        parts.append(f"issue count {issue_counts[-2]} -> {issue_counts[-1]}")
        if focus_themes:
            parts.append(f"last focus ({', '.join(sorted(focus_themes))}) {focus_status}")
        text = "; ".join(parts) if parts else "no comparable change."

    return {
        "sessions": len(reports),
        "issue_counts": issue_counts,
        "theme_trend": {k: v for k, v in trend.items()},
        "repeated_themes": repeated,
        "latest_themes": sorted(latest_themes),
        "focus_history": [r.get("focus", "") for r in reports if r.get("focus")],
        "focus_status": focus_status,
        "focus_achieved": focus_achieved,
        "text": text,
    }
