"""Score a report for quality: evidence coverage, confidence, hallucination risk.

Deterministic and offline. Used by the benchmark and available to callers that
want a quick confidence check on a generated report.
"""

from __future__ import annotations

from topspin_review.domain import report as report_schema

_FORBIDDEN = ("spin", "rpm", "km/h", "kph", "mph", "ball speed", "degrees", "exact angle")


def score(report: dict) -> dict:
    issues = [i for i in report_schema.issues(report) if isinstance(i, dict)]
    with_evidence = [i for i in issues if i.get("evidence_times")]
    coverage = round(len(with_evidence) / len(issues), 2) if issues else 0.0

    confidences = [i.get("confidence", "low") for i in issues]
    dist = {level: confidences.count(level) for level in ("high", "medium", "low")}

    text = " ".join(
        [str(report.get("summary", ""))]
        + [str(i.get("issue", "")) for i in issues]
        + [str(s) for s in (report.get("strengths") or [])]
    ).lower()
    hallucination_flags = [word for word in _FORBIDDEN if word in text]

    checks = {
        "has_summary": bool(report.get("summary")),
        "has_focus": bool(report.get("focus")),
        "has_issues": bool(issues),
        "evidence_coverage": coverage,
        "confidence": dist,
        "has_limitations": bool(report.get("limitations")),
        "hallucination_flags": hallucination_flags,
    }

    points = 0
    points += 30 * coverage
    points += 20 if checks["has_summary"] else 0
    points += 15 if checks["has_focus"] else 0
    points += 10 if checks["has_limitations"] else 0
    points += 10 if issues else 0
    points += 15 if not hallucination_flags else 0
    checks["score"] = int(round(points))
    return checks
