"""Compare two reports for the same player and describe what changed."""

from __future__ import annotations

from topspin_review.domain import progress, report as report_schema


def _mechanics(report: dict) -> dict:
    return (report.get("metrics") or {}).get("mechanics") or {}


def _mean_energy(report: dict) -> float:
    return float((report.get("metrics") or {}).get("mean_energy") or 0.0)


def compare(older: dict, newer: dict) -> dict:
    """Return improved/regressed themes and metric deltas (older -> newer)."""
    themes_old = progress.classify(" ".join(report_schema.issue_texts(older)))
    themes_new = progress.classify(" ".join(report_schema.issue_texts(newer)))

    improved = sorted(themes_old - themes_new)
    regressed = sorted(themes_new - themes_old)
    unchanged = sorted(themes_old & themes_new)

    mech_old, mech_new = _mechanics(older), _mechanics(newer)
    metric_deltas = {
        key: round(float(mech_new.get(key, 0)) - float(mech_old.get(key, 0)), 3)
        for key in sorted(set(mech_old) | set(mech_new))
    }

    name_old = older.get("source") or "older"
    name_new = newer.get("source") or "newer"
    from pathlib import Path

    parts = [f"{Path(name_old).name} -> {Path(name_new).name}"]
    if improved:
        parts.append("improved: " + ", ".join(improved))
    if regressed:
        parts.append("regressed: " + ", ".join(regressed))
    if unchanged:
        parts.append("still open: " + ", ".join(unchanged))
    energy_delta = round(_mean_energy(newer) - _mean_energy(older), 2)
    parts.append(f"mean energy {energy_delta:+}")

    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "metric_deltas": metric_deltas,
        "energy_delta": energy_delta,
        "text": "; ".join(parts),
    }
