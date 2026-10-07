"""Score saved reports and check them against ``tests/eval/expected.json``.

    python scripts/evaluate_reports.py

Prints a quality table and fails (non-zero) if an expectation is unmet. With no
reports or no matching expectations it passes, so it is CI-safe.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topspin_review.domain import (  # noqa: E402
    evaluate,
    progress,
)
from topspin_review.domain import report as report_schema  # noqa: E402
from topspin_review.storage import store

DEFAULT_MIN_SCORE = int(os.getenv("REPORT_MIN_SCORE", "60"))


def _themes(report: dict) -> set[str]:
    return progress.classify(" ".join(report_schema.issue_texts(report)))


def main() -> int:
    reports = store.get_reports()
    expected_file = ROOT / "tests" / "eval" / "expected.json"
    expected = {}
    if expected_file.exists():
        expected = json.loads(expected_file.read_text(encoding="utf-8"))

    if not reports:
        print("No reports to evaluate (analyze a video first). Nothing to check.")
        return 0

    failures: list[str] = []
    print(f"{'video':32} {'score':>5}  themes")
    for report in reports:
        stem = Path(report.get("source", "?")).stem
        scored = evaluate.score(report)
        themes = _themes(report)
        print(f"{stem:32} {scored['score']:>5}  {', '.join(sorted(themes)) or '-'}")

        spec = expected.get(stem)
        spec = spec if isinstance(spec, dict) else {}

        # Every report must clear the quality floor (per-video override allowed).
        min_score = int(spec.get("min_score", DEFAULT_MIN_SCORE))
        if scored["score"] < min_score:
            failures.append(f"{stem}: score {scored['score']} < {min_score}")

        missing = set(spec.get("themes") or []) - themes
        if missing:
            failures.append(f"{stem}: missing themes {sorted(missing)}")
        if scored["hallucination_flags"]:
            failures.append(f"{stem}: hallucination flags {scored['hallucination_flags']}")

    if failures:
        print("\nFAIL:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nAll expectations met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
