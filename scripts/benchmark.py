"""Deterministic benchmark over synthetic scenarios (no model calls).

    python scripts/benchmark.py

Each scenario synthesizes frames with a known motion pattern and checks the
pipeline reports it correctly, plus a report-quality rubric. Exit code is
non-zero if any scenario fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topspin_review.domain import evaluate, progress, report
from topspin_review.perception import (
    ball,  # noqa: E402
    motion,
)

RESULTS: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((bool(ok), f"{name}{(' - ' + detail) if detail else ''}"))


def _frames(positions: list[tuple[int, int]], size=(160, 90)) -> tuple[list[Image.Image], list[float]]:
    frames = []
    for x, y in positions:
        img = Image.new("RGB", size, (30, 30, 35))
        ImageDraw.Draw(img).rectangle([x, y, x + 12, y + 12], fill=(245, 245, 245))
        frames.append(img)
    return frames, [round(i * 0.5, 2) for i in range(len(positions))]


def _good_report() -> dict:
    return report.normalize(
        {
            "date": "2026-01-01",
            "sport": "table tennis",
            "summary": "summary",
            "strengths": ["low ready stance"],
            "issues": [{"issue": "no split step", "evidence_times": [1.0, 2.0], "confidence": "high"}],
            "drills": ["split-step drill"],
            "focus": "split step earlier",
            "limitations": ["sparse frames"],
        }
    )


def main() -> int:
    moving_f, moving_t = _frames([(10 + 8 * i, 62) for i in range(8)])
    still_f, still_t = _frames([(60, 62) for _ in range(8)])
    ball_f, ball_t = _frames([(10 + 8 * i, 30 + (i % 2) * 10) for i in range(8)])

    moving = motion.analyze(moving_f, moving_t)
    still = motion.analyze(still_f, still_t)
    check("stationary clip has lower energy", still["mean_energy"] < moving["mean_energy"])
    check(
        "stationary lower-body barely moves",
        still["mechanics"]["lower_lateral_range"] < 0.05,
        f"{still['mechanics']['lower_lateral_range']}",
    )
    check(
        "lateral mover detected",
        moving["mechanics"]["lower_lateral_range"] > 0.08,
        f"{moving['mechanics']['lower_lateral_range']}",
    )

    steps = motion.activity(ball_f, ball_t)
    info = ball.detect(ball_f, ball_t, steps)
    check("ball track detected", bool(info["tracks"]) and len(info["tracks"][0]["points"]) >= 3)

    good = _good_report()
    check("good report validates", report.validate(good) == [])
    check("good report scores high", evaluate.score(good)["score"] >= 80, str(evaluate.score(good)["score"]))

    trend = progress.summarize(
        [
            {"focus": "improve footwork", "issues": [{"issue": "footwork"}]},
            {**good, "strengths": ["good footwork"], "issues": [{"issue": "late prep"}]},
        ]
    )
    check("focus tracked as achieved", trend["focus_achieved"] is True)

    failed = [msg for ok, msg in RESULTS if not ok]
    for ok, msg in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
