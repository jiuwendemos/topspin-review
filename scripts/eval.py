"""Deterministic self-check for the analysis pipeline (no model calls).

    python scripts/eval.py

Checks sampling, motion metrics, the motion map, JSON extraction and report
schema handling against the synthetic sample clip. Exit code is non-zero if any
check fails, so it can be used as a regression gate in CI.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topspin_review.analysis import vision  # noqa: E402
from topspin_review.domain import compare as compare_mod  # noqa: E402
from topspin_review.domain import evaluate, progress, report  # noqa: E402
from topspin_review.perception import ball, cvutil, motion, sampling  # noqa: E402
from topspin_review.providers import backends  # noqa: E402
from topspin_review.reporting import export  # noqa: E402

RESULTS: list[tuple[bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((bool(condition), f"{name}{(' - ' + detail) if detail else ''}"))


def main() -> int:
    sample = ROOT / "runtime" / "data" / "sample.mp4"
    if not sample.exists():
        import make_sample

        make_sample.main()

    meta, frames, timestamps = sampling.sample_frames(sample, max_frames=10, use_cache=False)
    check("sample_frames returns frames", len(frames) > 0, f"n={len(frames)}")
    check("timestamps align and are sorted", len(timestamps) == len(frames) and timestamps == sorted(timestamps))
    check("meta has fps", float(meta.get("fps") or 0) > 0, f"fps={meta.get('fps')}")

    steps = motion.activity(frames, timestamps)
    ball_info = ball.detect(frames, timestamps, steps)
    metrics = motion.analyze(frames, timestamps, ball=ball_info)
    for key in ("activity", "track", "posture", "mechanics", "ball", "mean_energy", "peak_motion_time", "net_shift"):
        check(f"metrics has {key}", key in metrics)
    check("activity has one step per gap", len(metrics["activity"]) == max(0, len(frames) - 1))
    check("mechanics derived", bool(metrics.get("mechanics")))
    check("ball summary present", bool(ball_info.get("summary")))

    mm = motion.motion_map(frames)
    check("motion_map built", mm is not None and mm.size[0] > 0)

    dx, dy = motion.phase_shift(frames[0], frames[0])
    check("phase_shift of identical frame is zero", abs(dx) < 0.5 and abs(dy) < 0.5, f"({dx},{dy})")

    parsed = vision.extract_json('```json\n{"overall": "ok", "attentive_windows": []}\n```')
    check("extract_json handles fenced JSON", parsed.get("overall") == "ok")
    check("extract_json handles junk", vision.extract_json("no json here") == {})

    coerced = report.normalize({"issues": ["old string issue"]})
    check("normalize coerces string issue", coerced["issues"][0]["issue"] == "old string issue")

    good = report.normalize(
        {
            "date": "2026-01-01",
            "sport": "table tennis",
            "summary": "s",
            "strengths": ["a"],
            "issues": [{"issue": "b", "evidence_times": [1.2, 2.0], "confidence": "HIGH"}],
            "drills": ["d"],
            "focus": "f",
            "limitations": ["l"],
        }
    )
    check("normalize lower-cases confidence", good["issues"][0]["confidence"] == "high")
    check("validate accepts good report", report.validate(good) == [], str(report.validate(good)))
    check("validate flags missing evidence", bool(report.validate(coerced)))

    trend = progress.summarize(
        [
            {**good, "issues": [{"issue": "footwork: no split step", "evidence_times": [1.0], "confidence": "high"}]},
            {
                **good,
                "issues": [{"issue": "footwork: still reaching", "evidence_times": [2.0], "confidence": "medium"}],
            },
        ]
    )
    check("progress detects repeated theme", "footwork" in (trend.get("repeated_themes") or []))
    check("progress text computed", bool(trend.get("text")))

    md = export.to_markdown({**good, "source": "video11.mp4"})
    check("export markdown has title", "# Topspin Review report" in md and "video11.mp4" in md)
    check("export html is html", export.to_html(good).lstrip().startswith("<!doctype html"))

    mock_text = asyncio.run(backends.MockVisionBackend().complete([{"role": "user", "content": "attentive_windows"}]))
    check("mock backend returns JSON", "attentive_windows" in mock_text and vision.extract_json(mock_text))

    import numpy as _np

    red = _np.zeros((1, 1, 3), dtype=_np.float32)
    red[..., 0] = 1.0
    _, s, v = cvutil.rgb_to_hsv(red)[0, 0]
    check("rgb_to_hsv red", s > 0.99 and v > 0.99)
    comps = cvutil.components(_np.pad(_np.ones((2, 2), dtype=bool), 4))
    check("components finds one blob", len(comps) == 1 and comps[0]["area"] == 4)

    check("evaluate scores good report high", evaluate.score(good)["score"] >= 80, str(evaluate.score(good)["score"]))
    bad = report.normalize(
        {"date": "d", "sport": "s", "summary": "ball speed 40 km/h", "focus": "f", "issues": ["spin"]}
    )
    check("evaluate flags hallucination", bool(evaluate.score(bad)["hallucination_flags"]))

    older = {**good, "metrics": {"mechanics": {"lower_lateral_range": 0.1}}, "issues": [{"issue": "footwork"}]}
    newer = {
        **good,
        "metrics": {"mechanics": {"lower_lateral_range": 0.2}},
        "issues": [{"issue": "late prep"}],
        "strengths": ["good footwork"],
    }
    diff = compare_mod.compare(older, newer)
    check("compare detects improvement", "footwork" in diff["improved"], str(diff["improved"]))
    check("compare metric delta", diff["metric_deltas"].get("lower_lateral_range") == 0.1)

    box = motion.region_from_box((0.2, 0.1, 0.8, 0.9))
    check("region_from_box shape/content", box.shape == (180, 320) and box.any())

    check("backend usage summary", backends.summarize_usage([{"total_tokens": 5, "seconds": 1.0}])["total_tokens"] == 5)

    failed = [msg for ok, msg in RESULTS if not ok]
    for ok, msg in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
