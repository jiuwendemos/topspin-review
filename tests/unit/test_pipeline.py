"""Unit tests for the deterministic analysis/report pipeline (no model calls)."""

from __future__ import annotations

import asyncio

import numpy as np
from PIL import Image, ImageDraw

from topspin_review import observability as usage
from topspin_review import providers as backends
from topspin_review import reporting as export
from topspin_review.analysis import vision
from topspin_review.domain import compare as compare_mod
from topspin_review.domain import evaluate, progress, report
from topspin_review.perception import ball, cvutil, metrics


def moving_frames(n: int = 8, step: int = 8) -> tuple[list[Image.Image], list[float]]:
    frames: list[Image.Image] = []
    for i in range(n):
        img = Image.new("RGB", (160, 90), (30, 30, 35))
        d = ImageDraw.Draw(img)
        x = 15 + i * step
        d.rectangle([x, 62, x + 12, 74], fill=(245, 245, 245))
        frames.append(img)
    return frames, [round(i * 0.5, 2) for i in range(n)]


def stationary_frames(n: int = 8) -> tuple[list[Image.Image], list[float]]:
    frames = []
    for _ in range(n):
        img = Image.new("RGB", (160, 90), (30, 30, 35))
        ImageDraw.Draw(img).rectangle([60, 62, 72, 74], fill=(245, 245, 245))
        frames.append(img)
    return frames, [round(i * 0.5, 2) for i in range(n)]


def test_rgb_to_hsv_red():
    red = np.zeros((1, 1, 3), dtype=np.float32)
    red[..., 0] = 1.0
    h, s, v = cvutil.rgb_to_hsv(red)[0, 0]
    assert abs(h - 0.0) < 1e-6 or abs(h - 1.0) < 1e-6
    assert s > 0.99 and v > 0.99


def test_components_finds_blob():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True
    comps = cvutil.components(mask)
    assert len(comps) == 1 and comps[0]["area"] == 4


def test_largest_component():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:9, 6:9] = True
    big = cvutil.largest(mask)
    assert big is not None and int(big.sum()) == 9


def test_ball_tracks_moving_blob():
    frames, ts = moving_frames()
    steps = metrics.activity(frames, ts)
    info = ball.detect(frames, ts, steps)
    assert info["tracks"], "expected a tracked ball"
    assert len(info["tracks"][0]["points"]) >= 3


def test_metrics_keys_and_mechanics():
    frames, ts = moving_frames()
    steps = metrics.activity(frames, ts)
    measured = metrics.analyze(frames, ts, ball=ball.detect(frames, ts, steps))
    for key in ("activity", "track", "posture", "mechanics", "ball", "mean_energy", "peak_motion_time", "net_shift"):
        assert key in measured
    assert measured["mechanics"]["lower_lateral_range"] > 0


def test_motion_mechanics_move_vs_still():
    frames, ts = moving_frames()
    moving = metrics.analyze(frames, ts)
    still_frames, still_ts = stationary_frames()
    still = metrics.analyze(still_frames, still_ts)
    assert moving["mechanics"]["lower_lateral_range"] > still["mechanics"]["lower_lateral_range"]
    assert moving["mean_energy"] > still["mean_energy"]


def test_region_from_box():
    box = metrics.region_from_box((0.2, 0.1, 0.8, 0.9))
    assert box.shape == (180, 320) and box.any()


def test_report_schema_and_score():
    rep = report.normalize(
        {
            "date": "2026-01-01",
            "sport": "table tennis",
            "summary": "s",
            "strengths": ["a"],
            "issues": [{"issue": "b", "evidence_times": [1.0], "confidence": "high"}],
            "drills": ["d"],
            "focus": "f",
            "limitations": ["l"],
        }
    )
    assert report.validate(rep) == []
    scored = evaluate.score(rep)
    assert scored["evidence_coverage"] == 1.0
    assert scored["hallucination_flags"] == []
    assert scored["score"] >= 80


def test_evaluate_flags_hallucination():
    rep = report.normalize(
        {
            "date": "x",
            "sport": "t",
            "summary": "your ball speed was 40 km/h",
            "focus": "f",
            "issues": [{"issue": "bad spin", "evidence_times": [1.0], "confidence": "high"}],
        }
    )
    assert evaluate.score(rep)["hallucination_flags"]


def test_progress_focus_achieved():
    previous = {"focus": "improve footwork and split step", "issues": [{"issue": "footwork lag"}]}
    latest = {"focus": "add weight transfer", "issues": [{"issue": "late preparation"}], "strengths": ["good footwork"]}
    trend = progress.summarize([previous, latest])
    assert trend["focus_achieved"] is True
    assert trend["focus_status"] == "achieved"


def test_compare_detects_improvement():
    older = {"issues": [{"issue": "footwork"}], "metrics": {"mechanics": {"lower_lateral_range": 0.1}}}
    newer = {"issues": [{"issue": "late prep"}], "metrics": {"mechanics": {"lower_lateral_range": 0.2}}}
    diff = compare_mod.compare(older, newer)
    assert "footwork" in diff["improved"]
    assert diff["metric_deltas"]["lower_lateral_range"] == 0.1


def test_export_markdown_and_html():
    rep = report.normalize({"date": "d", "sport": "t", "summary": "s", "focus": "f", "issues": ["x"]})
    rep["source"] = "video11.mp4"  # normalize() intentionally drops non-schema keys
    assert "video11.mp4" in export.to_markdown(rep)
    assert export.to_html(rep).lstrip().startswith("<!doctype html")


def test_retrieval_context():
    from topspin_review.analysis import retrieval

    reports = [
        {"source": "a.mp4", "issues": [{"issue": "poor footwork and split step"}], "focus": "footwork"},
        {"source": "b.mp4", "issues": [{"issue": "late backswing"}], "focus": "preparation"},
    ]
    ctx = retrieval.context_text(reports, "footwork split step", k=2)
    assert "a.mp4" in ctx and "b.mp4" not in ctx


def test_extract_json_variants():
    assert vision.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert vision.extract_json("garbage") == {}


def test_usage_extraction():
    class Meta:
        input_tokens = 10
        output_tokens = 5
        total_tokens = 15

    class Msg:
        usage_metadata = Meta()

    extracted = usage.extract_usage(Msg())
    assert extracted["prompt_tokens"] == 10
    assert extracted["completion_tokens"] == 5
    assert extracted["total_tokens"] == 15

    class DictMsg:
        usage = {"prompt_tokens": 2, "completion_tokens": 3}

    assert usage.extract_usage(DictMsg())["total_tokens"] == 5
    assert usage.summarize([{"total_tokens": 5, "seconds": 1.0}])["total_tokens"] == 5


def test_mock_backend_offline():
    text = asyncio.run(backends.MockVisionBackend().complete([{"role": "user", "content": "attentive_windows"}]))
    assert "attentive_windows" in text


def test_service_describe():
    from topspin_review.interfaces import service

    desc = service.describe()
    assert desc["name"] == "analyze_sport_video"
    assert "video_path" in desc["input_params"]["properties"]
    assert service.parse_region_box([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)
    assert service.parse_region_box("bad") is None


def test_mcp_tool_defined():
    import pytest

    try:
        from topspin_review.interfaces.mcp import tools as mcp_tools
    except Exception as exc:  # openjiuwen not importable
        pytest.skip(f"openjiuwen unavailable: {exc}")
    assert mcp_tools.ALL_TOOLS, "no tools registered"
    names = {getattr(t, "card", None) and t.card.name for t in mcp_tools.ALL_TOOLS}
    assert "analyze_sport_video" in names


def test_api_routes_registered():
    import pytest

    pytest.importorskip("fastapi")
    from topspin_review.interfaces import api

    paths = {getattr(route, "path", None) for route in api.app.routes}
    assert {"/analyze", "/reports"} <= paths


def test_mcp_server_module_imports():
    from topspin_review.interfaces.mcp import server

    assert callable(server.main)


def test_pose_graceful_without_mediapipe():
    from topspin_review.perception import pose

    assert pose.summarize(None) == {}
    assert pose.overlay([], None) == []
