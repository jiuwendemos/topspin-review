"""Command-line interface for Topspin Review.

Usage:
    python -m topspin_review.interfaces.cli profile
    python -m topspin_review.interfaces.cli analyze [video_path]
    python -m topspin_review.interfaces.cli history
"""

from __future__ import annotations

import argparse
import sys

from topspin_review import config
from topspin_review import reporting as export
from topspin_review.bootstrap import run as run_async
from topspin_review.bootstrap import setup
from topspin_review.domain import compare as compare_mod
from topspin_review.domain import render
from topspin_review.storage import runtime, store

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SPORTS = ["table tennis", "tennis", "badminton", "squash", "padel"]
LEVELS = ["beginner", "intermediate", "advanced"]
HANDS = ["right", "left"]


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def _choose(prompt: str, options: list[str], default: str) -> str:
    print(f"{prompt}:")
    for i, option in enumerate(options, 1):
        print(f"  {i}) {option}")
    answer = input(f"Choose 1-{len(options)} (Enter = {default}): ").strip()
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    return answer if answer in options else default


def cmd_profile() -> int:
    p = store.get_profile()
    print("Set your profile (Enter keeps the current value).\n")
    sport = _choose("Sport", SPORTS, p.get("sport", "table tennis"))
    level = _choose("Level", LEVELS, p.get("level", "intermediate"))
    hand = _choose("Dominant hand", HANDS, p.get("dominant_hand", "right"))
    goal = _ask("Goal", p.get("goal", "improve"))
    store.set_profile({"sport": sport, "level": level, "dominant_hand": hand, "goal": goal})
    print("\nProfile saved to runtime/data/profile.json.")
    return 0


def _print_list(label: str, items: list) -> None:
    if not items:
        return
    print(f"\n  {label}:")
    for item in items:
        text = render.issue_line(item) if label == "Issues" else str(item)
        print(f"    - {text}")


def _print_report(report: dict) -> None:
    if not report:
        return
    print(f"\nReport — {render.video_name(report)} · {report.get('sport', '')} ({report.get('date', '')})")
    if report.get("summary"):
        print(f"\n  {report['summary']}")
    _print_list("Strengths", report.get("strengths") or [])
    _print_list("Issues", report.get("issues") or [])
    _print_list("Drills", report.get("drills") or [])
    if report.get("focus"):
        print(f"\n  Focus next session: {report['focus']}")
    if report.get("progress"):
        print(f"\n  Progress vs last time: {report['progress']}")
    _print_list("Limitations", report.get("limitations") or [])


def _parse_box(text: str | None) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    try:
        parts = [float(x) for x in text.split(",")]
    except ValueError as exc:
        raise SystemExit("--box needs 4 comma-separated numbers: left,top,right,bottom") from exc
    if len(parts) != 4:
        raise SystemExit("--box needs 4 comma-separated numbers: left,top,right,bottom")
    return (parts[0], parts[1], parts[2], parts[3])


def cmd_analyze(path: str, box: str | None = None, agentic: bool = False) -> int:
    from topspin_review.analysis import pipeline

    run = pipeline.analyze
    use_agentic = agentic or config.agentic_mode()
    if use_agentic:
        from topspin_review.analysis import agentic

        run = agentic.analyze

    try:
        outcome = run_async(run(path, region_box=_parse_box(box)))
    except (config.ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2
    result = outcome.get("result") if isinstance(outcome, dict) else None
    output = result.get("output") if isinstance(result, dict) else str(result)
    print("\n[coach] " + (output or "").strip())
    print(f"\n(frames analyzed: {outcome.get('frames')})")
    if outcome.get("report_path"):
        print(f"report saved to: {outcome['report_path']}")
    usage = (outcome.get("report") or {}).get("usage") or {}
    if usage:
        print(
            f"(usage: {usage.get('calls', 0)} calls · "
            f"{usage.get('total_tokens', 0)} tokens · "
            f"vision {usage.get('vision', {}).get('total_tokens', 0)} / "
            f"text {usage.get('text', {}).get('total_tokens', 0)})"
        )
    _print_report(outcome.get("report") or {})
    return 0


def cmd_history() -> int:
    reports = store.get_reports()
    if not reports:
        print("No reports yet. Run `analyze`.")
        return 0
    print(f"Saved reports ({len(reports)}):\n")
    for report in reports[-5:]:
        print(
            f"  {render.video_name(report)} · {report.get('date', '?')} · "
            f"{report.get('sport', '')} · {report.get('focus', '')}"
        )
    print()
    _print_report(reports[-1])
    return 0


def cmd_export(video: str | None = None) -> int:
    report = store.find_report(video) if video else (store.get_reports() or [None])[-1]
    if not report:
        print(f"No report for {video}." if video else "No reports yet. Run `analyze` first.")
        return 1 if video else 0
    paths = export.write(report)
    print(f"Exported {render.video_name(report)}:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


def cmd_compare(video_a: str, video_b: str) -> int:
    older, newer = store.find_report(video_a), store.find_report(video_b)
    if not older or not newer:
        missing = video_a if not older else video_b
        print(f"No report for {missing}. Analyze it first.")
        return 1
    result = compare_mod.compare(older, newer)
    print(f"\n{result['text']}\n")
    if result["metric_deltas"]:
        print("Metric deltas (newer - older):")
        for key, value in result["metric_deltas"].items():
            print(f"  {key}: {value:+}")
    return 0


def cmd_reset() -> int:
    store.reset()
    print("Cleared saved reports.")
    return 0


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(prog="topspin-review", description="Topspin Review CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("profile", help="set your profile")
    p_analyze = sub.add_parser("analyze", help="analyze a session video")
    p_analyze.add_argument("path", nargs="?", default=str(runtime.DATA_DIR / "sample.mp4"), help="video file path")
    p_analyze.add_argument(
        "--box",
        default=None,
        help="normalized player region 'left,top,right,bottom' (0..1) to ignore other people",
    )
    p_analyze.add_argument(
        "--agentic",
        action="store_true",
        help="let the model drive analysis by inspecting windows itself (slower)",
    )
    sub.add_parser("history", help="show recent reports")
    p_export = sub.add_parser("export", help="export the latest (or a specific) report")
    p_export.add_argument("video", nargs="?", default=None, help="video file name")
    p_compare = sub.add_parser("compare", help="compare two analyzed videos")
    p_compare.add_argument("video_a", help="older video file name")
    p_compare.add_argument("video_b", help="newer video file name")
    sub.add_parser("reset", help="clear saved reports")

    args = parser.parse_args(argv)

    if args.command == "profile":
        return cmd_profile()
    if args.command == "analyze":
        return cmd_analyze(args.path, getattr(args, "box", None), getattr(args, "agentic", False))
    if args.command == "history":
        return cmd_history()
    if args.command == "export":
        return cmd_export(args.video)
    if args.command == "compare":
        return cmd_compare(args.video_a, args.video_b)
    if args.command == "reset":
        return cmd_reset()
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
