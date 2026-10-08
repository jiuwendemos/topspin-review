"""Standalone analysis worker (run in its own process).

The Streamlit app launches this so the analysis runs with a clean ``__main__`` —
the backend runner spawns subprocesses, and Streamlit's script module caused a
duplicate-import/pickle error. Progress is streamed to a JSON file the UI polls.

    python -m topspin_review.interfaces.worker <video> --progress <path> [--agentic] [--box l,t,r,b]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from pathlib import Path

from topspin_review.analysis.strategies.progress import Progress
from topspin_review.bootstrap import run as run_async
from topspin_review.bootstrap import setup


def _parse_box(text: str | None) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    try:
        parts = [float(x) for x in text.split(",")]
    except ValueError:
        return None
    return tuple(parts) if len(parts) == 4 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="topspin-review-worker")
    parser.add_argument("video")
    parser.add_argument("--progress", required=True, help="path to write progress JSON to")
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--box", default=None, help="normalized l,t,r,b")
    args = parser.parse_args(argv)

    setup()
    prog = Progress()
    progress_path = Path(args.progress)
    stop = threading.Event()

    def reporter() -> None:
        while not stop.is_set():
            try:
                progress_path.write_text(json.dumps(prog.snapshot()), encoding="utf-8")
            except Exception:
                pass
            time.sleep(0.4)

    reporter_thread = threading.Thread(target=reporter, daemon=True)
    reporter_thread.start()

    from topspin_review.analysis import strategies

    strategy = strategies.resolve(args.agentic)

    try:
        run_async(strategy.analyze(strategies.Params(video_path=args.video, region_box=_parse_box(args.box), progress=prog)))
        prog.finish(Path(args.video).stem)
    except Exception as exc:  # noqa: BLE001
        prog.fail(f"{exc}\n\n{traceback.format_exc()}")
    finally:
        stop.set()
        try:
            progress_path.write_text(json.dumps(prog.snapshot()), encoding="utf-8")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
