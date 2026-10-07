"""Streamlit UI for Topspin Review — a video-first coaching journey.

Pages: Home -> Analyze -> Result -> Practice -> Progress -> You.

Run:
    streamlit run topspin_review/interfaces/web/ui.py
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import streamlit as st

from topspin_review import reporting as export
from topspin_review.analysis.progress import Progress
from topspin_review.bootstrap import run as run_async
from topspin_review.bootstrap import setup
from topspin_review.domain import compare as compare_mod
from topspin_review.domain import evaluate, render
from topspin_review.storage import runtime, store

setup()

SPORTS = ["table tennis", "tennis", "badminton", "squash", "padel"]
LEVELS = ["beginner", "intermediate", "advanced"]
HANDS = ["right", "left"]
PAGES = ["Home", "Analyze", "Practice", "Progress", "You"]


# --------------------------------------------------------------------------- #
# Background analysis + progress
# --------------------------------------------------------------------------- #
def _warm() -> None:
    try:
        from topspin_review.analysis import agentic, pipeline  # noqa: F401
    except Exception:
        pass


def _run_job(video: str, box, agentic: bool, prog: Progress, stem: str) -> None:
    prog.update("starting", 1)
    from topspin_review.analysis import pipeline

    run = pipeline.analyze
    if agentic:
        from topspin_review.analysis import agentic

        run = agentic.analyze
    try:
        run_async(run(video, region_box=box, progress=prog))
    except Exception as exc:  # noqa: BLE001
        prog.fail(str(exc))
    else:
        prog.finish(stem)


def _poll_job() -> None:
    job = st.session_state.get("job")
    if not job:
        return
    prog: Progress = job["prog"]
    thread: threading.Thread = job["thread"]
    snap = prog.snapshot()
    if thread.is_alive():
        pct = min(max(float(snap["pct"]), 0.0), 100.0)
        elapsed = time.monotonic() - float(job.get("start", time.monotonic()))
        st.progress(pct / 100.0, text=f"Analyzing… {snap['stage']} ({int(pct)}%) · {elapsed:.0f}s")
        time.sleep(0.5)
        st.rerun()
    st.session_state.pop("job", None)
    if snap.get("error"):
        st.session_state["job_error"] = snap["error"]
    else:
        st.session_state["latest_stem"] = snap.get("stem")
        st.session_state["page"] = "Result"
    st.rerun()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _video_info(path: str) -> dict:
    try:
        import imageio

        reader = imageio.get_reader(str(path), format="ffmpeg")
        meta = dict(reader.get_meta_data())
        reader.close()
        return {"fps": meta.get("fps"), "duration": meta.get("duration"), "size": meta.get("size")}
    except Exception:
        return {}


def _footage_check(path: str) -> None:
    info = _video_info(path)
    if not info:
        return
    size = info.get("size") or (0, 0)
    fps = info.get("fps") or 0
    duration = info.get("duration") or 0
    notes = []
    if duration and duration < 3:
        notes.append("very short (a few rallies are better)")
    if fps and fps < 25:
        notes.append(f"low frame rate ({fps:.0f} fps — 60+ shows footwork better)")
    if size and size[1] and size[1] < 480:
        notes.append("low resolution")
    if notes:
        st.warning("Footage tips: " + "; ".join(notes) + ". Side-on, full-body works best.")
    else:
        st.success("Looks good — analyzing this should give useful feedback.")


def _open(stem: str) -> None:
    st.session_state["latest_stem"] = stem
    st.session_state["seek"] = 0.0
    st.session_state["page"] = "Result"
    st.rerun()


def _go(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


# --------------------------------------------------------------------------- #
# Page: Home
# --------------------------------------------------------------------------- #
def _page_home() -> None:
    st.title("🏓 Topspin Review")
    st.caption("Upload a video of your game. Get one clear thing to work on, then practice it.")

    reports = store.get_reports()
    if not reports:
        st.info(
            "**Start here**\n\n1. Go to **Analyze**, choose a video of a few rallies.\n"
            "2. Get your feedback in about a minute.\n3. Follow your drills in **Practice**."
        )
        if st.button("Analyze a session", type="primary"):
            _go("Analyze")
        return

    latest = reports[-1]
    st.markdown("### Your focus right now")
    st.success(latest.get("focus") or "Keep up the consistency work.")
    if latest.get("progress"):
        st.caption(latest["progress"])

    col_a, col_b = st.columns([1, 1])
    if col_a.button("Analyze a new session", type="primary", use_container_width=True):
        _go("Analyze")
    if col_b.button("Open last report", use_container_width=True):
        _open(Path(latest.get("source", "")).stem)

    st.markdown("### Recent sessions")
    names = [render.video_name(r) for r in reports]
    for report in reversed(reports[-6:]):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{render.video_name(report)}**  \n{report.get('date', '')} — {report.get('focus', '')}")
        if c2.button("View", key=f"view_{render.video_name(report)}"):
            _open(Path(report.get("source", "")).stem)


# --------------------------------------------------------------------------- #
# Page: Analyze
# --------------------------------------------------------------------------- #
def _page_analyze() -> None:
    st.title("Analyze a session")
    st.caption("A few rallies from the side, full body, is ideal.")

    uploaded = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])
    use_sample = st.checkbox("…or use the built-in sample clip")

    target: Path | None = None
    if uploaded is not None:
        target = runtime.DATA_DIR / f"upload_{uploaded.name}"
    elif use_sample:
        target = runtime.DATA_DIR / "sample.mp4"
    if target is not None:
        _footage_check(str(target))

    with st.expander("Advanced"):
        st.checkbox("Let the AI decide what to review (slower)", key="_agentic")
        st.slider("How much of the video to study", 4, 30, value=12, key="_frames")
        st.number_input("Only study the first N seconds (0 = all)", 0, 3600, value=0, key="_seconds")
        if st.checkbox("Only look at me (ignore other people)", key="_use_box"):
            st.session_state["_box"] = (
                st.slider("left", 0.0, 1.0, 0.0, 0.05, key="_left"),
                st.slider("top", 0.0, 1.0, 0.0, 0.05, key="_top"),
                st.slider("right", 0.0, 1.0, 1.0, 0.05, key="_right"),
                st.slider("bottom", 0.0, 1.0, 1.0, 0.05, key="_bottom"),
            )
        else:
            st.session_state["_box"] = None

    os.environ["VISION_MAX_FRAMES"] = str(int(st.session_state.get("_frames", 12)))
    os.environ["VISION_MAX_SECONDS"] = str(int(st.session_state.get("_seconds", 0)))

    if st.button("Analyze", type="primary", disabled="job" in st.session_state):
        if target is None:
            st.warning("Choose a video (or tick the sample clip).")
        else:
            if uploaded is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(uploaded.getbuffer())
            if not target.exists():
                st.error(f"File not found: {target}")
            else:
                prog = Progress()
                worker = threading.Thread(
                    target=_run_job,
                    args=(
                        str(target),
                        st.session_state.get("_box"),
                        bool(st.session_state.get("_agentic")),
                        prog,
                        target.stem,
                    ),
                    daemon=True,
                )
                worker.start()
                st.session_state["job"] = {"thread": worker, "prog": prog, "start": time.monotonic()}
                st.rerun()


# --------------------------------------------------------------------------- #
# Page: Result
# --------------------------------------------------------------------------- #
def _moment_row(parent, idx: int, issue: dict) -> None:
    parent.markdown(f"- {issue.get('issue', '')}")
    times = issue.get("evidence_times") or []
    if times:
        cols = parent.columns(len(times))
        for c, t in zip(cols, times):
            if c.button(f"▶ {t}s", key=f"seek_{idx}_{t}"):
                st.session_state.seek = float(t)
                st.rerun()


def _page_result() -> None:
    reports = store.get_reports()
    latest = store.find_report(st.session_state.get("latest_stem", "")) if st.session_state.get("latest_stem") else None
    if latest is None:
        getter = getattr(store, "latest_report", None)
        latest = (getter() if callable(getter) else None) or (reports[-1] if reports else {})
    if not latest:
        st.info("No report yet.")
        if st.button("Analyze a session", type="primary"):
            _go("Analyze")
        return

    st.title(render.video_name(latest))
    st.caption(f"{latest.get('sport', '')} · {latest.get('date', '')}")

    src = latest.get("source")
    seek = float(st.session_state.get("seek", 0.0) or 0.0)
    if src and Path(src).exists():
        try:
            st.video(src, start_time=int(round(seek)))
        except TypeError:
            st.video(src)
        if seek:
            st.caption(f"Showing around {seek:.1f}s.")
    else:
        st.caption("(video not found)")

    if latest.get("focus"):
        st.success(f"### Focus next session\n{latest['focus']}")
    if latest.get("summary"):
        st.write(latest["summary"])

    strengths = latest.get("strengths") or []
    if strengths:
        st.markdown("### ✅ What's working")
        for item in strengths:
            st.markdown(f"- {item}")

    issues = latest.get("issues") or []
    if issues:
        st.markdown("### 🎯 What to fix")
        st.caption("Tap a time to jump the video to that moment.")
        for idx, item in enumerate(issues):
            if isinstance(item, dict):
                _moment_row(st, idx, item)
            else:
                st.markdown(f"- {item}")

    drills = latest.get("drills") or []
    if drills:
        st.markdown("### 🏋️ Drills to practice")
        for item in drills:
            st.markdown(f"- {item}")
        if st.button("Add these drills to Practice"):
            _go("Practice")

    metrics = latest.get("metrics") or {}
    artifacts = latest.get("artifacts") or {}
    if metrics:
        st.markdown("### 👀 Where you moved (and when)")
        st.caption(
            "How much the picture changed at each moment. A spike means big movement "
            "(a step or a swing); a flat stretch means little movement."
        )
        activity = metrics.get("activity") or []
        if activity:
            import pandas as pd

            df = pd.DataFrame(
                {"movement": [s["energy"] for s in activity]},
                index=[round(s["t"], 2) for s in activity],
            )
            df.index.name = "seconds"
            st.line_chart(df, y_label="movement")
            if metrics.get("peak_motion_time") is not None:
                st.caption(f"Biggest movement around t={metrics['peak_motion_time']}s.")
        mpath = artifacts.get("motion")
        if mpath and Path(mpath).exists():
            st.image(mpath, caption="Red = where you moved most during the clip.")
        with st.expander("Frames we looked at"):
            fpath = artifacts.get("frames")
            if fpath and Path(fpath).exists():
                st.image(fpath)
            ppath = artifacts.get("pose")
            if ppath and Path(ppath).exists():
                st.image(ppath, caption="Body pose overlay")
            if not fpath and not ppath:
                st.caption("Not available.")

    if latest.get("progress"):
        st.info(f"Since last time: {latest['progress']}")

    with st.expander("Keep in mind"):
        for item in latest.get("limitations") or []:
            st.markdown(f"- {item}")

    with st.expander("Details & download"):
        quality = evaluate.score(latest)
        st.write(f"Report confidence score: {quality['score']}/100")
        usage = latest.get("usage") or {}
        if usage:
            st.caption(f"Model usage: {usage.get('calls', 0)} calls · {usage.get('total_tokens', 0)} tokens")
        stem = Path(latest.get("source", "report")).stem
        st.download_button("Download report (Markdown)", export.to_markdown(latest), file_name=f"{stem}_report.md", mime="text/markdown")
        st.download_button("Download report (HTML)", export.to_html(latest), file_name=f"{stem}_report.html", mime="text/html")


# --------------------------------------------------------------------------- #
# Page: Practice
# --------------------------------------------------------------------------- #
def _page_practice() -> None:
    st.title("Practice")
    latest = store.latest_report() if hasattr(store, "latest_report") else (store.get_reports() or [{}])[-1]
    if not latest:
        st.info("Analyze a session first, then your drills will show up here.")
        if st.button("Analyze a session", type="primary"):
            _go("Analyze")
        return

    if latest.get("focus"):
        st.success(f"**Focus:** {latest['focus']}")
    drills = latest.get("drills") or []
    if not drills:
        st.info("No drills in the latest report.")
        return

    st.markdown("### This week's drills")
    done = 0
    for i, drill in enumerate(drills):
        key = f"drill_{i}"
        if st.checkbox(drill, key=key):
            done += 1
    st.progress(done / len(drills) if drills else 0.0, text=f"{done}/{len(drills)} drills checked off")


# --------------------------------------------------------------------------- #
# Page: Progress
# --------------------------------------------------------------------------- #
def _page_progress() -> None:
    st.title("Progress")
    reports = store.get_reports()
    if len(reports) < 1:
        st.info("Analyze a few sessions to see your progress.")
        return
    latest = reports[-1]
    trend = latest.get("progress_trend") or {}
    st.write(trend.get("text", "Not enough history yet."))

    counts = trend.get("issue_counts") or []
    if counts:
        st.markdown("### Issues per session (lower is better)")
        st.line_chart({"issues": counts})

    if len(reports) >= 2:
        with st.expander("Compare two sessions"):
            names = [render.video_name(r) for r in reports]
            left, right = st.columns(2)
            older = left.selectbox("Earlier", names, index=0)
            newer = right.selectbox("Later", names, index=len(names) - 1)
            if st.button("Compare") and older != newer:
                result = compare_mod.compare(reports[names.index(older)], reports[names.index(newer)])
                st.markdown(f"**What changed:** {result['text']}")
                for label, key in (("Improved", "improved"), ("Still an issue", "unchanged"), ("New/regressed", "regressed")):
                    items = result.get(key) or []
                    if items:
                        st.markdown(f"**{label}:** " + ", ".join(items))


# --------------------------------------------------------------------------- #
# Page: You
# --------------------------------------------------------------------------- #
def _page_you() -> None:
    st.title("You")
    profile = store.get_profile()
    st.markdown("### Your profile")
    sport = st.radio("Sport", SPORTS, index=SPORTS.index(profile["sport"]) if profile.get("sport") in SPORTS else 0, horizontal=True)
    level = st.radio("Level", LEVELS, index=LEVELS.index(profile["level"]) if profile.get("level") in LEVELS else 1, horizontal=True)
    hand = st.radio("Dominant hand", HANDS, index=HANDS.index(profile["dominant_hand"]) if profile.get("dominant_hand") in HANDS else 0, horizontal=True)
    goal = st.text_input("What you want to improve", value=profile.get("goal", "improve"))
    if st.button("Save profile", type="primary"):
        store.set_profile({"sport": sport, "level": level, "dominant_hand": hand, "goal": goal})
        st.success("Saved.")

    st.divider()
    if store.get_reports():
        if st.button("Clear all reports"):
            store.reset()
            st.session_state["latest_stem"] = None
            st.rerun()


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Topspin Review", page_icon="🏓", layout="centered")

if not st.session_state.get("_warm_started"):
    st.session_state["_warm_started"] = True
    threading.Thread(target=_warm, daemon=True).start()

st.session_state.setdefault("page", "Home")

with st.sidebar:
    st.markdown("## 🏓 Topspin Review")
    for name in PAGES:
        active = st.session_state.get("page") == name
        if st.button(name, key=f"nav_{name}", type="primary" if active else "secondary", use_container_width=True):
            _go(name)

if st.session_state.get("job_error"):
    st.error(f"Analysis failed: {st.session_state.pop('job_error')}")

_poll_job()

_PAGES = {
    "Home": _page_home,
    "Analyze": _page_analyze,
    "Result": _page_result,
    "Practice": _page_practice,
    "Progress": _page_progress,
    "You": _page_you,
}
_PAGES.get(st.session_state.get("page", "Home"), _page_home)()
