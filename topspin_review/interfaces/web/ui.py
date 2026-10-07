"""Streamlit UI for Topspin Review — a video-first coaching journey.

Pages: Sessions (history) -> Analyze -> Result -> Practice -> Progress -> You.

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
        st.rerun()
    st.session_state["latest_stem"] = snap.get("stem")
    st.switch_page(PAGE_OBJS["Result"])


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


PAGE_OBJS: dict = {}


def _go(page: str) -> None:
    st.switch_page(PAGE_OBJS[page])


def _open(stem: str) -> None:
    st.session_state["latest_stem"] = stem
    st.session_state["seek"] = 0.0
    st.switch_page(PAGE_OBJS["Result"])


def _prelude() -> None:
    """Runs at the top of every page: warm imports, show errors, poll the job."""
    if not st.session_state.get("_warm_started"):
        st.session_state["_warm_started"] = True
        threading.Thread(target=_warm, daemon=True).start()
    if st.session_state.get("job_error"):
        st.error(f"Analysis failed: {st.session_state.pop('job_error')}")
    _poll_job()


_ASK_CSS = """
<style>
.st-key-askbox {
    position: fixed;
    right: 1rem;
    bottom: 1rem;
    width: 22rem;
    max-width: calc(100vw - 2rem);
    z-index: 9999;
    background: var(--background-color, #ffffff);
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 0.6rem;
    padding: 0.6rem 0.7rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18);
}
.st-key-askbox [data-testid="stForm"] { border: none; padding: 0; }
.st-key-askbox [data-testid="stVerticalBlock"] { gap: 0.4rem; }
</style>
"""


def _ask_panel() -> None:
    """Floating 'ask about this report' panel — always visible, all pages, no jump.

    Fixed to the bottom-right (so it survives scrolling and any selected tab/page)
    and built from a plain form (no autofocus, so opening a session doesn't jump).
    """
    latest = _latest_report()
    if not latest:
        return
    stem = Path(latest.get("source", "")).stem
    key = f"qa_{stem}"
    history = st.session_state.setdefault(key, [])

    st.markdown(_ASK_CSS, unsafe_allow_html=True)
    with st.container(key="askbox"):
        st.markdown("**Ask about this report**")
        for question, answer in history[-2:]:
            st.caption(f"You: {question}")
            st.markdown(f"**Coach:** {answer}")
        with st.form(key=f"ask_{stem}", clear_on_submit=True):
            question = st.text_input(
                "Your question", placeholder="e.g. How do I fix my split-step?", label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Ask", use_container_width=True)
        if submitted and question:
            from topspin_review.interfaces import service

            with st.spinner("Thinking…"):
                answer = service.ask_sync(stem, question)
            history.append((question, answer))
            st.rerun()


# --------------------------------------------------------------------------- #
# Page: Sessions (history)
# --------------------------------------------------------------------------- #
def _page_sessions() -> None:
    _prelude()
    st.title("Sessions")
    st.caption("Every video you've analyzed. Pick one to open, or analyze a new one.")

    reports = store.get_reports()
    if st.button("Analyze a new video", type="primary"):
        _go("Analyze")
    if not reports:
        st.info("No sessions yet. Click **Analyze a new video** to get started.")
        return

    latest = reports[-1]
    st.markdown("### Current focus")
    st.success(latest.get("focus") or "Keep up the consistency work.")

    st.markdown("### History")
    for report in reversed(reports):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{render.video_name(report)}**  \n{report.get('date', '')} — {report.get('focus', '')}")
        if c2.button("Open", key=f"open_{render.video_name(report)}_{report.get('date', '')}"):
            _open(Path(report.get("source", "")).stem)


# --------------------------------------------------------------------------- #
# Page: Analyze
# --------------------------------------------------------------------------- #
def _page_analyze() -> None:
    _prelude()
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
def _latest_report() -> dict:
    reports = store.get_reports()
    latest = store.find_report(st.session_state.get("latest_stem", "")) if st.session_state.get("latest_stem") else None
    if latest is None:
        getter = getattr(store, "latest_report", None)
        latest = (getter() if callable(getter) else None) or (reports[-1] if reports else {})
    return latest


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
    _prelude()
    latest = _latest_report()
    if not latest:
        st.info("No report yet.")
        if st.button("Analyze a session", type="primary"):
            _go("Analyze")
        return

    st.subheader(render.video_name(latest))
    st.caption(f"{latest.get('sport', '')} · {latest.get('date', '')}")

    tab_report, tab_movement, tab_details = st.tabs(["Report", "Movement", "Details"])

    with tab_report:
        if latest.get("focus"):
            st.success(f"**Focus next session:** {latest['focus']}")
        if latest.get("summary"):
            st.write(latest["summary"])

        src = latest.get("source")
        seek = float(st.session_state.get("seek", 0.0) or 0.0)
        if src and Path(src).exists():
            try:
                st.video(src, start_time=int(round(seek)))
            except TypeError:
                st.video(src)
        else:
            st.caption("(video not found)")

        strengths = latest.get("strengths") or []
        if strengths:
            st.markdown("#### What's working")
            for item in strengths:
                st.markdown(f"- {item}")

        issues = latest.get("issues") or []
        if issues:
            st.markdown("#### What to fix")
            st.caption("Tap a time to jump the video to that moment.")
            for idx, item in enumerate(issues):
                if isinstance(item, dict):
                    _moment_row(st, idx, item)
                else:
                    st.markdown(f"- {item}")

        clips = latest.get("clips") or {}
        if clips:
            with st.expander("Watch the key moments"):
                for t, path in clips.items():
                    st.markdown(f"**Around {t}s**")
                    if Path(path).exists():
                        st.video(path)

        drills = latest.get("drills") or []
        if drills:
            st.markdown("#### Drills to practice")
            for item in drills:
                st.markdown(f"- {item}")
            if st.button("Add these drills to Practice"):
                _go("Practice")

        if latest.get("progress"):
            st.info(f"Since last time: {latest['progress']}")

    with tab_movement:
        metrics = latest.get("metrics") or {}
        artifacts = latest.get("artifacts") or {}
        if not metrics:
            st.info("No movement data.")
        else:
            activity = metrics.get("activity") or []
            if activity:
                import pandas as pd

                df = pd.DataFrame(
                    {"movement": [s["energy"] for s in activity]},
                    index=[round(s["t"], 2) for s in activity],
                )
                df.index.name = "seconds"
                st.line_chart(df, y_label="movement")
                st.caption(
                    "How much the picture changed at each moment: a spike is a step or a swing, "
                    "a flat stretch is little movement."
                )
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

    with tab_details:
        with st.expander("Keep in mind", expanded=True):
            for item in latest.get("limitations") or []:
                st.markdown(f"- {item}")
        quality = evaluate.score(latest)
        st.write(f"Report confidence score: {quality['score']}/100")
        usage = latest.get("usage") or {}
        if usage:
            st.caption(f"Model usage: {usage.get('calls', 0)} calls · {usage.get('total_tokens', 0)} tokens")
        stem = Path(latest.get("source", "report")).stem
        st.download_button(
            "Download report (Markdown)", export.to_markdown(latest), file_name=f"{stem}_report.md", mime="text/markdown"
        )
        st.download_button(
            "Download report (HTML)", export.to_html(latest), file_name=f"{stem}_report.html", mime="text/html"
        )




# --------------------------------------------------------------------------- #
# Page: Practice
# --------------------------------------------------------------------------- #
def _page_practice() -> None:
    _prelude()
    st.title("Practice")
    latest = _latest_report()
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
        if st.checkbox(drill, key=f"drill_{i}"):
            done += 1
    st.progress(done / len(drills) if drills else 0.0, text=f"{done}/{len(drills)} drills checked off")


# --------------------------------------------------------------------------- #
# Page: Progress
# --------------------------------------------------------------------------- #
def _page_progress() -> None:
    _prelude()
    st.title("Progress")
    reports = store.get_reports()
    if not reports:
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
    _prelude()
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

PAGE_OBJS.update(
    {
        "Sessions": st.Page(_page_sessions, title="Sessions", icon="📋", url_path="sessions", default=True),
        "Analyze": st.Page(_page_analyze, title="Analyze", icon="🎬", url_path="analyze"),
        "Practice": st.Page(_page_practice, title="Practice", icon="🏋️", url_path="practice"),
        "Progress": st.Page(_page_progress, title="Progress", icon="📈", url_path="progress"),
        "You": st.Page(_page_you, title="You", icon="👤", url_path="you"),
        "Result": st.Page(_page_result, title="Result", icon="🏓", url_path="result"),
    }
)

_nav = st.navigation([PAGE_OBJS[k] for k in ("Sessions", "Analyze", "Practice", "Progress", "You", "Result")])
_nav.run()

# App-level, outside the pages: the floating ask panel shows on every page/tab.
_ask_panel()
