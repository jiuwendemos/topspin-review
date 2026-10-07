"""Streamlit UI for Topspin Review — a video-first coaching journey.

Pages: Sessions (history) -> Analyze -> Result -> Practice -> Progress -> You.

Run:
    streamlit run topspin_review/interfaces/web/ui.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from topspin_review import reporting as export
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
#
# The analysis runs in its own process (see topspin_review.interfaces.worker).
# In-process runs collided with Streamlit's module handling: openjiuwen's runner
# spawns subprocesses, and Streamlit's script module caused the spawned child to
# re-import openjiuwen as a second copy, breaking pickling of message objects.
# The worker streams progress to a JSON file that we poll here.
# --------------------------------------------------------------------------- #
def _start_worker(target: Path, *, agentic: bool, box) -> None:
    progress_file = runtime.DATA_DIR / f"_progress_{target.stem}.json"
    try:
        progress_file.unlink()
    except OSError:
        pass
    cmd = [sys.executable, "-m", "topspin_review.interfaces.worker", str(target), "--progress", str(progress_file)]
    if agentic:
        cmd.append("--agentic")
    if box:
        cmd += ["--box", ",".join(str(x) for x in box)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(runtime.PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(cmd, cwd=str(runtime.PROJECT_ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st.session_state["job"] = {
        "proc": proc,
        "progress": str(progress_file),
        "stem": target.stem,
        "start": time.monotonic(),
    }


def _poll_job() -> None:
    job = st.session_state.get("job")
    if not job:
        return
    snap: dict = {}
    try:
        snap = json.loads(Path(job["progress"]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        snap = {}
    proc: subprocess.Popen = job["proc"]
    if proc.poll() is None:
        pct = min(max(float(snap.get("pct") or 0.0), 0.0), 100.0)
        elapsed = time.monotonic() - float(job.get("start", time.monotonic()))
        st.progress(pct / 100.0, text=f"Analyzing… {snap.get('stage', 'starting')} ({int(pct)}%) · {elapsed:.0f}s")
        time.sleep(0.5)
        st.rerun()
    st.session_state.pop("job", None)
    if snap.get("error"):
        st.session_state["job_error"] = snap["error"]
        st.rerun()
    st.session_state["latest_stem"] = snap.get("stem") or job.get("stem")
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
    from topspin_review.perception import sampling

    if sampling.is_image(path):
        st.info("Single photo: you'll get feedback on your **posture**, not movement or footwork.")
        return
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
    """Runs at the top of every page: show errors, poll the job."""
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
</style>
"""


def _ask_panel() -> None:
    """Floating 'ask about this report' panel — always visible, all pages, no jump."""
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
    st.caption("Every video or photo you've analyzed. Pick one to open, or analyze a new one.")

    reports = store.get_reports()
    if st.button("Analyze a new video or photo", type="primary"):
        _go("Analyze")
    if not reports:
        st.info("No sessions yet. Click **Analyze a new video or photo** to get started.")
        return

    latest = reports[-1]
    st.markdown("### Current focus")
    st.success(latest.get("focus") or "Keep up the consistency work.")

    st.markdown("### History")
    for report in reversed(reports):
        stem = Path(report.get("source", "")).stem
        if st.session_state.get("confirm_delete") == stem:
            c1, c2, c3 = st.columns([6, 1, 1])
            c1.warning(f"Delete **{render.video_name(report)}**?")
            if c2.button("Delete", key=f"confirm_{stem}", type="primary"):
                store.delete_report(stem)
                if st.session_state.get("latest_stem") == stem:
                    st.session_state["latest_stem"] = None
                st.session_state.pop(f"qa_{stem}", None)
                st.session_state["confirm_delete"] = None
                st.rerun()
            if c3.button("Cancel", key=f"cancel_{stem}"):
                st.session_state["confirm_delete"] = None
                st.rerun()
            continue
        c1, c2, c3 = st.columns([8, 1, 1])
        c1.markdown(f"**{render.video_name(report)}**  \n{report.get('date', '')} — {report.get('focus', '')}")
        if c2.button("Open", key=f"open_{stem}"):
            _open(stem)
        if c3.button("", icon=":material/close:", key=f"del_{stem}", type="tertiary", help="Delete this session"):
            st.session_state["confirm_delete"] = stem
            st.rerun()


# --------------------------------------------------------------------------- #
# Page: Analyze
# --------------------------------------------------------------------------- #
def _page_analyze() -> None:
    _prelude()
    st.title("Analyze a video or photo")
    st.caption("A few rallies from the side (full body) gives the most; a single photo gives posture feedback.")

    uploaded = st.file_uploader(
        "Choose a video or a single photo", type=["mp4", "mov", "avi", "mkv", "png", "jpg", "jpeg", "bmp", "webp"]
    )
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
            st.warning("Choose a video or photo (or tick the sample clip).")
        else:
            if uploaded is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(uploaded.getbuffer())
            if not target.exists():
                st.error(f"File not found: {target}")
            else:
                _start_worker(
                    target,
                    agentic=bool(st.session_state.get("_agentic")),
                    box=st.session_state.get("_box"),
                )
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
        if st.button("Analyze a video or photo", type="primary"):
            _go("Analyze")
        return

    st.subheader(render.video_name(latest))
    st.caption(f"{latest.get('sport', '')} · {latest.get('date', '')}")

    tab_report, tab_movement, tab_details, tab_technical = st.tabs(["Report", "Movement", "Details", "Technical"])

    with tab_report:
        if latest.get("focus"):
            st.success(f"**Focus next session:** {latest['focus']}")
        if latest.get("summary"):
            st.write(latest["summary"])

        src = latest.get("source")
        seek = float(st.session_state.get("seek", 0.0) or 0.0)
        if src and Path(src).exists():
            if src.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                st.image(src)
            else:
                try:
                    st.video(src, start_time=int(round(seek)))
                except TypeError:
                    st.video(src)
        else:
            st.caption("(media not found)")

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
        if not metrics or not (metrics.get("activity")):
            st.info("No movement data (single-image inputs have none).")
        else:
            import pandas as pd

            df = pd.DataFrame(
                {"movement": [s["energy"] for s in metrics["activity"]]},
                index=[round(s["t"], 2) for s in metrics["activity"]],
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

    with tab_details:
        with st.expander("Keep in mind", expanded=True):
            for item in latest.get("limitations") or []:
                st.markdown(f"- {item}")
        quality = evaluate.score(latest)
        st.write(f"Report confidence score: {quality['score']}/100")
        stem = Path(latest.get("source", "report")).stem
        st.download_button(
            "Download report (Markdown)", export.to_markdown(latest), file_name=f"{stem}_report.md", mime="text/markdown"
        )
        st.download_button(
            "Download report (HTML)", export.to_html(latest), file_name=f"{stem}_report.html", mime="text/html"
        )

    with tab_technical:
        st.caption("How this report was produced — model calls and timings.")
        usage = latest.get("usage") or {}
        if usage:
            text_u = usage.get("text") or {}
            vision_u = usage.get("vision") or {}
            model_secs = float(text_u.get("seconds", 0) or 0) + float(vision_u.get("seconds", 0) or 0)
            st.caption(
                f"Model usage: {usage.get('calls', 0)} calls · {usage.get('total_tokens', 0)} tokens · "
                f"{model_secs:.0f}s inside model calls "
                f"(text {float(text_u.get('seconds', 0) or 0):.0f}s / vision {float(vision_u.get('seconds', 0) or 0):.0f}s)"
            )
            models = usage.get("models") or {}
            if models:
                st.caption(
                    f"Text model: {models.get('text', '?')} · Vision model: {models.get('vision', '?')} · "
                    f"Provider: {models.get('provider', '?')} ({models.get('api_base', '')})"
                )
            vc = usage.get("vision_calls") or []
            tc = usage.get("text_calls") or []
            ctx = sum(int(c.get("prompt_tokens", 0) or 0) for c in vc + tc)
            gen = sum(int(c.get("completion_tokens", 0) or 0) for c in vc + tc)
            cached = sum(int(c.get("cached_tokens", 0) or 0) for c in vc + tc)
            st.caption(
                f"Tokens billed: {ctx + gen} total = {ctx} context/input + {gen} output"
                + (f" · {cached} served from cache" if cached else "")
                + f", across {len(vc) + len(tc)} calls. The whole context is re-sent on every call, "
                "so each call's input already includes all prior messages."
            )
        timings = latest.get("timings") or []
        if timings:
            import pandas as pd

            text_calls = usage.get("text_calls") or []
            vision_calls = usage.get("vision_calls") or []
            agent_stages = ("agent writing report", "agent: analyzing")
            vision_stage = {
                "overview": "vision: overview",
                "detail": "vision: detail",
                "reviewing image": "vision: reviewing image",
            }

            children: dict[str, list[dict]] = {}
            for call in vision_calls:
                pin = int(call.get("prompt_tokens", 0) or 0)
                pout = int(call.get("completion_tokens", 0) or 0)
                cached = int(call.get("cached_tokens", 0) or 0)
                cache_note = f", {cached} cached" if cached else ""
                stage = vision_stage.get(call.get("label") or "", "__agent__")
                children.setdefault(stage, []).append(
                    {
                        "what": f"    ↳ vision model call ({pin} context{cache_note} + {pout} out)",
                        "seconds": float(call.get("seconds", 0) or 0),
                    }
                )
            for stage in agent_stages:
                if "__agent__" in children:
                    children.setdefault(stage, []).extend(children.pop("__agent__"))

            rows: list[dict] = []
            for entry in timings:
                stage = entry.get("stage", "")
                secs = float(entry.get("seconds", 0) or 0)
                rows.append({"what": stage, "seconds": f"{secs:.2f}"})
                nested = list(children.get(stage, []))
                if stage in agent_stages:
                    for i, call in enumerate(text_calls, start=1):
                        pin = int(call.get("prompt_tokens", 0) or 0)
                        pout = int(call.get("completion_tokens", 0) or 0)
                        cached = int(call.get("cached_tokens", 0) or 0)
                        cache_note = f", {cached} cached" if cached else ""
                        nested.append(
                            {
                                "what": f"    ↳ model call {i} ({pin} context{cache_note} + {pout} out)",
                                "seconds": float(call.get("seconds", 0) or 0),
                            }
                        )
                    model_secs = sum(r["seconds"] for r in nested)
                    nested.append(
                        {
                            "what": "    ↳ agent overhead (tool calls, prompt building)",
                            "seconds": max(0.0, secs - model_secs),
                        }
                    )
                rows.extend({"what": r["what"], "seconds": f"{r['seconds']:.2f}"} for r in nested)
            st.markdown("**Where the time went**")
            st.caption("Every step in order; each model call is nested under the step that made it.")
            st.table(pd.DataFrame(rows))

        calls_path = (latest.get("artifacts") or {}).get("calls")
        if calls_path and Path(calls_path).exists():
            try:
                calls = json.loads(Path(calls_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                calls = []
            if calls:
                st.markdown("**Model calls in full**")
                st.caption(
                    "The exact context sent to the model (images shown as [image]) and the raw output. "
                    "This is the full, untruncated text."
                )
                st.download_button(
                    "Download all calls (JSON)",
                    json.dumps(calls, ensure_ascii=False, indent=2),
                    file_name="model_calls.json",
                    mime="application/json",
                )
                for i, call in enumerate(calls, start=1):
                    model = call.get("model") or ""
                    title = f"Call {i} — {call.get('label', '')}" + (f" · {model}" if model else "")
                    with st.expander(title):
                        tools = call.get("tools") or []
                        if tools:
                            st.caption("Tools offered: " + ", ".join(tools))
                        for message in call.get("input") or []:
                            st.markdown(f"**{message.get('role', '')}**")
                            st.code(message.get("content", ""), language="text")
                        st.markdown("**output**")
                        st.code(call.get("output", ""), language="text")

        tools_path = (latest.get("artifacts") or {}).get("tools")
        if tools_path and Path(tools_path).exists():
            try:
                called = json.loads(Path(tools_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                called = []
            st.markdown("**Tools called**")
            if not called:
                st.caption("No tools were called this run.")
            else:
                st.caption(", ".join(f"{t.get('name', '')} ({t.get('seconds', 0)}s)" for t in called))
                for i, tool in enumerate(called, start=1):
                    with st.expander(f"{i}. {tool.get('name', '')} — {tool.get('seconds', 0)}s"):
                        if tool.get("arguments"):
                            st.markdown("**arguments**")
                            st.code(str(tool.get("arguments")), language="text")
                        if tool.get("error"):
                            st.error(str(tool.get("error")))
                        st.markdown("**result**")
                        st.code(str(tool.get("result", "")), language="text")

    # Floating ask panel: only on the Result page, but visible across its tabs.
    _ask_panel()


# --------------------------------------------------------------------------- #
# Page: Practice
# --------------------------------------------------------------------------- #
def _page_practice() -> None:
    _prelude()
    st.title("Practice")
    latest = _latest_report()
    if not latest:
        st.info("Analyze a video or photo first, then your drills will show up here.")
        if st.button("Analyze a video or photo", type="primary"):
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
    sport = st.radio(
        "Sport", SPORTS, index=SPORTS.index(profile["sport"]) if profile.get("sport") in SPORTS else 0, horizontal=True
    )
    level = st.radio(
        "Level", LEVELS, index=LEVELS.index(profile["level"]) if profile.get("level") in LEVELS else 1, horizontal=True
    )
    hand = st.radio(
        "Dominant hand", HANDS, index=HANDS.index(profile["dominant_hand"]) if profile.get("dominant_hand") in HANDS else 0, horizontal=True
    )
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
