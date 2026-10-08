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
from topspin_review.interfaces.web.timeline import build_timeline, summary_stats
from topspin_review.storage import runtime, store

setup()

SPORTS = ["table tennis", "tennis", "badminton", "squash", "padel"]
LEVELS = ["beginner", "intermediate", "advanced"]
HANDS = ["right", "left"]

_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #0f172a;
    --muted: #64748b;
    --line: rgba(15, 23, 42, 0.08);
    --card: #ffffff;
    --accent: #4f46e5;
    --accent-soft: rgba(79, 70, 229, 0.10);
    --radius: 18px;
    --shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 12px 32px -16px rgba(16, 24, 40, 0.22);
}

html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(1200px 480px at 100% -10%, rgba(79, 70, 229, 0.10), transparent 60%),
        radial-gradient(900px 420px at -10% 10%, rgba(14, 165, 233, 0.08), transparent 55%),
        #f6f7fb;
}

.block-container { max-width: 1080px; padding-top: 2.4rem; padding-bottom: 4rem; }

h1, h2, h3, h4 { letter-spacing: -0.02em; color: var(--ink); font-weight: 700; }
h1 { font-weight: 800; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebarNav"] a { border-radius: 12px; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 18px 20px;
    box-shadow: var(--shadow);
    transition: transform .15s ease, box-shadow .15s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); }
[data-testid="stMetricLabel"] p {
    text-transform: uppercase;
    letter-spacing: .07em;
    font-size: .70rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] { font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; }

/* Section hero */
.tech-hero { margin: 0 0 1rem 0; }
.tech-hero h2 { margin: 0; font-size: 1.6rem; }
.tech-hero p { margin: .2rem 0 0; color: var(--muted); font-size: .92rem; }

.section-label {
    display: inline-flex; align-items: center; gap: .5rem;
    text-transform: uppercase; letter-spacing: .09em; font-size: .70rem;
    font-weight: 700; color: var(--accent);
    background: var(--accent-soft);
    padding: .32rem .6rem; border-radius: 999px;
    margin: 0 0 .5rem 0;
}

/* Expanders as cards */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03) !important;
    overflow: hidden;
    background: var(--card);
    margin-bottom: .5rem;
}
[data-testid="stExpander"] summary { font-weight: 600; padding: .75rem 1rem; }
[data-testid="stExpander"] summary:hover { background: #f8fafc; }

/* Tabs: transparent, underline for the active one */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.4rem;
    background: transparent;
    border-bottom: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: .5rem .15rem;
    background: transparent !important;
    border-radius: 0;
    font-weight: 600;
    color: var(--muted);
    border-bottom: 2px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover { background: transparent !important; color: var(--ink); }
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab"] > div { background: transparent !important; }

/* Tables */
[data-testid="stTable"] table, [data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden; border: 1px solid var(--line);
}
[data-testid="stTable"] thead th {
    background: #f8fafc !important; text-transform: uppercase;
    letter-spacing: .05em; font-size: .68rem; color: var(--muted) !important;
}
[data-testid="stTable"] tbody tr:hover { background: #f8fafc; }

/* Code */
pre, [data-testid="stCode"] pre {
    border-radius: 12px !important;
    border: 1px solid var(--line) !important;
    background: #0f172a0a !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px !important; font-weight: 600 !important;
    border: 1px solid var(--line) !important;
    transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: var(--shadow); }
.stButton > button[kind="primary"] { background: var(--accent) !important; border-color: var(--accent) !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 14px; }

hr { border-color: var(--line); }

/* Timeline */
.tl-stage {
    display: flex; justify-content: space-between; align-items: center;
    margin: 1.1rem 0 .4rem; font-weight: 700; color: var(--ink);
    padding-bottom: .35rem; border-bottom: 1px solid var(--line);
}
.tl-note { color: var(--muted); font-size: .85rem; margin: .1rem 0 .35rem 1.5rem; font-style: italic; }
.tl-time {
    text-align: right; color: var(--muted); font-weight: 600;
    font-variant-numeric: tabular-nums; padding-top: .62rem; font-size: .85rem;
}

/* Inline token badge */
.badge {
    display: inline-block; padding: .12rem .5rem; border-radius: 999px;
    background: #eef1f7; color: var(--muted); font-size: .72rem; font-weight: 600;
}
</style>
"""


def _inject_style() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)


def _hero(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"<div class='tech-hero'><h2>{title}</h2>" + (f"<p>{subtitle}</p>" if subtitle else "") + "</div>",
        unsafe_allow_html=True,
    )


def _section(label: str) -> None:
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Background analysis + progress
#
# The analysis runs in its own process (see topspin_review.interfaces.worker).
# In-process runs collided with Streamlit's module handling: the backend runner
# spawns subprocesses, and Streamlit's script module caused the spawned child to
# re-import the backend as a second copy, breaking pickling of message objects.
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
        bar_col, stop_col = st.columns([6, 1], vertical_alignment="bottom")
        bar_col.progress(pct / 100.0, text=f"Analyzing… {snap.get('stage', 'starting')} ({int(pct)}%) · {elapsed:.0f}s")
        if stop_col.button("Stop", key="stop_analysis"):
            try:
                proc.kill()
            except Exception:
                pass
            st.session_state.pop("job", None)
            st.session_state["job_error"] = "Analysis stopped."
            st.rerun()
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
    """Runs at the top of every page: inject style, show errors, poll the job."""
    _inject_style()
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
    _hero("Sessions", "Every video or photo you've analyzed — open one or analyze a new clip.")

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
    _hero(
        "Analyze a video or photo",
        "A few rallies from the side (full body) gives the most; a single photo gives posture feedback.",
    )

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

    _hero(render.video_name(latest), f"{latest.get('sport', '')} · {latest.get('date', '')}")

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
        usage = latest.get("usage") or {}
        artifacts = latest.get("artifacts") or {}
        models = usage.get("models") or {}
        stats = summary_stats(usage)

        _section("Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model calls", stats["calls"])
        c2.metric(
            "Tokens billed",
            f"{stats['total_tokens']:,}",
            help=f"{stats['context_tokens']:,} context + {stats['output_tokens']:,} output",
        )
        c3.metric("From cache", f"{stats['cached_tokens']:,}")
        c4.metric("Model time", f"{stats['model_seconds']:.0f}s")
        if models:
            st.caption(
                f"Text model: {models.get('text', '?')} · Vision model: {models.get('vision', '?')} · "
                f"{models.get('provider', '?')} ({models.get('api_base', '')})"
            )

        def _load_artifact(path_str):
            if not path_str:
                return None
            try:
                path = Path(path_str)
                return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            except (OSError, ValueError):
                return None

        obs_data = _load_artifact(artifacts.get("observability")) or {}
        calls = obs_data.get("calls") or []
        called_tools = obs_data.get("tools") or []
        if not calls or not called_tools:
            legacy_calls = _load_artifact(artifacts.get("calls"))
            legacy_tools = _load_artifact(artifacts.get("tools"))
            if not calls and isinstance(legacy_calls, list):
                calls = legacy_calls
            if not called_tools and isinstance(legacy_tools, list):
                called_tools = legacy_tools

        timeline = build_timeline(usage, latest.get("timings"), calls=calls, tools=called_tools)
        if timeline:
            st.divider()
            _section("Execution")
            ordered = any(r.get("seq") for r in timeline if r.get("kind") in ("model", "tool"))
            if ordered:
                st.caption("Ordered by the exact sequence in which each model/tool call started.")
            elif any(r.get("kind") in ("model", "tool") for r in timeline):
                st.caption("⚠ This report predates event timestamps — call order is not guaranteed. Re-analyze for the true sequence.")
            if calls or called_tools:
                st.download_button(
                    "Download run details (JSON)",
                    json.dumps({"calls": calls, "tools": called_tools}, ensure_ascii=False, indent=2),
                    file_name="run_details.json",
                    mime="application/json",
                )
            evt_index = 0
            event_css: list[str] = []
            prev_tools: list[str] = []
            for row in timeline:
                kind = row.get("kind")
                if kind == "stage":
                    st.markdown(
                        f"<div class='tl-stage'><span>{row.get('label', '')}</span>"
                        f"<span class='badge'>{float(row.get('seconds', 0)):.2f}s</span></div>",
                        unsafe_allow_html=True,
                    )
                elif kind == "overhead":
                    st.markdown(
                        f"<div class='tl-note'>agent overhead · {float(row.get('seconds', 0)):.2f}s</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    secs = float(row.get("seconds", 0) or 0)
                    evt_index += 1
                    key = f"evt_{evt_index}"
                    if kind == "tool":
                        title = f"Tool · {row.get('name', '')}"
                    else:
                        model = row.get("model") or ""
                        title = "↳ Model call" + (f" · {model}" if model else "")
                        if row.get("note"):
                            title += f" · {row['note']}"
                        tool_calls = row.get("requested") or row.get("called") or []
                        requested_names = [c.get("name", "") for c in tool_calls]
                        if requested_names:
                            title += "  →  calls " + ", ".join(requested_names)
                    with st.expander(title, key=key):
                        if kind == "tool":
                            st.markdown("**Input — arguments passed to the tool**")
                            st.code(str(row.get("arguments") or "{}"), language="json")
                            if row.get("error"):
                                st.error(str(row.get("error")))
                            st.markdown("**Output — result the tool returned**")
                            st.code(str(row.get("result", "")), language="text")
                        else:
                            offered = row.get("tools") or []
                            if offered:
                                st.caption("Tools offered to the model: " + ", ".join(offered))
                            requested = row.get("requested") or row.get("called") or []
                            messages = row.get("input") or []
                            st.markdown("**Input — everything sent to the model**")
                            if not messages:
                                st.caption("Input not saved (SAVE_CALL_IO is off) or unavailable.")
                            for message in messages:
                                role = message.get("role", "")
                                content = message.get("content", "")
                                if role == "assistant" and not str(content).strip():
                                    content = (
                                        "(tool call: " + ", ".join(prev_tools) + ")"
                                        if prev_tools
                                        else "(no text — the model called a tool here)"
                                    )
                                st.markdown(f"*{role}*")
                                st.code(content, language="text")
                            image_paths = [p for p in (row.get("images") or []) if Path(p).exists()]
                            if image_paths:
                                st.markdown(f"*images sent to the model ({len(image_paths)})*")
                                for path in image_paths:
                                    st.image(path)
                            prev_tools = [call.get("name", "") for call in requested]
                            st.markdown("**Output — what the model produced**")
                            if requested:
                                for call in requested:
                                    st.markdown(f"↳ requested tool `{call.get('name', '')}` with:")
                                    params = call.get("arguments")
                                    st.code(str(params) if params else "{}", language="json")
                            if row.get("output"):
                                st.code(row["output"], language="text")
                            elif requested:
                                st.caption("No text — the model only requested tool calls on this turn.")
                            elif not messages:
                                st.caption("Output not saved (SAVE_CALL_IO is off).")
                    event_css.append(
                        f".st-key-{key} summary {{ position: relative; }}"
                        f".st-key-{key} summary::after {{ content: '{secs:.2f}s'; position: absolute; "
                        f"right: 2.6rem; top: 50%; transform: translateY(-50%); color: var(--muted); "
                        f"font-size: 0.68rem; font-weight: 500; letter-spacing: .01em; "
                        f"font-variant-numeric: tabular-nums; }}"
                    )
            if event_css:
                st.markdown("<style>" + "".join(event_css) + "</style>", unsafe_allow_html=True)

    # Floating ask panel: only on the Result page, but visible across its tabs.
    _ask_panel()


# --------------------------------------------------------------------------- #
# Page: Practice
# --------------------------------------------------------------------------- #
def _page_practice() -> None:
    _prelude()
    _hero("Practice", "Turn the report's drills into this week's plan.")
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
    _hero("Progress", "How your game is trending across sessions.")
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
    _hero("You", "Your sport, level, and what you're working on.")
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
