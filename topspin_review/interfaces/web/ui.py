"""Streamlit UI for Topspin Review.

Run:
    streamlit run topspin_review/interfaces/web/ui.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import streamlit as st

from topspin_review import config
from topspin_review.bootstrap import setup
from topspin_review.domain import compare as compare_mod
from topspin_review.domain import evaluate, render
from topspin_review import reporting as export
from topspin_review.storage import runtime, store

setup()

st.set_page_config(page_title="Topspin Review", page_icon="🏓", layout="wide")

st.title("🏓 Topspin Review")
st.caption("Analyze a session video — measured motion, footwork, and drills.")

with st.sidebar:
    st.header("Profile")
    profile = store.get_profile()
    sport = st.text_input("Sport", value=profile.get("sport", "table tennis"))
    level = st.text_input("Level", value=profile.get("level", "intermediate"))
    hand = st.text_input("Dominant hand", value=profile.get("dominant_hand", "right"))
    goal = st.text_input("Goal", value=profile.get("goal", "improve"))
    if st.button("Save profile"):
        store.set_profile({"sport": sport, "level": level, "dominant_hand": hand, "goal": goal})
        st.success("Profile saved.")

    st.divider()
    st.subheader("Analyze a video")
    uploaded = st.file_uploader("Upload a clip", type=["mp4", "mov", "avi", "mkv"])
    use_sample = st.checkbox("Use runtime/data/sample.mp4")

    st.caption("Optional: restrict analysis to the player (normalized 0..1)")
    use_box = st.checkbox("Use player region")
    box = None
    if use_box:
        left = st.slider("left", 0.0, 1.0, 0.0, 0.05)
        top = st.slider("top", 0.0, 1.0, 0.0, 0.05)
        right = st.slider("right", 0.0, 1.0, 1.0, 0.05)
        bottom = st.slider("bottom", 0.0, 1.0, 1.0, 0.05)
        box = (left, top, right, bottom)

    if st.button("Analyze", type="primary"):
        if uploaded is not None:
            target = runtime.DATA_DIR / f"upload_{uploaded.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(uploaded.getbuffer())
        elif use_sample:
            target = runtime.DATA_DIR / "sample.mp4"
        else:
            target = None
        if target is None:
            st.warning("Upload a clip or tick “Use runtime/data/sample.mp4”.")
        elif not target.exists():
            st.error(f"Not found: {target}")
        else:
            with st.spinner("Measuring motion and analyzing frames…"):
                from topspin_review.analysis import pipeline

                try:
                    asyncio.run(pipeline.analyze(str(target), region_box=box))
                    st.success("Done.")
                    st.rerun()
                except (config.ConfigError, FileNotFoundError) as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)

    if st.button("Reset reports"):
        store.reset()
        st.session_state.pop("seek", None)
        st.rerun()

reports = store.get_reports()
latest = reports[-1] if reports else {}
artifacts = latest.get("artifacts") or {}
metrics = latest.get("metrics") or {}

st.subheader(f"{latest.get('sport', '')} — {latest.get('date', '')}" if latest else "No report yet")

tab_report, tab_motion, tab_progress, tab_compare, tab_history = st.tabs(
    ["Report", "Motion", "Progress", "Compare", "History"]
)

with tab_report:
    if not latest:
        st.info("No report yet. Analyze a clip from the sidebar.")
    else:
        src = latest.get("source")
        seek = int(st.session_state.get("seek", 0) or 0)
        if src and Path(src).exists():
            try:
                st.video(src, start_time=seek)
            except TypeError:
                st.video(src)
            if seek:
                st.caption(f"Jumped to t={seek}s.")
        else:
            st.caption(f"(video not found: {src})")

        if latest.get("summary"):
            st.write(latest["summary"])
        for label, key in (("Strengths", "strengths"), ("Drills", "drills")):
            items = latest.get(key) or []
            if items:
                st.markdown(f"**{label}**")
                for item in items:
                    st.markdown(f"- {item}")

        issues = latest.get("issues") or []
        if issues:
            st.markdown("**Issues** (click a timestamp to jump the video)")
            for idx, item in enumerate(issues):
                if isinstance(item, dict):
                    conf = item.get("confidence") or "?"
                    st.markdown(f"- {item.get('issue', '')} · <small>confidence {conf}</small>", unsafe_allow_html=True)
                    times = item.get("evidence_times") or []
                    if times:
                        cols = st.columns(len(times))
                        for c, t in zip(cols, times):
                            if c.button(f"▶ t={t}s", key=f"seek_{idx}_{t}"):
                                st.session_state.seek = int(float(t))
                                st.rerun()
                else:
                    st.markdown(f"- {item}")

        if latest.get("focus"):
            st.success(f"Focus next session: {latest['focus']}")
        if latest.get("progress"):
            st.info(f"Progress vs last time: {latest['progress']}")
        if latest.get("limitations"):
            st.caption("Limitations: " + "; ".join(latest["limitations"]))

        quality = evaluate.score(latest)
        st.caption(
            f"Quality: {quality['score']}/100 · evidence coverage {quality['evidence_coverage']} · "
            f"confidence {quality['confidence']}"
            + (f" · flags: {quality['hallucination_flags']}" if quality["hallucination_flags"] else "")
        )
        usage = latest.get("usage") or {}
        if usage:
            st.caption(
                f"Vision usage: {usage.get('calls', 0)} calls · "
                f"{usage.get('total_tokens', 0)} tokens · {usage.get('seconds', 0)}s"
            )

        if src:
            if st.button("Export Markdown + HTML"):
                paths = export.write(latest)
                st.success("Exported: " + ", ".join(paths.values()))

with tab_motion:
    if not metrics:
        st.info("No motion data yet.")
    else:
        activity = metrics.get("activity") or []
        st.caption(
            f"mean energy {metrics.get('mean_energy')} · peak at t={metrics.get('peak_motion_time')}s · "
            f"net shift {metrics.get('net_shift')}"
        )
        if activity:
            st.line_chart({"energy": [s["energy"] for s in activity]})
            st.caption("Movement energy between consecutive sampled frames.")
        mech = metrics.get("mechanics") or {}
        if mech:
            st.markdown("**Mechanics (proxy)**")
            st.json(mech)
        if metrics.get("ball"):
            st.markdown("**Ball / rally (heuristic)**")
            st.write(metrics["ball"].get("summary", ""))
            rallies = metrics["ball"].get("rallies") or []
            if rallies:
                st.json(rallies)
        if metrics.get("pose"):
            st.markdown("**Pose (mediapipe)**")
            st.json(metrics["pose"])
        for key, caption in (
            ("motion", "Motion map (red = movement)"),
            ("pose", "Pose overlay"),
            ("frames", "Sampled frames"),
        ):
            path = artifacts.get(key)
            if path and Path(path).exists():
                st.markdown(f"**{caption}**")
                st.image(path)

with tab_progress:
    trend = latest.get("progress_trend") or {}
    if not trend:
        st.info("No progress data yet.")
    else:
        st.write(trend.get("text", ""))
        st.caption(
            f"sessions: {trend.get('sessions')} · repeated themes: "
            f"{', '.join(trend.get('repeated_themes') or []) or 'none'} · "
            f"last focus: {trend.get('focus_status', 'n/a')}"
        )
        counts = trend.get("issue_counts") or []
        if counts:
            st.line_chart({"issues per session": counts})
        theme_trend = trend.get("theme_trend") or {}
        if theme_trend:
            st.markdown("**Theme counts per session**")
            st.json(theme_trend)
        focus_history = trend.get("focus_history") or []
        if focus_history:
            st.markdown("**Focus history**")
            for i, focus in enumerate(focus_history, 1):
                st.markdown(f"{i}. {focus}")

with tab_compare:
    if len(reports) < 2:
        st.info("Analyze at least two videos to compare.")
    else:
        names = [render.video_name(r) for r in reports]
        older = st.selectbox("Older", names, index=0)
        newer = st.selectbox("Newer", names, index=len(names) - 1)
        if st.button("Compare") and older != newer:
            a = reports[names.index(older)]
            b = reports[names.index(newer)]
            result = compare_mod.compare(a, b)
            st.write(result["text"])
            if result["metric_deltas"]:
                st.json(result["metric_deltas"])

with tab_history:
    if not reports:
        st.info("No reports yet.")
    for report in reversed(reports[-10:]):
        st.markdown(
            f"- **{render.video_name(report)}** · {report.get('date', '')} — {report.get('focus', '')}"
        )
