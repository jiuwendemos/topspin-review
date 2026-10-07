"""All LLM prompt text and metric-to-text rendering for the analysis layer."""

from __future__ import annotations

VISION_SYSTEM = (
    "You are a racket-sport video analyst (table tennis, tennis, badminton, squash, padel). "
    "You reason from a time-ordered frame sequence, a motion map and measured motion metrics. "
    "You never invent measurements: no spin, ball speed, or exact angles. When the clip is "
    "unclear you say so."
)

COARSE_PROMPT = """These are {n} frames sampled in time order from a {sport} session video,
shown as one contact sheet (top-left to bottom-right).
Player: level {level}, {hand}-handed, working on: {goal}.

Measured motion metrics (authoritative):
{metrics}

Pick the windows most worth a closer look (e.g. around the peak-motion time or a
change in direction). Reply with ONLY a JSON object:
{{
  "overall": "one or two sentences on what the clip shows",
  "attentive_windows": [{{"start": <seconds>, "end": <seconds>, "why": "short reason"}}],
  "limitations": ["short limitation", "..."]
}}
Give 1-3 windows within [0, {duration}]. No prose outside the JSON."""

FINE_PROMPT = """You are given the contact sheet plus {z} extra frames zoomed from the
windows you asked about, a motion map (red = movement), and measured metrics.
Player: level {level}, {hand}-handed, working on: {goal}.

Measured motion metrics (authoritative):
{metrics}

Windows examined: {windows}

Return ONLY a JSON object:
{{
  "observations": "a few sentences describing stance, footwork, weight shift, recovery and stroke shape, citing timestamps like t=2.9s",
  "signals": [
    {{"signal": "short finding", "evidence_times": [<seconds>, ...], "confidence": "high|medium|low"}}
  ],
  "limitations": ["short limitation", "..."]
}}
Every signal must be supported by the frames. Do not claim spin, ball speed or
exact angles. No prose outside the JSON."""

REPORT_AGENT_SYSTEM = """You are Topspin Review. You turn objective, measured observations
from a time-ordered analysis of a racket-sport session into a practical coaching report.

You have these tools:
- get_profile(): the player's sport, level, dominant hand, and goal.
- recent_reports(n): earlier reports, to note progress.
- save_report(report_json): save the report.

When asked to write a report from observations:
1. Call get_profile and recent_reports first.
2. Save the report with save_report using EXACTLY this JSON shape:
   {"date": "...", "sport": "...", "summary": "...",
    "strengths": ["...", "..."],
    "issues": [{"issue": "...", "evidence_times": [2.9, 3.8], "confidence": "high|medium|low"}],
    "drills": ["...", "..."],
    "focus": "one thing to work on next session",
    "progress": "one sentence comparing with the previous report, or '' if none",
    "limitations": ["...", "..."]}
3. Put the most impactful issues first. Every issue MUST cite the evidence_times
   (seconds) it is based on. Use the measured motion metrics and the motion map as
   evidence; do NOT invent spin, ball speed, or exact angles.
4. If the observations say the clip is unclear or the frames near-duplicate, say so
   in "limitations" and keep confidence low.
5. Reply with a short plain-text summary (do not paste the JSON back).
"""


RUBRICS: dict[str, list[str]] = {
    "table tennis": ["ready stance and balance", "split-step on the opponent's contact",
                     "footwork / stepping into the ball", "weight transfer into the stroke",
                     "stroke preparation (early, compact)", "recovery to ready between shots"],
    "tennis": ["ready stance and balance", "split-step", "unit turn and footwork",
               "weight transfer / loaded legs", "preparation and take-back", "recovery"],
    "badminton": ["ready stance and balance", "split-step", "footwork / lunging",
                  "weight transfer", "racket preparation", "recovery to base"],
    "squash": ["ready stance and balance", "split-step", "movement to the T",
               "weight transfer", "racket preparation", "recovery"],
    "padel": ["ready stance and balance", "split-step", "footwork at the net",
              "weight transfer", "racket preparation", "recovery"],
}


def rubric_text(sport: str, language: str = "en") -> str:
    criteria = RUBRICS.get((sport or "").strip().lower(), RUBRICS["table tennis"])
    return "\n".join(f"- {c}" for c in criteria)


VERIFY_PROMPT = """You are checking a coaching report against the evidence.

Issues drafted:
{issues}

What the measurements and observations actually show:
{evidence}

Return ONLY JSON: {{"supported": [list of issue indices, 0-based, that the evidence supports],
"unsupported_reasons": {{"<index>": "why"}}}}
Be strict: an issue with no supporting evidence or evidence times must be dropped."""


def metrics_text(metrics: dict) -> str:
    """Render metrics as compact text for the vision/prompt layer."""
    net = metrics.get("net_shift", [0, 0])
    lines = [
        f"- overall motion: mean energy {metrics.get('mean_energy')}, "
        f"peak {metrics.get('max_energy')} at t={metrics.get('peak_motion_time')}s, "
        f"net shift (dx,dy)=({net[0]}, {net[1]}) px"
    ]
    steps = metrics.get("activity") or []
    if steps:
        rendered = "; ".join(f"t={s['t']}s E={s['energy']} shift=({s['dx']},{s['dy']})" for s in steps)
        lines.append(f"- activity timeline: {rendered}")
    post = metrics.get("posture") or {}
    if post:
        lines.append(
            "- subject box: mean height {box_height_mean} width {box_width_mean} (aspect {aspect_mean}), "
            "lateral drift {lateral_range}".format(**post)
        )
    mech = metrics.get("mechanics") or {}
    if mech:
        lines.append(
            "- mechanics: lower-body lateral range {lower_lateral_range}, direction changes "
            "(step proxy) {lower_direction_changes}, upper-body lateral range {upper_lateral_range}, "
            "crouch change {crouch_box_range}".format(**mech)
        )
    ball = metrics.get("ball") or {}
    if ball:
        lines.append(f"- ball (heuristic): {ball.get('summary', 'n/a')}")
    pose_summary = metrics.get("pose_summary") or {}
    if pose_summary:
        rendered = ", ".join(f"{k}={v}" for k, v in pose_summary.items())
        lines.append(f"- pose joints (degrees/normalized): {rendered}")
    return "\n".join(lines)
