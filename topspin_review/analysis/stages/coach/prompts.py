"""Prompt text for the coaching report."""

from __future__ import annotations

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

VERIFY_SYSTEM = "You verify coaching-report issues against evidence and reply with ONLY the requested JSON."

VERIFY_PROMPT = """You are checking a coaching report against the evidence.

Issues drafted:
{issues}

What the measurements and observations actually show:
{evidence}

Return ONLY JSON: {{"supported": [list of issue indices, 0-based, that the evidence supports],
"unsupported_reasons": {{"<index>": "why"}}}}
Be strict: an issue with no supporting evidence or evidence times must be dropped."""

RUBRICS: dict[str, list[str]] = {
    "table tennis": [
        "ready stance and balance",
        "split-step on the opponent's contact",
        "footwork / stepping into the ball",
        "weight transfer into the stroke",
        "stroke preparation (early, compact)",
        "recovery to ready between shots",
    ],
    "tennis": [
        "ready stance and balance",
        "split-step",
        "unit turn and footwork",
        "weight transfer / loaded legs",
        "preparation and take-back",
        "recovery",
    ],
    "badminton": [
        "ready stance and balance",
        "split-step",
        "footwork / lunging",
        "weight transfer",
        "racket preparation",
        "recovery to base",
    ],
    "squash": [
        "ready stance and balance",
        "split-step",
        "movement to the T",
        "weight transfer",
        "racket preparation",
        "recovery",
    ],
    "padel": [
        "ready stance and balance",
        "split-step",
        "footwork at the net",
        "weight transfer",
        "racket preparation",
        "recovery",
    ],
}


def rubric_text(sport: str, language: str = "en") -> str:
    criteria = RUBRICS.get((sport or "").strip().lower(), RUBRICS["table tennis"])
    return "\n".join(f"- {c}" for c in criteria)
