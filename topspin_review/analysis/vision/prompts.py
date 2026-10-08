"""Prompt text for the vision passes."""

from __future__ import annotations

VISION_AGENT_SYSTEM = (
    "You are a racket-sport video analyst. You inspect images (contact sheets, motion "
    "maps, pose overlays, frames) by calling the read_file tool on the file paths you "
    "are given. Reason from what you see and the measured metrics. Never invent "
    "measurements: no spin, ball speed, or exact angles. Reply with ONLY the requested "
    "JSON object, with no prose outside it."
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

STILL_PROMPT = """This is a single still photo of a {sport} player.
Player: level {level}, {hand}-handed, working on: {goal}.

Describe objectively what the STATIC posture shows: ready stance, balance, knee bend,
torso lean, feet position and the racket/paddle position. You CANNOT assess movement,
footwork timing, spin, ball speed, or exact angles from one still image — say so in
"limitations".

Return ONLY a JSON object:
{{
  "observations": "a few sentences on the static posture",
  "signals": [{{"signal": "short finding", "evidence_times": [0], "confidence": "low|medium|high"}}],
  "limitations": ["single still image — motion and footwork cannot be assessed"]
}}"""
