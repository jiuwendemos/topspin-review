"""Analysis, grouped by business area.

- :mod:`~topspin_review.analysis.vision` — turn sampled frames into observations.
- :mod:`~topspin_review.analysis.coaching` — turn observations into the coaching
  report (report agent, tools, verification, retrieval of past sessions).
- :mod:`~topspin_review.analysis.strategies` — the analysis modes (`analyze` /
  `analyze_agentic`), the run machinery (`run_session`, `progress`) and each mode's
  own prompt text.

Each area owns its prompts; there are no shared modules at this level. Side-effect free.
"""
