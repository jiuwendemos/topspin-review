"""Analysis — a pipeline of two stages, run by two interchangeable strategies.

- :mod:`~topspin_review.analysis.stages.observe` — stage 1: video → observations.
- :mod:`~topspin_review.analysis.stages.coach` — stage 2: observations → report.
- :mod:`~topspin_review.analysis.pipeline` — the orchestrator: runs the stages in one
  of two modes (`pipeline/strategies/`) and holds the run machinery.

Side-effect free.
"""
