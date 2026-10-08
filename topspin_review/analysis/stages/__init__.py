"""Pipeline stages: ``observe`` (video → observations) and ``coach`` (observations → report)."""

from topspin_review.analysis.stages import coach, observe

__all__ = ["coach", "observe"]
