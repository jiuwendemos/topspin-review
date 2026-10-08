"""Pipeline orchestration — import from this package, never a concrete module.

    from topspin_review.analysis import pipeline

    strategy = pipeline.resolve(agentic=True)      # or pipeline.get_strategy("deterministic")
    result = await strategy.analyze(pipeline.Params(video_path, progress=...))
"""

from topspin_review.analysis.pipeline.params import Params
from topspin_review.analysis.pipeline.strategies import (
    AGENTIC,
    DETERMINISTIC,
    STRATEGIES,
    Strategy,
    default_name,
    get_strategy,
    resolve,
)

__all__ = [
    "AGENTIC",
    "DETERMINISTIC",
    "STRATEGIES",
    "Params",
    "Strategy",
    "default_name",
    "get_strategy",
    "resolve",
]
