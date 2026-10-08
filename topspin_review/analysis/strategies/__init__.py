"""Analysis strategies — import from this package, never a concrete module.

    from topspin_review.analysis import strategies

    strategy = strategies.resolve(agentic=True)      # or strategies.get_strategy("deterministic")
    result = await strategy.analyze(video_path, progress=...)
"""

from topspin_review.analysis.strategies.base import AnalyzeFn, Strategy
from topspin_review.analysis.strategies.registry import (
    AGENTIC,
    DETERMINISTIC,
    STRATEGIES,
    default_name,
    get_strategy,
    resolve,
)

__all__ = [
    "AGENTIC",
    "DETERMINISTIC",
    "STRATEGIES",
    "AnalyzeFn",
    "Strategy",
    "default_name",
    "get_strategy",
    "resolve",
]
