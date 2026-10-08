"""Analysis strategies — import from this package, never a concrete module.

    from topspin_review.analysis import strategies

    strategy = strategies.resolve(agentic=True)      # or strategies.get_strategy("deterministic")
    result = await strategy.analyze(video_path, progress=...)
"""

from topspin_review.analysis.strategies.resolve_strategy import (
    AGENTIC,
    DETERMINISTIC,
    STRATEGIES,
    default_name,
    get_strategy,
    resolve,
)
from topspin_review.analysis.strategies.strategy import AnalyzeFn, Strategy

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
