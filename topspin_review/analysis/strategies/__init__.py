"""Analysis strategies — import from this package, never a concrete module.

    from topspin_review.analysis import strategies

    strategy = strategies.resolve(agentic=True)      # or strategies.get_strategy("deterministic")
    result = await strategy.analyze(strategies.Params(video_path, progress=...))
"""

from topspin_review.analysis.strategies.base import Strategy
from topspin_review.analysis.strategies.params import Params
from topspin_review.analysis.strategies.resolve_strategy import (
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
    "Params",
    "Strategy",
    "default_name",
    "get_strategy",
    "resolve",
]
