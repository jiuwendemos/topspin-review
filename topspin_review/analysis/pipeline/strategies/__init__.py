"""Pipeline strategies — the two interchangeable modes, behind a base class + registry."""

from topspin_review.analysis.pipeline.strategies.base import Strategy
from topspin_review.analysis.pipeline.strategies.resolve_strategy import (
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
    "Strategy",
    "default_name",
    "get_strategy",
    "resolve",
]
