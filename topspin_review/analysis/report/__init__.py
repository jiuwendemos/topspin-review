"""Report-writing layer: the DeepAgent that writes the coaching report, its tools,
rails, retrieval of past reports, and verification of its issues."""

from .agent import build_agent
from .rails import build_rails

__all__ = ["build_agent", "build_rails"]
