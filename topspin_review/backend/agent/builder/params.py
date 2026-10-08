"""Parameters for :mod:`topspin_review.backend.agent.builder`.

Callers describe the agent they want with one of these dataclasses and hand it to
:func:`topspin_review.backend.agent.builder.build`. Nothing else (models, rails,
tools, decoration, telemetry) is passed separately — the params and the builder do
the whole work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Params:
    """Common inputs for building any backend agent."""

    system_prompt: str
    workspace: str | None = None
    max_iterations: int = 15
    record: bool = True
    """Capture usage + call/tool traces for this agent (built by the builder)."""
    media_dir: str | None = None
    """Where call-I/O artifacts (prompts, images) are saved when recording."""
    recorder: Any = None
    """Reuse an existing run recorder (built by the builder) to share one trace."""


@dataclass
class TextParams(Params):
    """A text DeepAgent (report writer, single-shot verify/Q&A, …).

    ``tools`` are plain callables (the builder decorates them), ``rails`` are rail
    *names* (the builder resolves them).
    """

    tools: list[Callable] = field(default_factory=list)
    rails: list[str] = field(default_factory=list)
    max_iterations: int = 15


@dataclass
class VisionParams(Params):
    """A multimodal vision DeepAgent that reads images via ``read_file``."""

    max_iterations: int = 4
