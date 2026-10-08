"""Backend layer: the single home for everything openjiuwen and provider related.

- :mod:`~topspin_review.backend.settings` — connection settings (provider, keys, URLs, model names, timeouts).
- :mod:`~topspin_review.backend.models` — openjiuwen model-client construction.
- :mod:`~topspin_review.backend.providers` — one module per vision provider, behind ``get_backend``.
- :mod:`~topspin_review.backend.agent` — DeepAgent construction.
- :mod:`~topspin_review.backend.rails` — the ``AgentRail`` base and ``MemoryRail``.
- :mod:`~topspin_review.backend.tools` — the ``@tool`` decorator.
- :mod:`~topspin_review.backend.runner` — Runner lifecycle, agent execution, callback events.
- :mod:`~topspin_review.backend.logs` — openjiuwen logging setup.

No ``from openjiuwen...`` import exists outside this package; no application
module is imported by it. Application behaviour stays in :mod:`topspin_review.config`.
"""

from topspin_review.backend.agent import create_agent
from topspin_review.backend.providers import (
    PROVIDERS,
    MockVisionBackend,
    OpenAIVisionBackend,
    VisionBackend,
    VisionResult,
    get_backend,
)
from topspin_review.backend.rails import AgentRail, memory_rail
from topspin_review.backend.tools import tool

__all__ = [
    "PROVIDERS",
    "AgentRail",
    "MockVisionBackend",
    "OpenAIVisionBackend",
    "VisionBackend",
    "VisionResult",
    "create_agent",
    "get_backend",
    "memory_rail",
    "tool",
]
