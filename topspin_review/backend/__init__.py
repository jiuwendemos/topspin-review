"""Backend layer: the single home for everything under-the-hood of the agentic system.

- :mod:`~topspin_review.backend.settings` — backend settings (provider, keys, URLs, model names, timeouts, embeddings, budget/tracing).
- :mod:`~topspin_review.backend.models` — openjiuwen model-client construction (internal).
- :mod:`~topspin_review.backend.providers` — one module per vision provider, behind ``get_backend``.
- :mod:`~topspin_review.backend.agent` — the agents file: ``create_agent()`` builds the model, resolves rails, and constructs the DeepAgent.
- :mod:`~topspin_review.backend.rails` — rail implementations and name resolution (internal).
- :mod:`~topspin_review.backend.observability` — usage/trace capture and the execution timeline.
- :mod:`~topspin_review.backend.tools` — ``ToolSpec`` + ``decorate()`` (openjiuwen tool decoration).
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
from topspin_review.backend.tools import ToolSpec, decorate

__all__ = [
    "PROVIDERS",
    "MockVisionBackend",
    "OpenAIVisionBackend",
    "ToolSpec",
    "VisionBackend",
    "VisionResult",
    "create_agent",
    "decorate",
    "get_backend",
]
