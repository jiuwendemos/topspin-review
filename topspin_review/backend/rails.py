"""openjiuwen rails integration: the ``AgentRail`` base, a token-budget rail,
``MemoryRail`` and the rail policy for a report agent.

This is under-the-hood agentic machinery. The policy here reads only backend
settings (token budget, embeddings), so no application module is imported.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from topspin_review.backend import observability
from topspin_review.backend import settings as backend_settings

try:  # openjiuwen is provided by the host environment
    from openjiuwen.core.single_agent.rail.base import AgentRail
except Exception:  # pragma: no cover - degrade to a plain class
    AgentRail = object  # type: ignore[assignment,misc]


def memory_rail(embedding_config: Any) -> Any:
    """The built-in openjiuwen MemoryRail for the given embedding config."""
    from openjiuwen.harness.rails import MemoryRail

    return MemoryRail(embedding_config=embedding_config)


class TokenBudgetRail(AgentRail):
    """Force-finish the run once counted tokens exceed ``budget``.

    ``count_tokens(response)`` is supplied by the caller so this stays free of
    application concerns.
    """

    priority = 70

    def __init__(self, budget: int, count_tokens: Callable[[Any], int]) -> None:
        super().__init__()
        self._budget = budget
        self._count_tokens = count_tokens
        self.used = 0

    async def after_model_call(self, ctx: Any, **kwargs: Any) -> None:
        if self._budget <= 0:
            return
        response = getattr(getattr(ctx, "inputs", None), "response", None)
        self.used += self._count_tokens(response)
        if self.used > self._budget:
            try:
                ctx.request_force_finish(
                    {"reason": "token_budget_exceeded", "used": self.used, "budget": self._budget}
                )
            except Exception:
                pass


def _count_tokens(response: Any) -> int:
    return int(observability.extract_usage(response).get("total_tokens", 0) or 0)


def build_rails() -> list:
    """Rails for a report agent: token budget (when set) + memory (when embeddings set)."""
    rails: list = []
    budget = backend_settings.token_budget()
    if budget > 0:
        rails.append(TokenBudgetRail(budget, _count_tokens))
    try:
        if backend_settings.has_embedding():
            rails.append(memory_rail(backend_settings.embedding_config()))
    except Exception:
        pass
    return rails
