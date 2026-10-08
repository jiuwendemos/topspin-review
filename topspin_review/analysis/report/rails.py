"""openjiuwen rails for the report agent: cost budget + optional memory.

Rails are openjiuwen's behavioural-constraint layer. We use them to
(a) hard-stop a run that blows the token budget and (b) opt into the built-in
``MemoryRail`` when embeddings are configured.
"""

from __future__ import annotations

from typing import Any

from topspin_review import config, observability
from topspin_review.backend import AgentRail, memory_rail
from topspin_review.backend import settings as backend_settings


class TokenBudgetRail(AgentRail):
    """Force-finish the run once cumulative model tokens exceed the budget."""

    priority = 70

    def __init__(self) -> None:
        super().__init__()
        self.used = 0

    async def after_model_call(self, ctx: Any, **kwargs: Any) -> None:
        budget = config.token_budget()
        if budget <= 0:
            return
        response = getattr(getattr(ctx, "inputs", None), "response", None)
        self.used += int(observability.extract_usage(response).get("total_tokens", 0) or 0)
        if self.used > budget:
            try:
                ctx.request_force_finish(
                    {"reason": "token_budget_exceeded", "used": self.used, "budget": budget}
                )
            except Exception:
                pass


def build_rails() -> list:
    """Assemble the report agent's rails (best-effort; never raises)."""
    rails: list = []
    if config.token_budget() > 0:
        rails.append(TokenBudgetRail())
    try:
        if backend_settings.has_embedding():
            rails.append(memory_rail(backend_settings.embedding_config()))
    except Exception:
        pass
    return rails
