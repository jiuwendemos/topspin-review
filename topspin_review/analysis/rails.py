"""openjiuwen rails for the report agent: cost budget + optional memory.

Rails are openjiuwen's behavioural-constraint layer. We use them to
(a) hard-stop a run that blows the token budget and (b) opt into the built-in
``MemoryRail`` when embeddings are configured.
"""

from __future__ import annotations

from typing import Any

try:  # openjiuwen is provided by the host environment
    from openjiuwen.core.single_agent.rail.base import AgentRail
except Exception:  # pragma: no cover - degrade to a plain class
    AgentRail = object  # type: ignore[assignment,misc]

from topspin_review import config, observability


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
        if config.has_embedding():
            from openjiuwen.harness.rails import MemoryRail

            rails.append(MemoryRail(embedding_config=config.embedding_config()))
    except Exception:
        pass
    return rails
