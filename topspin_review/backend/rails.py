"""openjiuwen rails: the ``AgentRail`` base, ``TokenBudgetRail``, ``MemoryRail``,
and the name → rail resolution used by the agents file.

Rails are under-the-hood machinery. Application code selects rails only by *name*
(via config); it never constructs them — :func:`resolve` builds the named rails
here and :func:`topspin_review.backend.agent_builder.build` attaches them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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


def sys_operation_rail() -> Any:
    """The openjiuwen SysOperationRail (mounts ``read_file`` etc.; required for vision)."""
    from openjiuwen.harness.rails import SysOperationRail

    return SysOperationRail()


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


def _token_budget_rail() -> Any | None:
    budget = backend_settings.token_budget()
    return TokenBudgetRail(budget, _count_tokens) if budget > 0 else None


def _memory_rail() -> Any | None:
    if not backend_settings.has_embedding():
        return None
    return memory_rail(backend_settings.embedding_config())


# Known rail names → factory (returns a rail, or None when it doesn't apply).
RAILS: dict[str, Callable[[], Any | None]] = {
    "token_budget": _token_budget_rail,
    "memory": _memory_rail,
}


def resolve(names: Iterable[str] | None) -> list:
    """Build the rails named in ``names`` (unknown names skipped; never raises)."""
    rails: list = []
    for name in names or ():
        factory = RAILS.get(str(name).strip())
        if factory is None:
            continue
        try:
            rail = factory()
        except Exception:
            rail = None
        if rail is not None:
            rails.append(rail)
    return rails
