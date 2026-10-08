"""openjiuwen rails integration: the ``AgentRail`` base and the built-in ``MemoryRail``."""

from __future__ import annotations

from typing import Any

try:  # openjiuwen is provided by the host environment
    from openjiuwen.core.single_agent.rail.base import AgentRail
except Exception:  # pragma: no cover - degrade to a plain class
    AgentRail = object  # type: ignore[assignment,misc]


def memory_rail(embedding_config: Any) -> Any:
    """The built-in openjiuwen MemoryRail for the given embedding config."""
    from openjiuwen.harness.rails import MemoryRail

    return MemoryRail(embedding_config=embedding_config)
