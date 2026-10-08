"""openjiuwen DeepAgent construction.

The only place ``create_deep_agent`` is imported; callers describe *what* agent
they want, not *how* openjiuwen builds it.
"""

from __future__ import annotations

from typing import Any


def create_agent(
    *,
    model: Any,
    system_prompt: str,
    tools: list,
    rails: list | None = None,
    max_iterations: int = 15,
    workspace: str | None = None,
) -> Any:
    """Build a report DeepAgent (single task loop, bounded iterations)."""
    from openjiuwen.harness import create_deep_agent

    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        rails=rails or [],
        enable_task_loop=False,
        max_iterations=max_iterations,
        workspace=workspace,
    )
