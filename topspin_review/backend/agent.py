"""The agents file: the single entry point for building a model-backed agent.

External code calls :func:`create_agent` (or the app-level report policy that wraps
it) and never constructs a model itself. The text model is created here and, when
``usage``/``trace`` are supplied, instrumented with observability — so
``backend.models`` and ``backend.observability`` stay internal details.
"""

from __future__ import annotations

from typing import Any

from topspin_review.backend import observability
from topspin_review.backend.models import make_text_model


def create_agent(
    *,
    system_prompt: str,
    tools: list,
    model: Any = None,
    rails: list | None = None,
    max_iterations: int = 15,
    workspace: str | None = None,
    usage: observability.UsageCollector | None = None,
    trace: observability.CallTrace | None = None,
) -> Any:
    """Build a text DeepAgent.

    If ``model`` is omitted the text model is constructed (and instrumented for
    usage/tracing when ``usage``/``trace`` are given), so callers never touch
    :mod:`topspin_review.backend.models`.
    """
    from openjiuwen.harness import create_deep_agent

    if model is None:
        model = make_text_model()
        if usage is not None or trace is not None:
            model = observability.attach(model, usage, trace)

    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        rails=rails or [],
        enable_task_loop=False,
        max_iterations=max_iterations,
        workspace=workspace,
    )
