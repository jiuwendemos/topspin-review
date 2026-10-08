"""The agents file: the single entry point for building a model-backed agent.

External code calls :func:`create_agent` (or the app-level report policy that wraps
it) and never constructs a model or a rail itself. The text model is created here
and, when ``usage``/``trace`` are supplied, instrumented with observability. Rails
are requested by *name* and resolved/built here — so ``backend.models``,
``backend.observability`` and the rail implementations stay internal details.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from topspin_review.backend import observability
from topspin_review.backend import rails as backend_rails
from topspin_review.backend import tools as backend_tools
from topspin_review.backend.models import make_text_model


def create_agent(
    *,
    system_prompt: str,
    tools: Sequence[backend_tools.ToolSpec] | None = None,
    model: Any = None,
    rails: Sequence[str] | None = None,
    max_iterations: int = 15,
    workspace: str | None = None,
    usage: observability.UsageCollector | None = None,
    trace: observability.CallTrace | None = None,
) -> Any:
    """Build a text DeepAgent.

    ``tools`` is a list of :class:`~topspin_review.backend.tools.ToolSpec` (plain
    callables + schema) which are decorated here. ``rails`` is a list of rail *names*
    (e.g. ``["token_budget", "memory"]``) which are resolved and built here. If
    ``model`` is omitted the text model is constructed (and instrumented for
    usage/tracing when ``usage``/``trace`` are given), so callers never touch
    :mod:`topspin_review.backend.models` or the openjiuwen tool decorator.
    """
    from openjiuwen.harness import create_deep_agent

    if model is None:
        model = make_text_model()
        if usage is not None or trace is not None:
            model = observability.attach(model, usage, trace)

    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=backend_tools.decorate_all(tools),
        rails=backend_rails.resolve(rails),
        enable_task_loop=False,
        max_iterations=max_iterations,
        workspace=workspace,
    )
