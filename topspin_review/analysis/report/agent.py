"""Build the report-writing DeepAgent.

The default agent (report system prompt + report tools) is what the deterministic
strategy uses. The agentic strategy reuses this builder, overriding the system
prompt, tools and iteration budget, so ``create_deep_agent`` is configured in one
place.
"""

from __future__ import annotations

from typing import Any

from topspin_review.analysis import prompts
from topspin_review.analysis.report import tools as report_tools
from topspin_review.backend import create_agent
from topspin_review.backend.models import make_text_model
from topspin_review.bootstrap import setup
from topspin_review.storage import runtime


def build_agent(
    model: Any = None,
    rails: list | None = None,
    *,
    system_prompt: str | None = None,
    tools: list | None = None,
    max_iterations: int = 15,
):
    """Construct a report DeepAgent.

    Defaults produce the deterministic report writer; callers (e.g. the agentic
    strategy) may override ``system_prompt``, ``tools`` and ``max_iterations``.
    """
    setup()

    return create_agent(
        model=model or make_text_model(),
        system_prompt=system_prompt or prompts.REPORT_AGENT_SYSTEM,
        tools=tools if tools is not None else report_tools.ALL_TOOLS,
        rails=rails or [],
        max_iterations=max_iterations,
        workspace=runtime.workspace(),
    )
