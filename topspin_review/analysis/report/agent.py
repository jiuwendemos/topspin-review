"""Report-agent policy: the prompt, tools and iterations for the report writer.

Thin applicative policy over :func:`topspin_review.backend.create_agent` — the
model is built and instrumented inside the backend agents file, never here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from topspin_review.analysis import prompts
from topspin_review.analysis.report import tools as report_tools
from topspin_review.backend import create_agent
from topspin_review.bootstrap import setup
from topspin_review.storage import runtime


def build_agent(
    *,
    rails: Sequence[str] | None = None,
    usage: Any = None,
    trace: Any = None,
    system_prompt: str | None = None,
    tools: list | None = None,
    max_iterations: int = 15,
):
    """Build the report-writing DeepAgent (system prompt + report tools)."""
    setup()
    return create_agent(
        system_prompt=system_prompt or prompts.REPORT_AGENT_SYSTEM,
        tools=tools if tools is not None else report_tools.ALL_TOOLS,
        rails=rails or [],
        max_iterations=max_iterations,
        workspace=runtime.workspace(),
        usage=usage,
        trace=trace,
    )


def build_single_shot_agent(system_prompt: str, *, max_iterations: int = 1):
    """A tool-less, single-turn agent for one text question/answer."""
    setup()
    return create_agent(
        system_prompt=system_prompt,
        tools=[],
        rails=[],
        max_iterations=max_iterations,
        workspace=runtime.workspace(),
    )
