"""Report-agent policy: the prompt, tools and iterations for the report writer.

Thin applicative policy over the backend agent builder — the model, tool
decoration, rails and observability happen inside the builder, never here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from topspin_review.analysis import prompts
from topspin_review.analysis.report import tools as report_tools
from topspin_review.backend import TextParams, build
from topspin_review.bootstrap import setup
from topspin_review.storage import runtime


def build_agent(
    *,
    rails: Sequence[str] | None = None,
    recorder: Any = None,
    system_prompt: str | None = None,
    tools: list | None = None,
    max_iterations: int = 15,
):
    """Build the report-writing DeepAgent (system prompt + report tools)."""
    setup()
    return build(
        TextParams(
            system_prompt=system_prompt or prompts.REPORT_AGENT_SYSTEM,
            tools=tools if tools is not None else report_tools.ALL_TOOLS,
            rails=list(rails) if rails else [],
            max_iterations=max_iterations,
            workspace=runtime.workspace(),
            recorder=recorder,
        )
    ).agent


def build_single_shot_agent(system_prompt: str, *, max_iterations: int = 1):
    """A tool-less, single-turn agent for one text question/answer."""
    setup()
    return build(
        TextParams(
            system_prompt=system_prompt,
            tools=[],
            rails=[],
            max_iterations=max_iterations,
            workspace=runtime.workspace(),
            record=False,
        )
    ).agent
