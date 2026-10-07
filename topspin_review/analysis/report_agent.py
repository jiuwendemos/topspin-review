"""Build the report-writing DeepAgent (system prompt + tools)."""

from __future__ import annotations

from topspin_review import config
from topspin_review.analysis import prompts, tools
from topspin_review.bootstrap import setup
from topspin_review.storage import runtime


def build_agent(model=None):
    from openjiuwen.harness import create_deep_agent

    setup()

    return create_deep_agent(
        model=model or config.make_model(),
        system_prompt=prompts.REPORT_AGENT_SYSTEM,
        tools=tools.ALL_TOOLS,
        enable_task_loop=False,
        max_iterations=15,
        workspace=runtime.workspace(),
    )
