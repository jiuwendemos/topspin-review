"""Single-shot text agent helper.

Every text LLM interaction goes through an openjiuwen agent + Runner (never a
direct ``Model.invoke``). This covers the tool-less, one-turn cases: report
verification and report Q&A.
"""

from __future__ import annotations

from typing import Any

from topspin_review.analysis.report.agent import build_single_shot_agent
from topspin_review.analysis.session import ensure_runner, run_agent


def _output_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, dict):
        for key in ("output", "content", "answer", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return value
            if value is not None and not isinstance(value, (dict, list)):
                return str(value)
        return str(result)
    return getattr(result, "content", None) or str(result)


async def run_text_agent(system_prompt: str, user_prompt: str, *, max_iterations: int = 1) -> str:
    """Run a single-turn, tool-less agent and return its text output."""
    await ensure_runner()
    agent = build_single_shot_agent(system_prompt, max_iterations=max_iterations)
    return _output_text(await run_agent(agent, user_prompt))
