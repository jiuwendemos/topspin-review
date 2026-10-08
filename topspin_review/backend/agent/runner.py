"""openjiuwen Runner lifecycle, agent execution and callback-event bridge.

The only place ``Runner``/callback event types are imported. Application code
starts the runner, runs agents and subscribes to model/tool events through here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_runner_started = False


async def start() -> None:
    """Start the global Runner once per process (best-effort, never raises)."""
    global _runner_started
    if _runner_started:
        return
    from openjiuwen.core.runner import Runner

    try:
        await Runner.start()
    except Exception:
        pass
    _runner_started = True


async def run_agent(agent: Any, query: str) -> Any:
    """Run a DeepAgent query on the global Runner (starts it if needed)."""
    await start()
    from openjiuwen.core.runner import Runner

    return await Runner.run_agent(agent, {"query": query})


def _output_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, dict):
        for key in ("output", "content", "answer", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        return str(result)
    return getattr(result, "content", None) or str(result)


async def run_text(system_prompt: str, user_prompt: str, *, max_iterations: int = 1) -> str:
    """Run a single-turn, tool-less text agent and return its text output."""
    from topspin_review.backend.agent.builder import TextParams, build

    built = build(TextParams(system_prompt=system_prompt, tools=[], rails=[], max_iterations=max_iterations, record=False))
    return _output_text(await run_agent(built.agent, user_prompt))


def on_tool_calls(started: Callable, finished: Callable, error: Callable) -> bool:
    """Subscribe to tool lifecycle events. Returns ``False`` if unavailable."""
    try:
        from openjiuwen.core.runner import Runner
        from openjiuwen.core.runner.callback.events import ToolCallEvents

        framework = Runner.callback_framework
        framework.on(ToolCallEvents.TOOL_CALL_STARTED)(started)
        framework.on(ToolCallEvents.TOOL_CALL_FINISHED)(finished)
        framework.on(ToolCallEvents.TOOL_CALL_ERROR)(error)
        return True
    except Exception:
        return False


def on_llm_output(observer: Callable) -> bool:
    """Subscribe to model-call output events. Returns ``False`` if unavailable."""
    try:
        from openjiuwen.core.runner import Runner
        from openjiuwen.core.runner.callback.events import LLMCallEvents

        Runner.callback_framework.on(LLMCallEvents.LLM_INVOKE_OUTPUT)(observer)
        return True
    except Exception:
        return False
