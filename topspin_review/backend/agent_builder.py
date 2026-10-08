"""The agent builder: build a model-backed agent from a params object.

This is the single entry point for constructing agents. Callers pass one of the
:mod:`~topspin_review.backend.agent_builder_params` dataclasses; the builder
constructs/instruments the model, decorates tools, resolves rails, builds the
DeepAgent and creates the run recorder (observability). Models, tool decoration,
rails and observability all stay internal to ``backend``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from topspin_review.backend import observability
from topspin_review.backend import rails as backend_rails
from topspin_review.backend import settings as backend_settings
from topspin_review.backend import tools as backend_tools
from topspin_review.backend.agent_builder_params import Params, TextParams, VisionParams
from topspin_review.backend.models import make_text_model, make_vision_model


@dataclass
class BuildResult:
    """A built agent plus the run recorder the builder created/reused."""

    agent: Any
    recorder: Any = None


def build(params: Params) -> BuildResult:
    """Build the agent described by ``params``."""
    backend_settings.validate()
    recorder = _recorder(params)
    if isinstance(params, VisionParams):
        agent = _build_vision(params, recorder)
    else:
        agent = _build_text(params, recorder)
    return BuildResult(agent=agent, recorder=recorder)


def _recorder(params: Params) -> Any:
    if params.recorder is not None:
        return params.recorder
    if not params.record:
        return None
    recorder = observability.Recorder(media_dir=params.media_dir)
    recorder.install()
    return recorder


def _instrument(model: Any, recorder: Any, kind: str, label: str) -> Any:
    if recorder is None:
        return model
    return observability.attach(model, recorder.usage(kind), recorder.calls, label=label)


def _build_text(params: TextParams, recorder: Any) -> Any:
    from openjiuwen.harness import create_deep_agent

    model = _instrument(params.model or make_text_model(), recorder, "text", "report agent")
    return create_deep_agent(
        model=model,
        system_prompt=params.system_prompt,
        tools=backend_tools.make_tools(params.tools),
        rails=backend_rails.resolve(params.rails),
        enable_task_loop=False,
        max_iterations=params.max_iterations,
        workspace=params.workspace,
    )


def _build_vision(params: VisionParams, recorder: Any) -> Any:
    from openjiuwen.harness import create_deep_agent

    model = _instrument(params.model or make_vision_model(), recorder, "vision", "vision")
    return create_deep_agent(
        model=model,
        system_prompt=params.system_prompt,
        tools=[],
        rails=[backend_rails.sys_operation_rail()],
        enable_task_loop=False,
        max_iterations=params.max_iterations,
        workspace=params.workspace,
        enable_read_image_multimodal=True,
    )
