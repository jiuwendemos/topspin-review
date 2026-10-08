"""Vision agent builder (multimodal DeepAgent that reads images via ``read_file``)."""

from __future__ import annotations

from typing import Any

from topspin_review.backend import settings as backend_settings
from topspin_review.backend.agent import rails as backend_rails
from topspin_review.backend.agent.builder.base import AgentBuilder
from topspin_review.backend.agent.builder.params import VisionParams
from topspin_review.backend.agent.models import ModelParams, build_model


class VisionBuilder(AgentBuilder):
    """Builds the vision DeepAgent from :class:`VisionParams`."""

    params: VisionParams

    def build(self) -> Any:
        from openjiuwen.harness import create_deep_agent

        model = self.instrument(self._model(), "vision", "vision")
        return create_deep_agent(
            model=model,
            system_prompt=self.params.system_prompt,
            tools=[],
            rails=[backend_rails.sys_operation_rail()],
            enable_task_loop=False,
            max_iterations=self.params.max_iterations,
            workspace=self.params.workspace,
            enable_read_image_multimodal=True,
        )

    @staticmethod
    def _model() -> Any:
        return build_model(
            ModelParams(
                model_name=backend_settings.vision_model_name(),
                temperature=backend_settings.vision_temperature(),
                timeout=int(backend_settings.llm_timeout()),
            )
        )
