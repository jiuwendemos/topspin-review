"""Text agent builder (report writer, single-shot verify/Q&A, …)."""

from __future__ import annotations

from typing import Any

from topspin_review.backend import settings as backend_settings
from topspin_review.backend.agent import rails as backend_rails
from topspin_review.backend.agent import tools as backend_tools
from topspin_review.backend.agent.builder.base import AgentBuilder
from topspin_review.backend.agent.builder.params import TextParams
from topspin_review.backend.agent.models import ModelParams, build_model


class TextBuilder(AgentBuilder):
    """Builds a text DeepAgent from :class:`TextParams`."""

    params: TextParams

    def build(self) -> Any:
        from openjiuwen.harness import create_deep_agent

        model = self.instrument(self._model(), "text", "report agent")
        return create_deep_agent(
            model=model,
            system_prompt=self.params.system_prompt,
            tools=backend_tools.make_tools(self.params.tools),
            rails=backend_rails.resolve(self.params.rails),
            enable_task_loop=False,
            max_iterations=self.params.max_iterations,
            workspace=self.params.workspace,
        )

    @staticmethod
    def _model() -> Any:
        return build_model(
            ModelParams(
                model_name=backend_settings.text_model_name(),
                temperature=backend_settings.text_temperature(),
                timeout=int(backend_settings.llm_timeout()),
            )
        )
