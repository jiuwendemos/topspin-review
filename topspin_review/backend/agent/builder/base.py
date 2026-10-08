"""Base agent builder: owns the run recorder and model instrumentation."""

from __future__ import annotations

from typing import Any

from topspin_review.backend.agent.builder.params import Params
from topspin_review.backend.telemetry import recorder as telemetry


class AgentBuilder:
    """Base for the text/vision builders.

    Creates (or reuses) the run recorder and instruments the model with it; the
    concrete builder only constructs the DeepAgent.
    """

    def __init__(self, params: Params) -> None:
        self.params = params
        self.recorder = self._make_recorder()

    def _make_recorder(self) -> Any:
        params = self.params
        if params.recorder is not None:
            return params.recorder
        if not params.record:
            return None
        recorder = telemetry.Recorder(media_dir=params.media_dir)
        recorder.install()
        return recorder

    def instrument(self, model: Any, kind: str, label: str) -> Any:
        """Attach usage/trace capture to a model when a recorder is present."""
        if self.recorder is None:
            return model
        return telemetry.attach(model, self.recorder.usage(kind), self.recorder.calls, label=label)

    def build(self) -> Any:
        """Construct the agent (implemented by subclasses)."""
        raise NotImplementedError
