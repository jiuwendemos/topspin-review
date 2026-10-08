"""Build an openjiuwen Model from params.

The only place that knows how a model is wired to a provider/endpoint.
"""

from __future__ import annotations

from typing import Any

from topspin_review.backend import settings as backend_settings
from topspin_review.backend.agent.models.params import ModelParams


def build_model(params: ModelParams) -> Any:
    """Construct an openjiuwen ``Model`` for the configured provider/endpoint."""
    from openjiuwen.core.foundation.llm import (
        Model,
        ModelClientConfig,
        ModelRequestConfig,
    )

    client = ModelClientConfig(
        client_provider=backend_settings.model_provider(),
        api_key=backend_settings.api_key(),
        api_base=backend_settings.api_base(),
        timeout=params.timeout,
        verify_ssl=backend_settings.verify_ssl(),
    )
    request = ModelRequestConfig(model=params.model_name, temperature=params.temperature, top_p=0.9)
    return Model(model_client_config=client, model_config=request)
