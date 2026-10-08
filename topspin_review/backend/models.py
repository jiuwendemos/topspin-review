"""Construct openjiuwen model clients for the text and vision models.

Kept separate from application settings: this is the only place that knows how a
model client is wired to a provider/endpoint.
"""

from __future__ import annotations

from topspin_review.backend import settings as backend_settings


def _make(model_name: str, temperature: float, timeout: int):
    from openjiuwen.core.foundation.llm import (
        Model,
        ModelClientConfig,
        ModelRequestConfig,
    )

    client = ModelClientConfig(
        client_provider=backend_settings.model_provider(),
        api_key=backend_settings.api_key(),
        api_base=backend_settings.api_base(),
        timeout=timeout,
        verify_ssl=backend_settings.verify_ssl(),
    )
    request = ModelRequestConfig(model=model_name, temperature=temperature, top_p=0.9)
    return Model(model_client_config=client, model_config=request)


def make_text_model():
    """Text model that writes the coaching report."""
    return _make(
        backend_settings.text_model_name(),
        backend_settings.text_temperature(),
        int(backend_settings.llm_timeout()),
    )


def make_vision_model():
    """Vision model that reads sampled video frames."""
    return _make(
        backend_settings.vision_model_name(),
        backend_settings.vision_temperature(),
        int(backend_settings.llm_timeout()),
    )
