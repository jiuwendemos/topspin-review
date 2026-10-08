"""The vision-provider registry and factory.

Each provider is one module in this package. To add one, implement
:class:`~topspin_review.backend.providers.base.VisionBackend`, add a module, and
register it in ``PROVIDERS``.
"""

from __future__ import annotations

from topspin_review.backend import settings as backend_settings
from topspin_review.backend.providers.base import VisionBackend
from topspin_review.backend.providers.mock import MockVisionBackend
from topspin_review.backend.providers.openai import OpenAIVisionBackend

PROVIDERS: dict[str, type] = {
    "openai": OpenAIVisionBackend,
    "mock": MockVisionBackend,
}


def get_backend(name: str | None = None) -> VisionBackend:
    """Return the configured vision provider (``VISION_BACKEND``), default ``openai``."""
    resolved = (name or backend_settings.vision_backend()).strip().lower()
    provider = PROVIDERS.get(resolved, OpenAIVisionBackend)
    return provider()
