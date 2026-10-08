"""Vision providers — import from this package, never a concrete module.

    from topspin_review.backend.providers import get_backend
"""

from topspin_review.backend.providers.base import VisionBackend, VisionResult
from topspin_review.backend.providers.mock import MockVisionBackend
from topspin_review.backend.providers.openai import OpenAIVisionBackend
from topspin_review.backend.providers.registry import PROVIDERS, get_backend

__all__ = [
    "PROVIDERS",
    "MockVisionBackend",
    "OpenAIVisionBackend",
    "VisionBackend",
    "VisionResult",
    "get_backend",
]
