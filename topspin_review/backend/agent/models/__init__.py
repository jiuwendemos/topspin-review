"""Model construction — a generic openjiuwen model builder.

Import from this package (the folder) rather than its submodules.
"""

from topspin_review.backend.agent.models.builder import build_model
from topspin_review.backend.agent.models.params import ModelParams

__all__ = ["ModelParams", "build_model"]
