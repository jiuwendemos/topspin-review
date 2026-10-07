"""Topspin Review — analyze a racket-sport session video into coaching insights."""

from .storage import runtime

# Configure logging (→ runtime/logs) as early as possible so no openjiuwen import
# can write its default ./logs/ into the project root. Idempotent.
runtime.setup()

__version__ = "0.1.0"
