"""Backend layer: the single home for everything under-the-hood of the agentic system.

The application-facing surface is small and explicit:

- ``build`` + ``TextParams`` / ``VisionParams`` — construct agents.
- ``run_agent`` — run a built agent (starts the Runner if needed).
- ``ConfigError`` — raised when backend config is missing/invalid.
- ``configure_logging`` — route openjiuwen logging.

Internally, ``backend/`` is grouped by concern: ``agent/`` (builder, params,
models, rails, tools, runner) and ``telemetry/`` (usage, traces, recorder), with
``settings.py`` and ``logs.py`` at the root. Everything there is an implementation
detail, imported only inside ``backend``. No ``from openjiuwen...`` import exists
outside this package; no application module is imported by it.
"""

from topspin_review.backend.agent.builder import TextParams, VisionParams, build
from topspin_review.backend.agent.runner import run_agent
from topspin_review.backend.logs import configure as configure_logging
from topspin_review.backend.settings import ConfigError

__all__ = [
    "ConfigError",
    "TextParams",
    "VisionParams",
    "build",
    "configure_logging",
    "run_agent",
]
