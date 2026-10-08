"""Backend layer: the single home for everything under-the-hood of the agentic system.

The application-facing surface is small and explicit:

- ``build`` + ``TextParams`` / ``VisionParams`` — construct agents.
- ``run_agent`` — run a built agent (starts the Runner if needed).
- ``ConfigError`` — raised when backend config is missing/invalid.
- ``configure_logging`` — route openjiuwen logging.
- ``make_tool`` — publish a callable as a tool (e.g. the MCP interface).

Everything else in ``backend/`` (settings, models, rails, observability, tools
internals, runner, logs, agent_builder) is an implementation detail and is only
imported inside ``backend``. No ``from openjiuwen...`` import exists outside this
package; no application module is imported by it.
"""

from topspin_review.backend.agent_builder import build
from topspin_review.backend.agent_builder_params import TextParams, VisionParams
from topspin_review.backend.logs import configure as configure_logging
from topspin_review.backend.runner import run_agent
from topspin_review.backend.settings import ConfigError
from topspin_review.backend.tools import make_tool

__all__ = [
    "ConfigError",
    "TextParams",
    "VisionParams",
    "build",
    "configure_logging",
    "make_tool",
    "run_agent",
]
