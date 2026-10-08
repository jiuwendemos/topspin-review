"""Result of building an agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BuildResult:
    """A built agent plus the run recorder the builder created/reused."""

    agent: Any
    recorder: Any = None
