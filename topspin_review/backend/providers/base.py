"""The vision-provider interface and transport result type.

Providers are pure transport: they call the model and return a
:class:`VisionResult`. Usage accounting, call tracing and event sequencing are
*applicative* concerns and live in ``analysis/vision_telemetry.py``, so this
package depends on nothing above it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class VisionResult:
    """Provider-agnostic result of one vision call."""

    text: str
    backend: str = ""
    model: str = ""
    seconds: float = 0.0
    raw: Any = None  # provider response, handed upstream for usage extraction


class VisionBackend(Protocol):
    name: str

    async def complete(self, messages: list[dict[str, Any]], label: str = "") -> VisionResult: ...
