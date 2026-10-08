"""The analysis-strategy contract.

Every strategy implements one async entrypoint with the same signature, so
callers can obtain *a* strategy and run it without caring which one it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class AnalyzeFn(Protocol):
    """The single entrypoint every analysis strategy implements."""

    async def __call__(
        self,
        video_path: str,
        region_box: tuple[float, float, float, float] | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Strategy:
    """A named analysis strategy."""

    name: str
    description: str
    analyze: AnalyzeFn
