"""Deterministic offline vision provider (no network calls; used by tests)."""

from __future__ import annotations

from typing import Any

from topspin_review.backend.providers.base import VisionResult


class MockVisionBackend:
    """Deterministic offline backend for tests; no network calls."""

    name = "mock"

    async def complete(self, messages: list[dict[str, Any]], label: str = "") -> VisionResult:
        text = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        if "attentive_windows" in text:
            return VisionResult(
                text=(
                    '{"overall": "mock overview", '
                    '"attentive_windows": [{"start": 1.0, "end": 3.0, "why": "peak motion"}], '
                    '"limitations": ["mock backend"]}'
                ),
                backend=self.name,
            )
        return VisionResult(
            text=(
                '{"observations": "mock observation at t=2.0s", '
                '"signals": [{"signal": "mock signal", "evidence_times": [2.0], "confidence": "low"}], '
                '"limitations": ["mock backend"]}'
            ),
            backend=self.name,
        )
