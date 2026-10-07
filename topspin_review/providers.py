"""Pluggable vision backends behind one small interface.

The default backend calls the OpenAI-compatible vision model, with retries, a
timeout and per-call usage tracking. Swap it with ``VISION_BACKEND=mock`` for
offline tests, or add a new class here (a local model, or a provider that accepts
real video) without touching the pipeline.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from topspin_review import config, observability


class VisionBackend(Protocol):
    async def complete(self, messages: list[dict[str, Any]], label: str = "") -> str: ...
    def usage_summary(self) -> dict: ...
    def usage_calls(self) -> list[dict]: ...


class OpenAIVisionBackend:
    """Vision model via the shared OpenAI-compatible client, with retries."""

    def __init__(self, trace=None) -> None:
        self._model = config.make_vision_model()
        self.usages: list[dict] = []
        self._trace = trace

    async def complete(self, messages: list[dict[str, Any]], label: str = "") -> str:
        last: Exception | None = None
        for attempt in range(config.llm_retries() + 1):
            start = time.monotonic()
            try:
                result = await asyncio.wait_for(self._model.invoke(messages), timeout=config.llm_timeout())
                record = {
                    "backend": "openai",
                    "label": label,
                    "model": observability.model_name(self._model) or config.vision_model_name(),
                    "seconds": round(time.monotonic() - start, 2),
                }
                record.update(observability.extract_usage(result))
                self.usages.append(record)
                text = getattr(result, "content", str(result)) or ""
                if self._trace is not None:
                    self._trace.capture(
                        f"vision: {label}" if label else "vision",
                        messages,
                        text,
                        model=record["model"],
                    )
                return text
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < config.llm_retries():
                    await asyncio.sleep(min(2**attempt, 8))
        raise last if last is not None else RuntimeError("vision call failed")

    def usage_summary(self) -> dict:
        return {"backend": "openai", **observability.summarize(self.usages)}

    def usage_calls(self) -> list[dict]:
        return [dict(u) for u in self.usages]


class MockVisionBackend:
    """Deterministic offline backend for tests; no network calls."""

    async def complete(self, messages: list[dict[str, Any]], label: str = "") -> str:
        text = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        if "attentive_windows" in text:
            return (
                '{"overall": "mock overview", '
                '"attentive_windows": [{"start": 1.0, "end": 3.0, "why": "peak motion"}], '
                '"limitations": ["mock backend"]}'
            )
        return (
            '{"observations": "mock observation at t=2.0s", '
            '"signals": [{"signal": "mock signal", "evidence_times": [2.0], "confidence": "low"}], '
            '"limitations": ["mock backend"]}'
        )

    def usage_summary(self) -> dict:
        return {"backend": "mock", "calls": 0, "seconds": 0.0}

    def usage_calls(self) -> list[dict]:
        return []


def get_backend(name: str | None = None, trace=None) -> VisionBackend:
    resolved = (name or config.vision_backend()).strip().lower()
    if resolved == "mock":
        return MockVisionBackend()
    return OpenAIVisionBackend(trace=trace)
