"""Applicative telemetry around a transport vision backend.

Providers are pure transport (see :mod:`topspin_review.backend.providers`). This
wrapper adds the application-side concerns — token/latency usage accounting, model
call-I/O tracing and event sequencing — and hands callers plain text, mirroring
what :func:`topspin_review.observability.attach` does for the text model.
"""

from __future__ import annotations

import time
from typing import Any

from topspin_review import observability
from topspin_review.backend import VisionBackend, get_backend


class RecordingVisionBackend:
    """Wrap a transport backend, record usage + call I/O, and return plain text."""

    def __init__(self, inner: VisionBackend, trace=None) -> None:
        self._inner = inner
        self._trace = trace
        self._usages: list[dict] = []

    async def complete(self, messages: list[dict[str, Any]], label: str = "") -> str:
        start = time.monotonic()
        seq = observability.next_seq()
        result = await self._inner.complete(messages, label)

        record = observability.extract_usage(result.raw)
        record["backend"] = result.backend
        record["label"] = label
        record["seconds"] = result.seconds
        if result.model and not record.get("model"):
            record["model"] = result.model
        self._usages.append(record)

        if self._trace is not None:
            self._trace.capture(
                f"vision: {label}" if label else "vision",
                messages,
                result.text,
                model=record.get("model", ""),
                usage=record,
                started=start,
                seq=seq,
            )
        return result.text

    def usage_summary(self) -> dict:
        return {"backend": getattr(self._inner, "name", ""), **observability.summarize(self._usages)}

    def usage_calls(self) -> list[dict]:
        return [dict(u) for u in self._usages]


def recording_backend(trace=None) -> RecordingVisionBackend:
    """The configured provider wrapped in a recording backend."""
    return RecordingVisionBackend(get_backend(), trace=trace)
