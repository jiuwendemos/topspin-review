"""OpenAI-compatible vision provider (the default).

Pure transport: invoke the model with retries/timeout and return a
:class:`VisionResult`. No telemetry, no application state.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from topspin_review.backend import models as backend_models
from topspin_review.backend import settings as backend_settings
from topspin_review.backend.providers.base import VisionResult


class OpenAIVisionBackend:
    """Vision model via the shared OpenAI-compatible client, with retries."""

    name = "openai"

    def __init__(self) -> None:
        self._model = backend_models.make_vision_model()

    async def complete(self, messages: list[dict[str, Any]], label: str = "") -> VisionResult:
        last: Exception | None = None
        retries = backend_settings.llm_retries()
        for attempt in range(retries + 1):
            start = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    self._model.invoke(messages), timeout=backend_settings.llm_timeout()
                )
                text = getattr(result, "content", str(result)) or ""
                return VisionResult(
                    text=text,
                    backend=self.name,
                    model=backend_settings.vision_model_name(),
                    seconds=round(time.monotonic() - start, 2),
                    raw=result,
                )
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < retries:
                    await asyncio.sleep(min(2**attempt, 8))
        raise last if last is not None else RuntimeError("vision call failed")
