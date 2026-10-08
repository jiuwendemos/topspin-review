"""The run Recorder and the model ``attach`` wrapper (builder-owned telemetry)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from topspin_review.backend.telemetry.traces import CallbackTrace, CallTrace, ToolTrace, next_seq
from topspin_review.backend.telemetry.usage import UsageCollector, extract_usage


def model_name(model: Any) -> str:
    """Best-effort name of the concrete model behind an openjiuwen ``Model``."""
    for attr in ("model_config", "config"):
        cfg = getattr(model, attr, None)
        for key in ("model_name", "model"):
            name = getattr(cfg, key, None)
            if name:
                return str(name)
    for key in ("model_name", "model"):
        name = getattr(model, key, None)
        if name:
            return str(name)
    return ""


def attach(
    model: Any,
    collector: UsageCollector | None = None,
    trace: CallTrace | None = None,
    label: str = "report agent",
) -> Any:
    """Wrap ``model.invoke`` to record token usage and (optionally) call I/O."""
    try:
        original = model.invoke
    except Exception:
        return model

    name = model_name(model)

    async def invoke(messages, *args, **kwargs):
        start = time.monotonic()
        seq = next_seq()
        result = await original(messages, *args, **kwargs)
        record = extract_usage(result)
        record["seconds"] = round(time.monotonic() - start, 2)
        if name:
            record["model"] = name
        if collector is not None:
            collector.add(record)
        if trace is not None:
            trace.capture(
                label,
                messages,
                getattr(result, "content", result),
                model=name,
                tools=kwargs.get("tools"),
                usage=record,
                started=start,
                seq=seq,
                requested=getattr(result, "tool_calls", None),
            )
        return result

    try:
        model.invoke = invoke
    except Exception:
        pass
    return model


class Recorder:
    """Run-scoped telemetry, owned by the agent builder.

    Holds the usage collectors (text/vision), the model call-I/O trace, the tool
    trace and the optional framework callback trace. Built by the builder when an
    agent is built with recording enabled; application code only reads it.
    """

    def __init__(self, media_dir: str | Path | None = None) -> None:
        from topspin_review.backend import settings as backend_settings

        self.calls = CallTrace(
            record_io=backend_settings.save_call_io(),
            media_dir=Path(media_dir) if media_dir is not None else None,
        )
        self.tools = ToolTrace()
        self.text = UsageCollector()
        self.vision = UsageCollector()
        self.callback = CallbackTrace() if backend_settings.trace_callbacks() else None
        self._installed = False

    def install(self) -> None:
        """Install the global tool/callback observers once per run."""
        if self._installed:
            return
        self.tools.install()
        if self.callback is not None:
            self.callback.install()
        self._installed = True

    def usage(self, kind: str = "text") -> UsageCollector:
        return self.vision if kind == "vision" else self.text

    def models(self) -> dict[str, str]:
        """The text/vision model + provider identifiers for the usage block."""
        from topspin_review.backend import settings as backend_settings

        return {
            "text": backend_settings.text_model_name(),
            "vision": backend_settings.vision_model_name(),
            "provider": backend_settings.model_provider(),
            "api_base": backend_settings.api_base(),
        }
