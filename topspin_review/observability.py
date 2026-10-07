"""Token/latency usage capture shared by the vision backend and the text agent."""

from __future__ import annotations

import time
from typing import Any

_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

_ALIASES = {
    "prompt_tokens": ("prompt_tokens", "input_tokens"),
    "completion_tokens": ("completion_tokens", "output_tokens"),
    "total_tokens": ("total_tokens",),
}


def _get(usage: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if value is not None:
            return value
    return None


def extract_usage(result: Any) -> dict:
    """Best-effort token extraction from a model result.

    openjiuwen exposes ``usage_metadata`` on the AssistantMessage; OpenAI-style
    clients use ``usage``. Cache counters (DeepSeek ``prompt_cache_hit_tokens``,
    OpenAI ``prompt_tokens_details.cached_tokens``, Anthropic cache fields) are
    captured too.
    """
    usage = (
        getattr(result, "usage_metadata", None)
        or getattr(result, "usage", None)
        or getattr(result, "token_usage", None)
    )
    if usage is None:
        return {}
    out: dict = {}
    for key, names in _ALIASES.items():
        value = _get(usage, names)
        if value is not None:
            out[key] = value
    if "total_tokens" not in out and ("prompt_tokens" in out or "completion_tokens" in out):
        out["total_tokens"] = int(out.get("prompt_tokens", 0)) + int(out.get("completion_tokens", 0))

    cached = _get(usage, ("prompt_cache_hit_tokens", "cache_read_input_tokens", "cached_tokens"))
    if cached is None:
        details = _get(usage, ("prompt_tokens_details", "prompt_token_details"))
        if details is not None:
            cached = _get(details, ("cached_tokens",))
    if cached is not None:
        out["cached_tokens"] = int(cached)
    miss = _get(usage, ("prompt_cache_miss_tokens",))
    if miss is not None:
        out["cache_miss_tokens"] = int(miss)
    created = _get(usage, ("cache_creation_input_tokens",))
    if created is not None:
        out["cache_creation_tokens"] = int(created)

    model = getattr(result, "model", None) or getattr(result, "model_name", None) or _get(usage, ("model",))
    if model:
        out["model"] = str(model)
    return out


def summarize(usages: list[dict]) -> dict:
    totals = {k: sum(int(u.get(k, 0) or 0) for u in usages) for k in _KEYS}
    seconds = round(sum(float(u.get("seconds", 0.0) or 0.0) for u in usages), 2)
    cached = sum(int(u.get("cached_tokens", 0) or 0) for u in usages)
    return {"calls": len(usages), "seconds": seconds, "cached_tokens": cached, **totals}


def _clip(text: Any, limit: int | None = 8000) -> str:
    text = str(text or "")
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n… (+{len(text) - limit} more chars)"


def _content_text(content: Any) -> str:
    """Flatten a message content (str or OpenAI-style parts) to text, images elided."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                kind = part.get("type")
                if kind == "text":
                    parts.append(str(part.get("text", "")))
                elif kind in ("image_url", "image"):
                    parts.append("[image]")
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content or "")


def _message_text(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        role = str(message.get("role", "?"))
        return role, _content_text(message.get("content", ""))
    role = getattr(message, "role", None) or type(message).__name__
    return str(role), _content_text(getattr(message, "content", ""))


def sanitize_messages(messages: Any, limit: int | None = 8000) -> list[dict]:
    """Messages as ``[{role, content}]``; images elided, text clipped only if ``limit`` set."""
    if messages is None:
        return []
    if isinstance(messages, (str, dict)) or not hasattr(messages, "__iter__"):
        messages = [messages]
    out = []
    for message in messages:
        role, text = _message_text(message)
        out.append({"role": role, "content": _clip(text, limit)})
    return out


class CallTrace:
    """Records the input messages and output text of every model call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def capture(self, label: str, messages: Any, output: Any, model: str = "", tools: Any = None) -> None:
        try:
            entry = {
                "label": label,
                "model": model,
                "input": sanitize_messages(messages, limit=None),
                "output": _clip(output, None),
            }
            if tools:
                entry["tools"] = [str(getattr(t, "name", t)) for t in tools]
            self.calls.append(entry)
        except Exception:
            pass


class ToolTrace:
    """Records tool calls (name, arguments, result, duration) via callback events.

    openjiuwen's tool lifecycle fires ``TOOL_CALL_STARTED``/``TOOL_CALL_FINISHED``
    (and ``TOOL_CALL_ERROR``) with ``tool_name``/``tool_id``/``inputs``/``result``.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._open: dict[Any, dict] = {}

    def install(self) -> bool:
        """Register the observers. Returns ``False`` if unavailable (never raises)."""
        try:
            from openjiuwen.core.runner import Runner
            from openjiuwen.core.runner.callback.events import ToolCallEvents

            framework = Runner.callback_framework

            async def _started(*args: Any, **kwargs: Any) -> None:
                key = kwargs.get("tool_id") or kwargs.get("tool_name")
                self._open[key] = {
                    "name": str(kwargs.get("tool_name") or ""),
                    "arguments": _clip(kwargs.get("inputs"), 4000),
                    "start": time.monotonic(),
                }

            async def _finished(*args: Any, **kwargs: Any) -> None:
                key = kwargs.get("tool_id") or kwargs.get("tool_name")
                rec = self._open.pop(key, None) or {
                    "name": str(kwargs.get("tool_name") or ""),
                    "arguments": "",
                    "start": time.monotonic(),
                }
                rec["seconds"] = round(time.monotonic() - rec.pop("start", time.monotonic()), 2)
                rec["result"] = _clip(kwargs.get("result"), 4000)
                self.calls.append(rec)

            async def _error(*args: Any, **kwargs: Any) -> None:
                key = kwargs.get("tool_id") or kwargs.get("tool_name")
                rec = self._open.pop(key, None)
                if rec is None:
                    return
                rec["seconds"] = round(time.monotonic() - rec.pop("start", time.monotonic()), 2)
                rec["error"] = str(kwargs.get("error"))
                self.calls.append(rec)

            framework.on(ToolCallEvents.TOOL_CALL_STARTED)(_started)
            framework.on(ToolCallEvents.TOOL_CALL_FINISHED)(_finished)
            framework.on(ToolCallEvents.TOOL_CALL_ERROR)(_error)
            return True
        except Exception:
            return False

    def records(self) -> list[dict]:
        return [dict(c) for c in self.calls]


class UsageCollector:
    def __init__(self) -> None:
        self.usages: list[dict] = []

    def add(self, record: dict | None) -> None:
        if record:
            self.usages.append(record)

    def summary(self) -> dict:
        return summarize(self.usages)

    def records(self) -> list[dict]:
        return [dict(u) for u in self.usages]


class CallbackTrace:
    """Capture usage for every model call via ``Runner.callback_framework``.

    This is a lighter-weight, framework-native alternative/supplement to
    :func:`attach`; it observes the global ``LLM_INVOKE_OUTPUT`` event so both
    vision and text calls are covered without wrapping each model.
    """

    def __init__(self) -> None:
        self.usages: list[dict] = []

    def install(self) -> bool:
        """Register the observer. Returns ``False`` if unavailable (never raises)."""
        try:
            from openjiuwen.core.runner import Runner
            from openjiuwen.core.runner.callback.events import LLMCallEvents

            async def _observe(*args: Any, **kwargs: Any) -> None:
                for obj in list(args) + list(kwargs.values()):
                    record = extract_usage(obj)
                    if record:
                        self.usages.append(record)
                        return

            Runner.callback_framework.on(LLMCallEvents.LLM_INVOKE_OUTPUT)(_observe)
            return True
        except Exception:
            return False

    def summary(self) -> dict:
        return summarize(self.usages)


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


def attach(model: Any, collector: UsageCollector, trace: CallTrace | None = None) -> Any:
    """Wrap ``model.invoke`` to record token usage and (optionally) call I/O."""
    try:
        original = model.invoke
    except Exception:
        return model

    name = model_name(model)

    async def invoke(messages, *args, **kwargs):
        start = time.monotonic()
        result = await original(messages, *args, **kwargs)
        record = extract_usage(result)
        record["seconds"] = round(time.monotonic() - start, 2)
        if name:
            record["model"] = name
        collector.add(record)
        if trace is not None:
            trace.capture(
                "report agent",
                messages,
                getattr(result, "content", result),
                model=name,
                tools=kwargs.get("tools"),
            )
        return result

    try:
        model.invoke = invoke
    except Exception:
        pass
    return model
