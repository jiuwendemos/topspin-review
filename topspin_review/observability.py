"""Token/latency usage capture shared by the vision backend and the text agent."""

from __future__ import annotations

from typing import Any

_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

_ALIASES = {
    "prompt_tokens": ("prompt_tokens", "input_tokens"),
    "completion_tokens": ("completion_tokens", "output_tokens"),
    "total_tokens": ("total_tokens",),
}


def extract_usage(result: Any) -> dict:
    """Best-effort token extraction from a model result.

    openjiuwen exposes ``usage_metadata`` on the AssistantMessage; OpenAI-style
    clients use ``usage``. Either attribute, or the message's own ``metadata``,
    is accepted.
    """
    usage = (
        getattr(result, "usage_metadata", None)
        or getattr(result, "usage", None)
        or getattr(result, "token_usage", None)
    )
    if usage is None:
        return {}
    if isinstance(usage, dict):
        out = {}
        for key, names in _ALIASES.items():
            for name in names:
                if name in usage and usage[name] is not None:
                    out[key] = usage[name]
                    break
    else:
        out = {}
        for key, names in _ALIASES.items():
            for name in names:
                value = getattr(usage, name, None)
                if value is not None:
                    out[key] = value
                    break
    if "total_tokens" not in out and ("prompt_tokens" in out or "completion_tokens" in out):
        out["total_tokens"] = int(out.get("prompt_tokens", 0)) + int(out.get("completion_tokens", 0))
    return out


def summarize(usages: list[dict]) -> dict:
    totals = {k: sum(int(u.get(k, 0) or 0) for u in usages) for k in _KEYS}
    seconds = round(sum(float(u.get("seconds", 0.0) or 0.0) for u in usages), 2)
    return {"calls": len(usages), "seconds": seconds, **totals}


class UsageCollector:
    def __init__(self) -> None:
        self.usages: list[dict] = []

    def add(self, record: dict | None) -> None:
        if record:
            self.usages.append(record)

    def summary(self) -> dict:
        return summarize(self.usages)


def attach(model: Any, collector: UsageCollector) -> Any:
    """Wrap ``model.invoke`` to record token usage (best-effort, never fatal)."""
    try:
        original = model.invoke
    except Exception:
        return model

    async def invoke(messages, *args, **kwargs):
        result = await original(messages, *args, **kwargs)
        collector.add(extract_usage(result))
        return result

    try:
        model.invoke = invoke
    except Exception:
        pass
    return model
