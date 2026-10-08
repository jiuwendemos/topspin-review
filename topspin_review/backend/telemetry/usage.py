"""Token/latency usage extraction and accumulation."""

from __future__ import annotations

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


class UsageCollector:
    """Accumulates per-call usage records and summarizes them."""

    def __init__(self) -> None:
        self.usages: list[dict] = []

    def add(self, record: dict | None) -> None:
        if record:
            self.usages.append(record)

    def summary(self) -> dict:
        return summarize(self.usages)

    def records(self) -> list[dict]:
        return [dict(u) for u in self.usages]
