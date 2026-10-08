"""Token/latency usage capture shared by the vision backend and the text agent."""

from __future__ import annotations

import base64
import itertools
import json
import threading
import time
from pathlib import Path
from typing import Any

# A process-wide monotonic sequence assigned when an event *starts*, so model
# and tool calls can be ordered exactly even when their timestamps tie.
_SEQ = itertools.count(1)
_SEQ_LOCK = threading.Lock()


def next_seq() -> int:
    with _SEQ_LOCK:
        return next(_SEQ)

_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

# Stage names emitted by the pipeline that own model calls, and which vision
# pass maps to which stage. Kept here so the UI never hard-codes them.
AGENT_STAGES = ("agent writing report", "agent: analyzing")
VISION_STAGE = {
    "overview": "vision: overview",
    "detail": "vision: detail",
    "reviewing image": "vision: reviewing image",
}

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


def summary_stats(usage: dict | None) -> dict:
    """Headline numbers across a report's ``usage`` block."""
    usage = usage or {}
    calls = list(usage.get("vision_calls") or []) + list(usage.get("text_calls") or [])
    context = sum(int(c.get("prompt_tokens", 0) or 0) for c in calls)
    output = sum(int(c.get("completion_tokens", 0) or 0) for c in calls)
    cached = sum(int(c.get("cached_tokens", 0) or 0) for c in calls)
    seconds = sum(float(c.get("seconds", 0) or 0) for c in calls)
    return {
        "calls": len(calls),
        "context_tokens": context,
        "output_tokens": output,
        "total_tokens": context + output,
        "cached_tokens": cached,
        "model_seconds": round(seconds, 1),
    }


def _event_note(event: dict) -> str:
    if event.get("kind") == "model":
        pin = int(event.get("prompt_tokens", 0) or 0)
        pout = int(event.get("completion_tokens", 0) or 0)
        cached = int(event.get("cached_tokens", 0) or 0)
        note = f"{pin} in / {pout} out"
        if cached:
            note += f" · {cached} cached"
        return note
    return ""


def build_timeline(
    usage: dict | None,
    timings: list[dict] | None,
    calls: list[dict] | None = None,
    tools: list[dict] | None = None,
) -> list[dict]:
    """A single chronological execution timeline.

    Every phase (stage) is followed, in order, by the events that happened in
    it — vision/report model calls and tool calls — interleaved by start time.
    Event rows carry their full detail (prompt/output or arguments/result) so a
    UI can render one narrative instead of separate lists.

    Falls back to ``usage`` counts when the detailed ``calls`` list is absent.
    """
    usage = usage or {}
    events: dict[str, list[dict]] = {}

    # Token counts live in ``usage`` (always present); the detailed ``calls``
    # list adds prompt/output text, model name and start time. Merge detail onto
    # the usage records (or use the detail alone if usage is missing).
    detail = [dict(c) for c in (calls or [])]
    claimed = [False] * len(detail)

    def _take(norm: str) -> dict | None:
        for idx, cand in enumerate(detail):
            if claimed[idx]:
                continue
            label = (cand.get("label") or "")
            cand_norm = label.split(":", 1)[1].strip().lower() if label.startswith("vision:") else label.lower()
            if cand_norm == norm or norm in cand_norm or cand_norm in norm:
                claimed[idx] = True
                return cand
        return None

    def _merge(base: dict, extra: dict | None) -> dict:
        merged = dict(base)
        if not extra:
            return merged
        for key, value in extra.items():
            if value is None or value == "" or value == []:
                continue
            merged[key] = value
        return merged

    vision_usage = usage.get("vision_calls") or []
    text_usage = usage.get("text_calls") or []
    if vision_usage or text_usage:
        for call in vision_usage:
            label = str(call.get("label") or "")
            stage = VISION_STAGE.get(label, "__agent__")
            events.setdefault(stage, []).append(
                {"depth": 1, "kind": "model", "label": label, **_merge(call, _take(label.lower()))}
            )
        for i, call in enumerate(text_usage, start=1):
            events.setdefault("__agent__", []).append(
                {"depth": 1, "kind": "model", "label": f"model call {i}", **_merge(call, _take("report agent"))}
            )
    else:
        for call in detail:
            label = str(call.get("label") or "")
            key = label.split(":", 1)[1].strip() if label.startswith("vision:") else label
            stage = VISION_STAGE.get(key, "__agent__")
            events.setdefault(stage, []).append({"depth": 1, "kind": "model", "label": label, **call})

    for tool in tools or []:
        events.setdefault("__agent__", []).append({"depth": 1, "kind": "tool", **tool})

    for items in events.values():
        items.sort(key=lambda e: (float(e.get("started", 0) or 0), int(e.get("seq", 0) or 0)))

    # Attribute each executed tool to the model call that requested it (the
    # model call immediately before it in the agent's ordered events).
    last_model: dict | None = None
    for event in events.get("__agent__", []):
        if event.get("kind") == "model":
            event.setdefault("called", [])
            last_model = event
        elif event.get("kind") == "tool" and last_model is not None:
            last_model["called"].append({"name": event.get("name", ""), "arguments": event.get("arguments", "")})

    rows: list[dict] = []
    for entry in timings or []:
        stage = entry.get("stage", "")
        secs = float(entry.get("seconds", 0) or 0)
        rows.append({"depth": 0, "kind": "stage", "label": stage, "seconds": secs})
        if stage in AGENT_STAGES:
            nested = list(events.get("__agent__", []))
            used = sum(float(e.get("seconds", 0) or 0) for e in nested)
            overhead = round(secs - used, 2)
            if overhead > 0.05:
                nested = nested + [
                    {"depth": 1, "kind": "overhead", "label": "agent overhead", "seconds": overhead}
                ]
        else:
            nested = list(events.get(stage, []))
        for event in nested:
            event = dict(event)
            event["note"] = _event_note(event)
            rows.append(event)
    return rows


def _clip(text: Any, limit: int | None = 8000) -> str:
    text = str(text or "")
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n… (+{len(text) - limit} more chars)"


def _content_text(content: Any) -> str:
    """Flatten a message content (str or OpenAI-style parts) to text, images summarised."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        images = 0
        for part in content:
            if isinstance(part, dict):
                kind = part.get("type")
                if kind == "text":
                    parts.append(str(part.get("text", "")))
                elif kind in ("image_url", "image"):
                    images += 1
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        text = "\n".join(parts)
        if images:
            label = "[image]" if images == 1 else f"[image ×{images}]"
            text = (text + "\n" if text else "") + label
        return text
    return str(content or "")


def _iter_image_urls(messages: Any) -> list[str]:
    urls: list[str] = []
    if messages is None:
        return urls
    if isinstance(messages, (str, dict)) or not hasattr(messages, "__iter__"):
        messages = [messages]
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in ("image_url", "image"):
                continue
            ref = part.get("image_url") or part.get("image") or {}
            url = ref.get("url") if isinstance(ref, dict) else ref
            if isinstance(url, str) and url.startswith("data:image") and ";base64," in url:
                urls.append(url)
    return urls


def _tool_call_names(raw: Any) -> list[str]:
    names = []
    for call in raw or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name:
            names.append(str(name))
    return names


def _message_text(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        role = str(message.get("role", "?"))
        text = _content_text(message.get("content", ""))
        names = _tool_call_names(message.get("tool_calls"))
    else:
        role = getattr(message, "role", None) or type(message).__name__
        text = _content_text(getattr(message, "content", ""))
        names = _tool_call_names(getattr(message, "tool_calls", None))
    if names:
        text = (text + "\n" if text else "") + f"[requested tools: {', '.join(names)}]"
    return str(role), text


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

    def __init__(self, record_io: bool = True, media_dir: Path | None = None) -> None:
        self.calls: list[dict] = []
        self.record_io = record_io
        self.media_dir = media_dir

    def _save_images(self, stem: str, messages: Any) -> list[str]:
        if self.media_dir is None:
            return []
        paths: list[str] = []
        try:
            self.media_dir.mkdir(parents=True, exist_ok=True)
            for i, url in enumerate(_iter_image_urls(messages), start=1):
                header, _, payload = url.partition(";base64,")
                ext = "png" if "png" in header else ("jpg" if ("jpeg" in header or "jpg" in header) else "bin")
                path = self.media_dir / f"{stem}_img{i}.{ext}"
                path.write_bytes(base64.b64decode(payload))
                paths.append(str(path))
        except Exception:
            return paths
        return paths

    def capture(
        self,
        label: str,
        messages: Any,
        output: Any,
        model: str = "",
        tools: Any = None,
        usage: dict | None = None,
        started: float = 0.0,
        seq: int = 0,
        requested: Any = None,
    ) -> None:
        try:
            entry = {
                "label": label,
                "model": model,
                "started": started,
                "seq": seq,
                "input": sanitize_messages(messages, limit=None) if self.record_io else [],
                "output": _clip(output, None) if self.record_io else "",
            }
            if tools:
                entry["tools"] = [str(getattr(t, "name", t)) for t in tools]
            if requested:
                entry["requested"] = _tool_calls(requested)
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens"):
                if usage and usage.get(key) is not None:
                    entry[key] = int(usage.get(key) or 0)
            if self.record_io and self.media_dir is not None:
                images = self._save_images(f"call{seq or len(self.calls) + 1}", messages)
                if images:
                    entry["images"] = images
            self.calls.append(entry)
        except Exception:
            pass


def _as_text(value: Any, limit: int | None = 4000) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            return _clip(json.dumps(value, ensure_ascii=False, default=str, indent=2), limit)
        except Exception:
            pass
    return _clip(value, limit)


def _tool_calls(raw: Any) -> list[dict]:
    out = []
    for call in raw or []:
        if isinstance(call, dict):
            name, arguments = call.get("name"), call.get("arguments")
        else:
            name, arguments = getattr(call, "name", None), getattr(call, "arguments", None)
        entry: dict = {"name": str(name or "")}
        if arguments:
            entry["arguments"] = _clip(arguments, 2000)
        out.append(entry)
    return out


_INTERNAL_ARGS = {"session", "self", "cls", "ctx", "context"}


def _tool_arguments(inputs: Any) -> str:
    try:
        args, kwargs = inputs
        kwargs = {
            key: value
            for key, value in (kwargs or {}).items()
            if key not in _INTERNAL_ARGS and isinstance(value, (str, int, float, bool, list, dict, type(None)))
        }
        if not args:
            payload: Any = kwargs
        elif not kwargs:
            payload = list(args)
        else:
            payload = {"args": [a for a in args if isinstance(a, (str, int, float, bool, list, dict))], "kwargs": kwargs}
        return _as_text(payload, 4000)
    except Exception:
        return _clip(str(inputs), 4000)


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
        from topspin_review.backend.runner import on_tool_calls

        async def _started(*args: Any, **kwargs: Any) -> None:
            key = kwargs.get("tool_id") or kwargs.get("tool_name")
            self._open[key] = {
                "name": str(kwargs.get("tool_name") or ""),
                "arguments": _tool_arguments(kwargs.get("inputs")),
                "started": time.monotonic(),
                "seq": next_seq(),
            }

        async def _finished(*args: Any, **kwargs: Any) -> None:
            key = kwargs.get("tool_id") or kwargs.get("tool_name")
            rec = self._open.pop(key, None) or {
                "name": str(kwargs.get("tool_name") or ""),
                "arguments": "",
                "started": time.monotonic(),
                "seq": next_seq(),
            }
            rec["seconds"] = round(time.monotonic() - rec.get("started", time.monotonic()), 2)
            rec["result"] = _as_text(kwargs.get("result"), 4000)
            self.calls.append(rec)

        async def _error(*args: Any, **kwargs: Any) -> None:
            key = kwargs.get("tool_id") or kwargs.get("tool_name")
            rec = self._open.pop(key, None)
            if rec is None:
                return
            rec["seconds"] = round(time.monotonic() - rec.get("started", time.monotonic()), 2)
            rec["error"] = str(kwargs.get("error"))
            self.calls.append(rec)

        return on_tool_calls(_started, _finished, _error)

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
        from topspin_review.backend.runner import on_llm_output

        async def _observe(*args: Any, **kwargs: Any) -> None:
            for obj in list(args) + list(kwargs.values()):
                record = extract_usage(obj)
                if record:
                    self.usages.append(record)
                    return

        return on_llm_output(_observe)

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
