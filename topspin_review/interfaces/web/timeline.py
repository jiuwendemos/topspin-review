"""Presentation helpers that shape a report's telemetry into UI data.

These are pure functions over the report's ``usage`` block and observability
artifacts — they live with the UI that renders them, not in the backend.
"""

from __future__ import annotations

# Stage names emitted by the pipeline that own model calls, and which vision
# pass maps to which stage.
AGENT_STAGES = ("agent writing report", "agent: analyzing")
VISION_STAGE = {
    "overview": "vision: overview",
    "detail": "vision: detail",
    "reviewing image": "vision: reviewing image",
}


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
            label = cand.get("label") or ""
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
                nested = nested + [{"depth": 1, "kind": "overhead", "label": "agent overhead", "seconds": overhead}]
        else:
            nested = list(events.get(stage, []))
        for event in nested:
            event = dict(event)
            event["note"] = _event_note(event)
            rows.append(event)
    return rows
