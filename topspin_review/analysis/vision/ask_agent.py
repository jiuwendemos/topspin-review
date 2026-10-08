"""Ask the vision agent and parse its JSON reply."""

from __future__ import annotations

import json
import re
from typing import Any

from topspin_review.backend import run_agent


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a model reply."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _output_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, dict):
        for key in ("output", "content", "answer", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        return str(result)
    return getattr(result, "content", None) or str(result)


async def ask(agent: Any, prompt: str) -> dict:
    """Run the vision agent and parse its JSON reply."""
    return extract_json(_output_text(await run_agent(agent, prompt)))
