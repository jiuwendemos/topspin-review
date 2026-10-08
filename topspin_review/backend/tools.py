"""openjiuwen tool decoration (internal).

The agent builder passes plain callables through :func:`make_tools`; metadata
(name, description, input schema) is auto-extracted from the function name /
docstring / signature. Nothing here is part of the public `backend` API.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

try:  # openjiuwen is provided by the host environment
    from openjiuwen.core.foundation.tool import tool as _tool
except Exception:  # pragma: no cover

    def _tool(func=None, **_kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        return _wrap(func) if func is not None else _wrap


def make_tool(
    func: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
    input_params: dict[str, Any] | None = None,
) -> Any:
    """Expose a plain callable as an openjiuwen tool (auto-extracting metadata)."""
    if name is None and description is None and input_params is None:
        return _tool(func)
    return _tool(name=name, description=description, input_params=input_params)(func)


def make_tools(funcs: Iterable[Callable] | None) -> list:
    """Expose every callable as a tool (used by the agent builder)."""
    return [make_tool(func) for func in funcs or ()]
