"""openjiuwen tool decoration.

Application code describes a tool with :class:`ToolSpec` — a plain callable plus its
name/description/input-params schema — and never touches the openjiuwen decorator.
The agents file calls :func:`decorate_all` to turn specs into framework tools, so
nothing about tooling decoration is known outside ``backend``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

try:  # openjiuwen is provided by the host environment
    from openjiuwen.core.foundation.tool import tool as _tool
except Exception:  # pragma: no cover

    def _tool(**_kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        return _wrap


@dataclass(frozen=True)
class ToolSpec:
    """A framework-neutral description of a callable tool."""

    func: Callable
    name: str
    description: str
    input_params: dict[str, Any]


def decorate(spec: ToolSpec) -> Any:
    """Turn a :class:`ToolSpec` into an openjiuwen tool."""
    return _tool(name=spec.name, description=spec.description, input_params=spec.input_params)(spec.func)


def decorate_all(specs: Iterable[ToolSpec] | None) -> list:
    """Decorate every spec (used by the agents file)."""
    return [decorate(spec) for spec in specs or ()]
