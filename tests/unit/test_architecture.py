"""Enforce the layered dependency direction (architecture as a test).

Dependencies may only point inward:
    interfaces -> analysis -> backend / perception -> domain
with ``observability``, ``storage`` and ``config`` as neutral leaves.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "topspin_review"

# Layers that are single modules at the package root (not sub-packages).
TOP_LAYERS = {"observability", "reporting"}

# For each layer, the app layers it must NOT import.
FORBIDDEN: dict[str, set[str]] = {
    "domain": {"observability", "perception", "analysis", "backend", "storage", "reporting", "interfaces"},
    "observability": {"domain", "perception", "analysis", "storage", "reporting", "interfaces"},
    "perception": {"analysis", "backend", "interfaces"},
    "backend": {"domain", "perception", "analysis", "reporting", "interfaces"},
    "storage": {"domain", "perception", "analysis", "backend", "reporting", "interfaces"},
    "reporting": {"observability", "perception", "analysis", "backend", "interfaces"},
    "analysis": {"interfaces"},
}


def _layer_of(path: Path) -> str | None:
    rel = path.relative_to(PKG)
    if len(rel.parts) == 1:
        return rel.stem if rel.stem in TOP_LAYERS else None
    return rel.parts[0]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_layer_dependencies_point_inward():
    violations: list[str] = []
    for path in PKG.rglob("*.py"):
        layer = _layer_of(path)
        forbidden = FORBIDDEN.get(layer or "")
        if not forbidden:
            continue
        for name in _imports(path):
            for other in forbidden:
                if name == f"topspin_review.{other}" or name.startswith(f"topspin_review.{other}."):
                    violations.append(f"{path.relative_to(PKG)}: {layer} -> {other} ({name})")
    assert not violations, "layer violations:\n" + "\n".join(violations)


def test_domain_is_pure():
    third_party = ("numpy", "PIL", "imageio", "openjiuwen", "streamlit", "fastapi", "mcp")
    violations: list[str] = []
    for path in (PKG / "domain").rglob("*.py"):
        for name in _imports(path):
            root = name.split(".")[0]
            if root in third_party:
                violations.append(f"{path.name}: imports {name}")
    assert not violations, "domain must stay dependency-free:\n" + "\n".join(violations)
