"""Enforce the layered dependency direction (architecture as a test).

Rules (dependencies may only point inward):
- ``domain`` imports no other app layer and no third-party libraries.
- ``perception`` imports no ``analysis``/``interfaces`` layer.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "topspin_review"
LAYERS = {"domain", "perception", "analysis", "providers", "storage", "reporting", "interfaces"}

FORBIDDEN = {
    "domain": {"perception", "analysis", "providers", "storage", "reporting", "interfaces"},
    "perception": {"analysis", "interfaces"},
}


def _layer_of(path: Path) -> str | None:
    rel = path.relative_to(PKG)
    return rel.parts[0] if len(rel.parts) > 1 else None


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
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
