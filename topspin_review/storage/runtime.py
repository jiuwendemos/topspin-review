"""Runtime directory layout and logging setup (self-contained).

All generated state lives under ``runtime/`` next to the project so the project
root holds only source:

    runtime/
    ├── workspace/   # DeepAgent workspace scaffold (memory/, todo/, ...)
    ├── logs/        # agent logs
    └── data/        # local JSON state, caches, artifacts
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # topspin_review/
PROJECT_ROOT = _PACKAGE_DIR.parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
WORKSPACE_DIR = RUNTIME_DIR / "workspace"
LOG_DIR = RUNTIME_DIR / "logs"
DATA_DIR = RUNTIME_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

_configured = False


def setup() -> None:
    """Create the runtime directories (idempotent)."""
    global _configured
    if _configured:
        return
    for directory in (WORKSPACE_DIR, LOG_DIR, DATA_DIR, CACHE_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    _configured = True


def workspace() -> str:
    """Absolute path used as the DeepAgent workspace root."""
    return str(WORKSPACE_DIR)
