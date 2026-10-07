"""Runtime directory layout and logging setup (self-contained).

All generated state lives under ``runtime/`` next to the project so the project
root holds only source:

    runtime/
    ├── workspace/   # DeepAgent workspace scaffold (memory/, todo/, ...)
    ├── logs/        # openjiuwen logs
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
    """Create the runtime directories and route openjiuwen logging into LOG_DIR."""
    global _configured
    if _configured:
        return
    for directory in (WORKSPACE_DIR, LOG_DIR, DATA_DIR, CACHE_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        from openjiuwen.core.common.logging.log_config import configure_log_config

        configure_log_config(
            {
                "backend": "default",
                "level": "INFO",
                "log_path": str(LOG_DIR),
                "output": ["file"],
                "interface_output": ["file"],
                "performance_output": ["file"],
            }
        )
    except Exception:
        pass
    _configured = True


def workspace() -> str:
    """Absolute path used as the DeepAgent workspace root."""
    return str(WORKSPACE_DIR)
