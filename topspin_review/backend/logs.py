"""Route openjiuwen logging to a directory.

The only place openjiuwen's logging config is imported.
"""

from __future__ import annotations

_configured = False


def configure(log_dir: str) -> None:
    """Send openjiuwen logs to ``log_dir`` (best-effort, once per process)."""
    global _configured
    if _configured:
        return
    try:
        from openjiuwen.core.common.logging.log_config import configure_log_config

        configure_log_config(
            {
                "backend": "default",
                "level": "INFO",
                "log_path": str(log_dir),
                "output": ["file"],
                "interface_output": ["file"],
                "performance_output": ["file"],
            }
        )
    except Exception:
        pass
    _configured = True
