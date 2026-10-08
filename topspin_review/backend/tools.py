"""openjiuwen ``@tool`` decorator, re-exported.

The rest of the app defines tools without importing openjiuwen directly. Falls
back to a no-op decorator when openjiuwen isn't installed, so tool modules stay
importable offline.
"""

from __future__ import annotations

try:  # openjiuwen is provided by the host environment
    from openjiuwen.core.foundation.tool import tool
except Exception:  # pragma: no cover

    def tool(**_kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        return _wrap
