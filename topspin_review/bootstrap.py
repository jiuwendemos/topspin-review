"""Explicit runtime bootstrap: create runtime dirs and configure logging.

Entry points call :func:`setup` before importing heavy dependencies; importing the
package itself does nothing (no side effects). ``setup`` is idempotent.
"""

from __future__ import annotations

from .storage import runtime


def setup() -> None:
    runtime.setup()
