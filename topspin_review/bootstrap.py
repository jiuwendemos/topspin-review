"""Explicit runtime bootstrap: create runtime dirs and configure logging.

Entry points call :func:`setup` before importing heavy dependencies; importing the
package itself does nothing (no side effects). ``setup`` is idempotent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

from .storage import runtime

_T = TypeVar("_T")


def setup() -> None:
    runtime.setup()
    from topspin_review.backend.logs import configure as configure_logging

    configure_logging(str(runtime.LOG_DIR))


def _quiet_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Ignore benign 'connection reset' noise from the Windows proactor transport."""
    if isinstance(context.get("exception"), ConnectionResetError):
        return
    loop.default_exception_handler(context)


def run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` on a fresh event loop that ignores reset-connection noise.

    Uses the default loop type (Proactor on Windows — required by the ffmpeg
    frame reader), but installs an exception filter so a model endpoint closing an
    idle connection does not print a scary traceback.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(_quiet_exception_handler)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        asyncio.set_event_loop(None)
        loop.close()
