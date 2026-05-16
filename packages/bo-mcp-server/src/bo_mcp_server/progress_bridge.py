"""Bridge between bo-engine's synchronous progress hook and MCP's async session.

bo-engine emits :class:`bo_engine.progress.ProgressEvent` via a
synchronous callback because most of its work runs under
``asyncio.to_thread``. The MCP session, on the other hand, exposes
``ctx.report_progress(progress, total, message)`` as an *async* method.

This module bridges the two: build a sync callback that captures the
running event loop and an async session, and forwards each event by
scheduling the async ``report_progress`` call back onto that loop via
``asyncio.run_coroutine_threadsafe``. Failures in the forwarding path
are swallowed (logged at WARNING) so a flaky MCP transport never takes
down the BO loop — the contract on the engine side is the same.

Use :func:`make_progress_callback_from_context` when you have a FastMCP
:class:`Context` (the case inside ``@mcp.tool`` handlers). For tests and
non-MCP callers, pass any async callable matching
``Callable[[float, float | None, str | None], Awaitable[None]]`` to
:func:`build_progress_callback`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from bo_engine.progress import ProgressCallback, ProgressEvent

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)


ReportProgress = Callable[[float, float | None, str | None], Awaitable[None]]
"""Signature of MCP ``Context.report_progress`` so we do not depend on
the MCP type from non-MCP test paths."""


def build_progress_callback(
    report_progress: ReportProgress,
    loop: asyncio.AbstractEventLoop,
) -> ProgressCallback:
    """Build a sync :class:`ProgressCallback` that forwards to an async sink.

    The returned callback is thread-safe: it submits the async call to
    the given loop and does not block on its completion. Exceptions are
    logged but never re-raised so progress reporting can fail without
    impacting BO correctness.
    """

    async def _send(progress: float, total: float | None, message: str | None) -> None:
        await report_progress(progress, total, message)

    def callback(event: ProgressEvent) -> None:
        progress = event.progress if event.progress is not None else 0.0
        message = f"{event.phase}: {event.message}"
        try:
            asyncio.run_coroutine_threadsafe(_send(progress, event.total, message), loop)
        except RuntimeError:
            # Loop is closed or not running. Fire-and-forget: drop the
            # event rather than crash the BO loop.
            logger.debug("Progress event dropped (loop unavailable): %s", event.phase)

    return callback


def make_progress_callback_from_context(
    ctx: Context | None,
) -> ProgressCallback | None:
    """Make a callback that routes progress to an MCP session, if available.

    Returns ``None`` when the context is missing or carries no progress
    token (e.g. the client did not opt in to progress notifications).
    Engine-side code treats a ``None`` callback as silent, so falling
    through to ``None`` is the right default.
    """
    if ctx is None:
        return None

    meta = ctx.request_context.meta
    if meta is None or meta.progressToken is None:
        return None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Should never happen — MCP tools run inside an event loop —
        # but defend against the case where the bridge is constructed
        # outside that scope (tests, scripts).
        logger.debug("No running event loop; cannot build progress bridge.")
        return None

    return build_progress_callback(ctx.report_progress, loop)
