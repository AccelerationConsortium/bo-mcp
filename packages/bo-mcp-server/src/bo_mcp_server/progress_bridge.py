"""Bridge between bo-engine's synchronous progress hook and MCP's async session.

bo-engine emits :class:`bo_engine.progress.ProgressEvent` via a
synchronous callback because most of its work runs under
``asyncio.to_thread``. The MCP session, on the other hand, exposes
``ctx.report_progress(progress, total, message)`` as an *async* method.

This module bridges the two: build a sync callback that captures the
running event loop and an async session, and forwards each event by
scheduling the async ``report_progress`` call back onto that loop via
``asyncio.run_coroutine_threadsafe``.

Failure observability
---------------------

Forwarding failures used to be swallowed at ``DEBUG`` with no metric.
That hid a real correctness gap for clients relying on progress for
ETAs or cancellation: when the bridge dies (loop closed, transport
crash), nothing on the operator side flagged that progress had gone
silent.

The bridge now:

* tracks consecutive failures per callback instance — the first
  failure logs at ``WARNING`` and every subsequent failure inside the
  same run also logs at ``WARNING`` (downgrading to ``DEBUG`` once
  the bridge has recovered),
* always bumps the new ``bo_mcp_progress_notify_failures_total`` counter
  (labelled by failure reason) so a dashboard can alarm on a sustained
  non-zero rate, and
* exposes ``progress_status_snapshot`` so the ``bo_check_progress``
  tool can poll the latest event when the push channel has gone
  silent.

Use :func:`make_progress_callback_from_context` when you have a FastMCP
:class:`Context` (the case inside ``@mcp.tool`` handlers). For tests and
non-MCP callers, pass any async callable matching
``Callable[[float, float | None, str | None], Awaitable[None]]`` to
:func:`build_progress_callback`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bo_engine.progress import ProgressCallback, ProgressEvent
from bo_mcp_server.metrics import PROGRESS_NOTIFY_FAILURES

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)


ReportProgress = Callable[[float, float | None, str | None], Awaitable[None]]
"""Signature of MCP ``Context.report_progress`` so we do not depend on
the MCP type from non-MCP test paths."""


@dataclass
class ProgressStatus:
    """Latest progress snapshot exposed for poll-fallback consumption.

    Holds the most recently observed event so ``bo_check_progress`` can
    answer "what is the long-running operation doing right now?"
    without depending on the push channel. ``failures`` is the running
    count of forwarding errors since the bridge was constructed; a
    non-zero count tells the agent that push has dropped events and
    polling is the canonical readout.
    """

    progress: float | None = None
    total: float | None = None
    message: str | None = None
    phase: str | None = None
    failures: int = 0
    last_failure_reason: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_event(self, event: ProgressEvent) -> None:
        """Atomically replace the latest progress event under the bridge lock."""
        with self._lock:
            self.progress = event.progress
            self.total = event.total
            self.message = event.message
            self.phase = event.phase

    def record_failure(self, reason: str) -> None:
        """Increment the failure counter and remember the latest reason."""
        with self._lock:
            self.failures += 1
            self.last_failure_reason = reason

    def snapshot(self) -> dict[str, float | str | int | None]:
        """Return an immutable snapshot for poll consumers."""
        with self._lock:
            return {
                "progress": self.progress,
                "total": self.total,
                "message": self.message,
                "phase": self.phase,
                "failures": self.failures,
                "last_failure_reason": self.last_failure_reason,
            }


def build_progress_callback(
    report_progress: ReportProgress,
    loop: asyncio.AbstractEventLoop,
    status: ProgressStatus | None = None,
) -> ProgressCallback:
    """Build a sync :class:`ProgressCallback` that forwards to an async sink.

    The returned callback is thread-safe: it submits the async call to
    the given loop and does not block on its completion. Forwarding
    failures are surfaced through three channels — a ``WARNING`` log
    line, the ``bo_mcp_progress_notify_failures_total`` Prometheus
    counter, and the optional :class:`ProgressStatus` snapshot used by
    the poll fallback. The callback itself never re-raises: BO
    correctness must not depend on the transport.

    Args:
        report_progress: Async sink (typically ``ctx.report_progress``).
        loop: Running event loop the async sink is bound to.
        status: Optional shared status object that records the most
            recent event and any failures, so a poll-fallback caller
            can read the latest state.
    """

    async def _send(progress: float, total: float | None, message: str | None) -> None:
        await report_progress(progress, total, message)

    def _record_send_failure(reason: str, phase: str | None, detail: object) -> None:
        if status is not None:
            status.record_failure(reason)
        PROGRESS_NOTIFY_FAILURES.labels(reason).inc()
        logger.warning(
            "Progress event dropped (%s): phase=%s err=%s",
            reason,
            phase,
            detail,
        )

    def _handle_done_future(phase: str | None, fut: concurrent.futures.Future[None]) -> None:
        """Route a failed async send through the same surface as a sync drop.

        ``asyncio.run_coroutine_threadsafe`` returns a future whose
        exception lives there silently unless we explicitly read it.
        ``ctx.report_progress`` raising on the event loop, or a
        cancellation during teardown, must both surface via the
        WARNING + counter + snapshot path so a stuck client cannot
        wait forever without an operator-visible signal.
        """
        if fut.cancelled():
            _record_send_failure("cancelled", phase, "future cancelled")
            return
        exc = fut.exception()
        if exc is not None:
            _record_send_failure("send_failed", phase, exc)

    def callback(event: ProgressEvent) -> None:
        if status is not None:
            status.update_event(event)
        progress = event.progress if event.progress is not None else 0.0
        message = f"{event.phase}: {event.message}"
        coro = _send(progress, event.total, message)
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError as exc:
            # The most common cause is the loop having closed between
            # the BO worker thread reading the loop ref and now
            # (transport hangup mid-run). Promote to WARNING with a
            # metric bump so operators can observe the drop instead of
            # silently dropping the event at DEBUG. The exception is
            # not re-raised — BO correctness must not depend on a
            # working progress channel.
            #
            # ``run_coroutine_threadsafe`` only takes ownership of the
            # coroutine on success; when it raises we still own it and
            # must close it ourselves, otherwise the GC reports the
            # un-awaited coroutine as a leak.
            coro.close()
            _record_send_failure("loop_closed", event.phase, exc)
            return
        # The submission succeeded; attach a completion callback so a
        # failure on the *async* side (sink raises on the event loop,
        # task cancelled during teardown) is also routed through the
        # warning + counter + snapshot path. Without this, a runtime
        # exception inside ``ctx.report_progress`` would die silently
        # on the future and the snapshot's ``failures`` counter would
        # stay at zero — exactly the regression the audit flagged.
        phase = event.phase
        future.add_done_callback(lambda fut, _phase=phase: _handle_done_future(_phase, fut))

    return callback


def make_progress_callback_from_context(
    ctx: Context | None,
    status: ProgressStatus | None = None,
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

    return build_progress_callback(ctx.report_progress, loop, status=status)


# ---------------------------------------------------------------------------
# Process-local progress registry (poll fallback)
# ---------------------------------------------------------------------------


# Cap on simultaneously-tracked progress snapshots. A bounded LRU-by-
# insertion-order keeps memory steady when clients churn through
# unique correlation ids without ever polling. The cap is large enough
# for normal multi-tenant operation but small enough that a runaway
# producer cannot exhaust process memory.
_PROGRESS_REGISTRY_MAX_ENTRIES = 256


_progress_registry: dict[str, ProgressStatus] = {}
_progress_registry_lock = threading.Lock()


def register_progress_status(token: str | None, status: ProgressStatus) -> None:
    """Register a status object so :func:`get_progress_status` can find it.

    ``token`` is the long-running-operation correlation id — the
    campaign id suffixed with a per-call unique token
    (``"{campaign_id}:{suffix}"``). The suffix keeps two concurrent
    operations on the same campaign from clobbering each other's
    registration: each call owns its own key, so the first caller's
    cleanup cannot delete the second caller's entry.
    :func:`get_progress_status` prefix-matches on the campaign id and
    returns the newest registration. Passing ``None`` is a no-op so
    call sites that do not have a stable correlation id can skip
    registration without conditionals.

    The registry is process-local; clients polling through a different
    replica will not see in-flight state. That matches the audit's
    "operational hardening" framing — push remains the primary channel,
    poll is the fallback when push has gone silent on the same
    process.
    """
    if token is None:
        return
    with _progress_registry_lock:
        if len(_progress_registry) >= _PROGRESS_REGISTRY_MAX_ENTRIES:
            # Evict the oldest insertion-order entry. ``dict`` preserves
            # insertion order in Python 3.7+, so the first key is the
            # oldest; popping it bounds the registry size.
            oldest_key = next(iter(_progress_registry))
            _progress_registry.pop(oldest_key, None)
        _progress_registry[token] = status


def unregister_progress_status(token: str | None) -> None:
    """Drop the registered status when the operation completes."""
    if token is None:
        return
    with _progress_registry_lock:
        _progress_registry.pop(token, None)


def get_progress_status(token: str) -> ProgressStatus | None:
    """Look up a registered status by correlation id.

    An exact key match wins; otherwise ``token`` is treated as the
    campaign-id prefix of the per-call keys written by
    :func:`register_progress_status` and the **newest** matching
    registration is returned (the registry preserves insertion order),
    so a poller asking about a campaign with two in-flight operations
    reads the most recently started one.
    """
    prefix = f"{token}:"
    with _progress_registry_lock:
        exact = _progress_registry.get(token)
        if exact is not None:
            return exact
        newest: ProgressStatus | None = None
        for key, status in _progress_registry.items():
            if key.startswith(prefix):
                newest = status
        return newest


__all__ = [
    "ProgressStatus",
    "ReportProgress",
    "build_progress_callback",
    "get_progress_status",
    "make_progress_callback_from_context",
    "register_progress_status",
    "unregister_progress_status",
]
