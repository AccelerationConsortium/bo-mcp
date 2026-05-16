"""Progress reporting hooks for long-running BO operations.

GP fitting, SAASBO MCMC, and acquisition optimization can take 10-30+
seconds. Without a way to surface intermediate state, agents and UIs
cannot show progress, enforce soft timeouts, or offer cancellation —
they see a long opaque RPC and the user assumes the process hung.

This module defines a small, transport-neutral hook that bo-engine
calls at coarse milestones (e.g. "fit_gp_start", "acquisition_start").
The server layer wires the hook to ``notifications/progress`` for MCP
or to a Prometheus / OpenTelemetry sink elsewhere.

Design notes:

- The hook is synchronous because most BoTorch entry points run inside
  ``asyncio.to_thread``; an async callback would have to bounce back
  to the event loop on every milestone. Server-side adapters that need
  to await an async sink should marshal via
  ``asyncio.run_coroutine_threadsafe`` themselves.
- ``progress`` is monotonically non-decreasing inside one call. ``total``
  is optional: not every phase knows its total upfront (MCMC sample
  count is, GP fit isn't).
- The hook is always optional. Callers that pass ``None`` get the
  current "silent" behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEvent:
    """Coarse milestone emitted during a long-running BO operation."""

    phase: str
    """Stable short identifier (``"fit_gp_start"``, ``"acquisition_start"``,
    ``"acquisition_done"``, ``"diagnostics_start"``, etc.). Stable enough
    that downstream alerting / UIs can dispatch on it without parsing
    free-form text."""

    message: str
    """Human-readable message suitable for showing to a user."""

    progress: float | None = None
    """Optional fractional progress in ``[0, total]``. The MCP spec
    treats ``progress`` as a free-form number paired with ``total`` so
    fractional values are fine."""

    total: float | None = None
    """Optional total for the current phase. ``None`` means the phase
    is event-only (e.g. a start/done milestone)."""


ProgressCallback = Callable[[ProgressEvent], None]
"""Synchronous callback invoked at progress milestones.

Implementations must be cheap and non-blocking. They are called from
worker threads under ``asyncio.to_thread``; long-running or blocking
work inside the callback stalls the operation just as if it ran inline.
"""


def emit(callback: ProgressCallback | None, event: ProgressEvent) -> None:
    """Invoke ``callback`` with ``event`` if it is provided.

    Wrapping the optional-callback pattern keeps every call site free
    of ``if callback is not None`` boilerplate, and lets future versions
    add cross-cutting behavior (e.g. timing, telemetry, swallow-and-log)
    in one place.
    """
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001 — progress reporting must never crash optimization
        # A bug in the consumer (e.g. broken MCP session) must not
        # propagate into the BO loop. Log via the standard logger so
        # the failure is observable without taking the operation down.
        import logging

        logging.getLogger(__name__).exception(
            "Progress callback raised; swallowing", extra={"phase": event.phase}
        )
