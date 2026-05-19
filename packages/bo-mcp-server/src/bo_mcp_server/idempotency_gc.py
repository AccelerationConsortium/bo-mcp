"""Background sweep for expired idempotency_cache rows.

The opportunistic per-key purge in
:func:`bo_mcp_server.idempotency._read_existing` only fires when a
specific ``(tool_name, idempotency_key)`` pair is queried. Rows for
calls that were never retried therefore accumulate until something
queries the slot. On a long-running deployment this is unbounded
growth — index bloat eventually slows every reservation
``INSERT``.

This module exposes a lifespan-managed ``asyncio`` task that
periodically deletes every row with ``expires_at <= now`` and reports
the removed-row count to Prometheus. We intentionally avoid pulling
``apscheduler`` or relying on Postgres ``pg_cron``: a sleep-loop
inside the FastAPI / MCP server lifespan is enough for a single-
process deployment and keeps the dependency tree slim. Multi-process
deployments that already run ``pg_cron`` should set
``IDEMPOTENCY_CACHE_GC_INTERVAL_SECONDS=0`` to disable the sweep and
schedule the same ``DELETE`` server-side.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError

from bo_mcp_server.idempotency import purge_expired_cache_rows
from bo_mcp_server.metrics import record_idempotency_gc
from bo_mcp_server.settings import get_idempotency_cache_gc_interval_seconds

logger = logging.getLogger(__name__)


async def _sweep_once() -> int:
    """Run one purge cycle, swallowing only documented storage errors.

    ``SQLAlchemyError`` is caught so a transient DB blip cannot kill
    the background task — the next interval retries. Programming bugs
    propagate so they are not silently buried under the loop's
    error-handling.
    """
    try:
        removed = await purge_expired_cache_rows()
    except SQLAlchemyError:
        logger.warning("Idempotency cache GC sweep failed", exc_info=True)
        return 0
    record_idempotency_gc(removed)
    if removed:
        logger.info("Idempotency cache GC removed %d expired rows", removed)
    return removed


async def _sweep_loop(interval_seconds: float) -> None:
    """Run :func:`_sweep_once` every ``interval_seconds`` until cancelled.

    Sleeps via :func:`asyncio.sleep` so an outer ``CancelledError`` from
    lifespan shutdown is observed promptly.
    """
    while True:
        await _sweep_once()
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise


@asynccontextmanager
async def idempotency_gc_lifespan() -> AsyncGenerator[asyncio.Task[None] | None]:
    """Async context that owns the GC sweep task.

    Yields the running task (or ``None`` when the sweep is disabled)
    so tests can ``await`` an explicit shutdown. The lifespan path in
    :mod:`api.main` enters this context alongside the database
    lifespan; production interval defaults to 1h and is tunable via
    ``IDEMPOTENCY_CACHE_GC_INTERVAL_SECONDS``.
    """
    interval = get_idempotency_cache_gc_interval_seconds()
    if interval <= 0:
        logger.info("Idempotency cache GC sweep disabled (interval=%.1f)", interval)
        yield None
        return

    task = asyncio.create_task(_sweep_loop(interval), name="idempotency-cache-gc")
    logger.info("Idempotency cache GC sweep started (interval=%.1fs)", interval)
    try:
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        logger.info("Idempotency cache GC sweep stopped")
