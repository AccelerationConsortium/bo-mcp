"""Periodic GC of the idempotency_cache (TODO 8.23).

The opportunistic per-key purge in
:func:`bo_mcp_server.idempotency._read_existing` only fires when that
specific ``(tool_name, idempotency_key)`` pair is queried. Without a
periodic sweep, rows for never-retried calls accumulate until index
bloat eventually slows every reservation. This suite exercises the
sweep in isolation: it shrinks the response TTL, fills the cache, and
asserts the lifespan-managed task drains expired rows.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.idempotency import apply_idempotency, purge_expired_cache_rows
from bo_mcp_server.idempotency_gc import idempotency_gc_lifespan
from bo_mcp_server.storage import get_session
from bo_mcp_server.storage.models import IdempotencyCacheModel

pytestmark = pytest.mark.usefixtures("setup_database")


async def _populate_expired_rows(count: int) -> None:
    """Insert ``count`` already-expired idempotency cache rows.

    Bypasses :func:`apply_idempotency` so we can deterministically
    backdate ``expires_at`` rather than waiting for wall-clock TTL.
    """
    past = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    async with get_session() as session:
        for i in range(count):
            session.add(
                IdempotencyCacheModel(
                    tool_name="gc_test",
                    idempotency_key=f"key-{i}",
                    request_hash=f"hash-{i}",
                    reservation_token=f"tok-{i:032d}",
                    response_json='{"success": true}',
                    created_at=past,
                    expires_at=past,
                )
            )


async def _row_count() -> int:
    """Return the current idempotency_cache row count."""
    async with get_session() as session:
        result = await session.execute(select(IdempotencyCacheModel))
        return len(result.scalars().all())


@pytest.mark.asyncio
async def test_purge_expired_cache_rows_drains_stale_entries() -> None:
    """Direct call to the purge helper deletes every expired row."""
    await _populate_expired_rows(7)
    assert await _row_count() == 7

    removed = await purge_expired_cache_rows()
    assert removed == 7
    assert await _row_count() == 0


@pytest.mark.asyncio
async def test_purge_leaves_live_reservations_intact() -> None:
    """The sweep must not touch rows whose ``expires_at`` is still in the future.

    A pending reservation winner (``response_json == ""``) gets its
    expires_at set to ``now + reservation_ttl``; a successful response
    gets the full 24h response TTL. Both shapes are 'live' until the
    timestamp passes — the sweep is keyed exclusively off
    ``expires_at <= now``.
    """

    async def run(_session: AsyncSession) -> dict[str, Any]:
        return {"success": True}

    await apply_idempotency(
        tool_name="live_tool",
        idempotency_key="live-key",
        request_payload={"k": "v"},
        executor=run,
    )
    assert await _row_count() == 1

    await _populate_expired_rows(3)
    assert await _row_count() == 4

    removed = await purge_expired_cache_rows()
    assert removed == 3, "only expired rows should be removed"
    assert await _row_count() == 1, "the live reservation must survive the sweep"


@pytest.mark.asyncio
async def test_lifespan_sweep_drains_table_after_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continuous expired rows + a fast sweep → table size stabilizes near zero.

    Models the 'long-lived deployment' scenario: never-retried calls
    leak rows, the lifespan-managed task runs on a short interval,
    table size stays bounded.
    """
    monkeypatch.setenv("IDEMPOTENCY_CACHE_GC_INTERVAL_SECONDS", "0.05")

    await _populate_expired_rows(20)
    assert await _row_count() == 20

    async with idempotency_gc_lifespan() as task:
        assert task is not None, "non-zero interval must start the sweep task"
        # Give the loop time for at least one purge cycle.
        await asyncio.sleep(0.25)

    assert await _row_count() == 0, "background sweep must drain expired rows"


@pytest.mark.asyncio
async def test_lifespan_sweep_disabled_when_interval_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``IDEMPOTENCY_CACHE_GC_INTERVAL_SECONDS=0`` opts out of the in-process sweep.

    Deployments that already run ``pg_cron`` (or equivalent) for the
    same DELETE can disable the in-process sweep to avoid double-
    scheduling. The context manager yields ``None`` to signal that
    no task was started.
    """
    monkeypatch.setenv("IDEMPOTENCY_CACHE_GC_INTERVAL_SECONDS", "0")
    async with idempotency_gc_lifespan() as task:
        assert task is None, "zero interval must not start the sweep task"
