"""Regression test: ``snapshot_db_pool`` must not lazy-create the engine.

A ``/metrics`` scrape that happens before any real DB traffic should
observe an empty snapshot — opening a connection pool *because*
Prometheus is scraping would skew startup metrics (connection-pool
init time, "engine present" gauges) and conflict with the contract the
helper's docstring advertises.

Reference: Prometheus's "observability without side effects" guidance —
https://prometheus.io/docs/practices/instrumentation/ — explicitly
discourages "metrics that create the thing they measure".
"""

from __future__ import annotations

import pytest

from bo_mcp_server.metrics import snapshot_db_pool


@pytest.mark.asyncio
async def test_snapshot_returns_empty_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the engine slot is None the snapshot must return ``{}``.

    Crucially, the call must NOT call ``_get_engine()`` — that
    function lazily constructs an engine, which is the side effect
    this fix prevents.
    """
    from bo_mcp_server.storage import database

    # Snapshot a state where the engine slot is empty. Saving the
    # current slot lets the test restore it so concurrent test fixtures
    # don't see a wiped engine.
    saved = getattr(database, "_engine", None)
    monkeypatch.setattr(database, "_engine", None, raising=False)

    # Sentinel: if anything inside ``snapshot_db_pool`` calls
    # ``_get_engine`` we'd see the engine slot mutated.
    def _fail_if_called() -> None:
        msg = "snapshot_db_pool must not call _get_engine"
        raise AssertionError(msg)

    monkeypatch.setattr(database, "_get_engine", _fail_if_called)

    assert snapshot_db_pool() == {}

    # Restore so the surrounding suite still has a working engine.
    monkeypatch.setattr(database, "_engine", saved, raising=False)
