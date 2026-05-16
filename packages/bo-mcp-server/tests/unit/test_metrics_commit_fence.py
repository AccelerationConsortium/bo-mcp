"""Regression test: campaign-created counter only fires after commit.

Reference: SQLAlchemy's ``after_commit`` event fires once the
underlying transaction is durable and is skipped on rollback — exactly
the commit fence we need for transactional metrics. See
https://docs.sqlalchemy.org/en/20/orm/events.html#sqlalchemy.orm.SessionEvents.after_commit.

Before this fix the counter was bumped immediately after the
``async with session_scope(session) as db`` block exited. The
session_scope no-ops on session __aexit__ when the caller passes an
external session (the ``apply_idempotency`` path), so the counter
incremented while the row was still uncommitted; a downstream
rollback then inflated the metric relative to persisted state.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bo_mcp_server.metrics import (
    CAMPAIGNS_CREATED,
    record_campaign_created_after_commit,
)
from bo_mcp_server.storage.models import Base

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession]:
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as s:
        yield s
    await engine.dispose()


def _counter_value(backend: str) -> float:
    """Return the current value of the campaigns-created counter for ``backend``."""
    return float(CAMPAIGNS_CREATED.labels(backend)._value.get())  # noqa: SLF001


@pytest.mark.asyncio
async def test_counter_increments_only_after_commit(session: AsyncSession) -> None:
    """Arming the hook does not bump the counter; commit does."""
    before = _counter_value("botorch-fence-A")

    record_campaign_created_after_commit(session, "botorch-fence-A")
    # The hook is armed but the transaction has not committed yet.
    assert _counter_value("botorch-fence-A") == before

    await session.commit()
    # ``after_commit`` fires synchronously inside SQLAlchemy's event
    # machinery, so by the time ``session.commit()`` returns the
    # counter reflects the durable state.
    assert _counter_value("botorch-fence-A") == before + 1


@pytest.mark.asyncio
async def test_counter_does_not_increment_on_rollback(session: AsyncSession) -> None:
    """A rolled-back transaction must not bump the counter.

    Production sessions are one-per-operation: ``apply_idempotency``
    constructs the session, hands it to the operation (which arms this
    listener), then commits or rolls back, after which the session is
    disposed. The regression we care about is "rollback fires before
    after_commit, the session is then discarded" — bumping the counter
    in that path would inflate the metric relative to persisted state.
    """
    before = _counter_value("botorch-fence-B")

    record_campaign_created_after_commit(session, "botorch-fence-B")
    await session.rollback()
    # SQLAlchemy ``after_commit`` does not fire when the transaction
    # was rolled back, so the counter never moved.
    assert _counter_value("botorch-fence-B") == before


@pytest.mark.asyncio
async def test_listeners_are_detached_after_commit(session: AsyncSession) -> None:
    """Listeners must not accumulate across repeated arms on a long-lived session.

    Each arming attaches both an ``after_commit`` and an ``after_rollback``
    listener. Without explicit cleanup those listeners would survive
    every commit and pile up on an externally-owned long-lived session
    (the ``apply_idempotency`` session-aware path), slowing each
    subsequent transaction by a constant factor and leaking memory.
    The fix schedules ``event.remove`` for both listeners via
    ``loop.call_soon`` so they detach after the dispatch returns.
    """
    from sqlalchemy import text

    sync = session.sync_session
    # ``dispatch.after_commit`` / ``after_rollback`` are dynamically
    # registered SQLAlchemy event descriptors; ty doesn't see them on
    # the dispatcher protocol, so the type ignore is intentional.
    before_commit = len(sync.dispatch.after_commit)  # ty: ignore[unresolved-attribute]
    before_rollback = len(sync.dispatch.after_rollback)  # ty: ignore[unresolved-attribute]

    # Arm and commit ten times; without cleanup we'd see 20 net new
    # listeners (10 commit + 10 rollback).
    for index in range(10):
        await session.execute(text("SELECT 1"))
        record_campaign_created_after_commit(session, f"backend-cleanup-{index}")
        await session.commit()
        # ``loop.call_soon`` runs the cleanup at the next event-loop
        # tick — yielding here lets it run before we inspect the
        # listener counts.
        await asyncio.sleep(0)

    assert len(sync.dispatch.after_commit) == before_commit  # ty: ignore[unresolved-attribute]
    assert (
        len(sync.dispatch.after_rollback) == before_rollback  # ty: ignore[unresolved-attribute]
    )


@pytest.mark.asyncio
async def test_listeners_are_detached_after_rollback(session: AsyncSession) -> None:
    """The rollback path also tears down both listeners."""
    from sqlalchemy import text

    sync = session.sync_session
    before_commit = len(sync.dispatch.after_commit)  # ty: ignore[unresolved-attribute]
    before_rollback = len(sync.dispatch.after_rollback)  # ty: ignore[unresolved-attribute]

    for index in range(10):
        await session.execute(text("SELECT 1"))
        record_campaign_created_after_commit(session, f"backend-rb-{index}")
        await session.rollback()
        await asyncio.sleep(0)

    assert len(sync.dispatch.after_commit) == before_commit  # ty: ignore[unresolved-attribute]
    assert (
        len(sync.dispatch.after_rollback) == before_rollback  # ty: ignore[unresolved-attribute]
    )


@pytest.mark.asyncio
async def test_connection_bound_session_without_outer_transaction_fires_metric() -> None:
    """A connection-bound session with NO outer transaction is still durable.

    Earlier iterations of the joined-session guard skipped every
    session whose bind was a :class:`Connection`, but that was
    broader than the actual hazard: ``AsyncSession(bind=connection)``
    is a normal durable shape when the connection isn't already
    inside a transaction — the session opens its own root transaction
    on that connection and ``commit()`` writes to the database. The
    metric must fire in this case; the friend's review pinned the
    regression by pointing out this overly-broad skip.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    before = _counter_value("backend-conn-bound-durable")
    try:
        # Connection-bound but NO outer transaction: the session will
        # open its own root transaction on this connection.
        async with engine.connect() as connection:
            assert connection.in_transaction() is False
            test_session = AsyncSession(bind=connection, expire_on_commit=False)
            await test_session.execute(text("SELECT 1"))
            record_campaign_created_after_commit(test_session, "backend-conn-bound-durable")
            await test_session.commit()
            # Yield so the deferred listener detach runs.
            await asyncio.sleep(0)
            await test_session.close()
    finally:
        await engine.dispose()

    assert _counter_value("backend-conn-bound-durable") == before + 1


@pytest.mark.asyncio
async def test_non_durable_commit_detaches_listener_for_later_durable_commit() -> None:
    """A skipped non-durable commit must not let a *later* commit bump the counter.

    Bug shape: ``connection.begin()`` → arm → non-durable
    ``session.commit()`` (skipped) → ``connection.rollback()`` (does
    NOT fire the session's ``after_rollback`` listener) → later
    **durable** ``session.commit()`` on the same session. Without
    this fix the listener stayed armed through the external rollback
    and the next genuine commit retroactively credited the rolled-
    back campaign.

    The fix disarms + detaches both listeners on the non-durable
    commit path so the rollback through ``Connection`` cannot leak a
    stale arming forward into the session's lifetime.

    Crucially the second commit in this test must NOT be wrapped in
    another ``connection.begin()`` — otherwise it would itself be
    non-durable and the test would pass even with the original
    stale-listener bug. The session autobegins its own root
    transaction on the bare connection (after the external rollback
    closes the outer one), so the second ``session.commit()`` is a
    real durable commit.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    before = _counter_value("backend-detach-after-skip")
    try:
        async with engine.connect() as connection:
            outer_transaction = await connection.begin()
            test_session = AsyncSession(bind=connection, expire_on_commit=False)

            # First commit: non-durable (joined to outer transaction).
            await test_session.execute(text("SELECT 1"))
            record_campaign_created_after_commit(test_session, "backend-detach-after-skip")
            await test_session.commit()
            await asyncio.sleep(0)  # let the scheduled detach run

            # External rollback — does not fire after_rollback on the
            # session, so the stale arming (if any) would survive.
            await outer_transaction.rollback()

            # A later, **durable** commit on the bare connection
            # (no outer transaction). The metric must NOT bump for
            # the campaign that was rolled back. Without the
            # disarm-on-skip fix, the still-armed listener would
            # fire here and credit the rolled-back campaign.
            assert connection.in_transaction() is False
            await test_session.execute(text("SELECT 1"))
            await test_session.commit()
            await asyncio.sleep(0)
            await test_session.close()
    finally:
        await engine.dispose()

    assert _counter_value("backend-detach-after-skip") == before


@pytest.mark.asyncio
async def test_outer_transaction_no_savepoint_rollback_does_not_bump() -> None:
    """An outer transaction without ``begin_nested`` also defers the metric.

    The friend-1/friend-2 back-and-forth narrowed the joined-session
    guard from "isinstance(bind, Connection)" → only-savepoints →
    inspect-connection-at-commit. This shape covers what the second
    iteration missed: ``connection.begin()`` without
    ``session.begin_nested()``. In that pattern
    ``session.commit()`` commits the session's autobegun transaction
    inside the host's outer transaction; SQLAlchemy fires
    ``after_commit`` with ``in_nested_transaction() == False`` and
    the durable-commit signature is ambiguous. The right check is at
    commit time: the bound Connection is still in a transaction (the
    outer is still active), so the metric must skip.

    Regression: an outer rollback must leave the counter untouched.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    before = _counter_value("backend-outer-no-savepoint")
    try:
        async with engine.connect() as connection:
            outer_transaction = await connection.begin()
            test_session = AsyncSession(bind=connection, expire_on_commit=False)
            # NO begin_nested — the application session commits "into"
            # the outer transaction directly.
            await test_session.execute(text("SELECT 1"))
            record_campaign_created_after_commit(test_session, "backend-outer-no-savepoint")
            await test_session.commit()
            # The outer is still open; teardown rolls it back.
            await outer_transaction.rollback()
            await test_session.close()
    finally:
        await engine.dispose()

    assert _counter_value("backend-outer-no-savepoint") == before


@pytest.mark.asyncio
async def test_savepoint_release_does_not_bump_when_outer_rolls_back() -> None:
    """SAVEPOINT release inside an externally managed outer transaction is not durable.

    The test isolation pattern in ``tests/conftest_postgres.py`` (and
    the equivalent for any host that wraps each request in a
    rollback-only outer transaction) opens the connection's outer
    transaction, then a ``begin_nested()`` SAVEPOINT, then lets the
    application call ``session.commit()`` — which **releases the
    SAVEPOINT** rather than committing to the database. The
    ``after_commit`` event still fires; without the
    ``in_nested_transaction()`` guard the metric would inflate every
    time the test fixture rolls back the outer transaction at
    teardown.

    This regression test mirrors that shape with the SQLite
    ``StaticPool``: outer ``connection.begin()`` → ``begin_nested()``
    → arm → ``session.commit()`` (SAVEPOINT release) → outer
    rollback. The counter must not move.
    """
    from sqlalchemy import event, text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    before = _counter_value("backend-savepoint")
    try:
        async with engine.connect() as connection:
            outer_transaction = await connection.begin()
            test_session = AsyncSession(bind=connection, expire_on_commit=False)
            await test_session.begin_nested()

            @event.listens_for(test_session.sync_session, "after_transaction_end")
            def _restart_savepoint(_sync_session, _transaction) -> None:  # noqa: ARG001
                # Mirror the test-fixture's restart-savepoint hook so
                # the application's ``commit()`` can repeatedly release
                # and re-open SAVEPOINTs inside the outer transaction.
                sync_conn = connection.sync_connection
                if sync_conn is not None and not sync_conn.in_nested_transaction():
                    sync_conn.begin_nested()

            # Materialize the transaction state and arm the metric.
            await test_session.execute(text("SELECT 1"))
            record_campaign_created_after_commit(test_session, "backend-savepoint")
            # Application-level commit — releases the SAVEPOINT but
            # the outer transaction is still active.
            await test_session.commit()

            # The outer transaction is now rolled back — exactly what
            # the conftest fixture does at teardown.
            await outer_transaction.rollback()
            await test_session.close()
    finally:
        await engine.dispose()

    # The counter must not have moved: the SAVEPOINT release was not
    # durable, and the outer transaction's rollback discarded any
    # writes that would have justified the bump.
    assert _counter_value("backend-savepoint") == before


@pytest.mark.asyncio
async def test_rollback_then_unrelated_commit_does_not_bump_counter(
    session: AsyncSession,
) -> None:
    """Arming, rolling back, then committing unrelated work must not fire.

    The listener is transaction-scoped: rolling back the arming
    transaction disarms the commit hook so a subsequent commit on the
    same long-lived session — for entirely unrelated work — cannot
    retroactively credit the rolled-back campaign. Without the
    ``after_rollback`` disarm step the listener would survive the
    rollback and resurrect the counter bump on the next commit, even
    though the original transaction never made it to the log.

    The test materializes a real transaction by issuing a ``SELECT 1``
    before arming, then a ``ROLLBACK`` invalidates that transaction;
    the next ``commit`` corresponds to an entirely new transaction
    that has no relationship to the original arming. This mirrors the
    production hazard the friend's review flagged.
    """
    from sqlalchemy import text

    before = _counter_value("botorch-fence-C")

    # Materialize a transaction so ``after_rollback`` actually fires.
    await session.execute(text("SELECT 1"))
    record_campaign_created_after_commit(session, "botorch-fence-C")
    await session.rollback()
    # An unrelated commit on the same session must not retroactively
    # bump the counter — the listener was attached for the rolled-back
    # transaction only.
    await session.execute(text("SELECT 1"))
    await session.commit()
    assert _counter_value("botorch-fence-C") == before
