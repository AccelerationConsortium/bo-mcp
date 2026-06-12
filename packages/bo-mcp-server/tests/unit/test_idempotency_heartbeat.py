"""Reservation heartbeat for slow idempotent operations.

The pending-reservation TTL defaults to 10 minutes. SAASBO MCMC and
large-batch generation can plausibly exceed that, at which point a
concurrent retry would reclaim the slot and the original operation
would surface its writes as a ``stale_owner`` envelope (the existing
conflict path).

The heartbeat extends ``expires_at`` from inside the active context
so the legitimately-slow caller keeps its slot. The mechanism is
opt-in via :func:`reservation_heartbeat`; only call sites that have
been audited as 'will-not-block-the-event-loop' should wrap their
body in it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.idempotency import (
    _PENDING_SENTINEL,
    _active_reservation,
    _ActiveReservation,
    _extend_reservation_ttl,
    apply_idempotency,
    extend_active_reservation,
    reservation_heartbeat,
)
from bo_mcp_server.storage import get_session
from bo_mcp_server.storage.models import IdempotencyCacheModel

pytestmark = pytest.mark.usefixtures("setup_database")


async def _seed_pending_reservation(tool: str, key: str, token: str) -> dt.datetime:
    """Insert a pending reservation row in the recent past.

    Returns the original ``expires_at`` so the test can assert the
    heartbeat moved it forward.
    """
    original_expiry = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=5)
    async with get_session() as session:
        session.add(
            IdempotencyCacheModel(
                tool_name=tool,
                idempotency_key=key,
                request_hash="hash",
                reservation_token=token,
                response_json=_PENDING_SENTINEL,
                created_at=dt.datetime.now(dt.UTC),
                expires_at=original_expiry,
            )
        )
    return original_expiry


async def _read_expiry(tool: str, key: str) -> dt.datetime:
    async with get_session() as session:
        row = (
            (
                await session.execute(
                    select(IdempotencyCacheModel).where(
                        IdempotencyCacheModel.tool_name == tool,
                        IdempotencyCacheModel.idempotency_key == key,
                    )
                )
            )
            .scalars()
            .one()
        )
        ts = row.expires_at
    return ts if ts.tzinfo else ts.replace(tzinfo=dt.UTC)


@pytest.mark.asyncio
async def test_extend_reservation_ttl_moves_expiry_forward() -> None:
    """A successful extension shifts ``expires_at`` past the original value."""
    tool, key, token = "heartbeat_tool", "hb-1", "tok-aaaaaaaaaaaaaaaaaaaaaaaaaa"
    original = await _seed_pending_reservation(tool, key, token)

    bumped = await _extend_reservation_ttl(tool, key, token, extra_seconds=600)
    assert bumped is True

    new_expiry = await _read_expiry(tool, key)
    assert new_expiry > original


@pytest.mark.asyncio
async def test_extend_reservation_never_shortens_existing_expiry() -> None:
    """A heartbeat tick cannot pull ``expires_at`` backward.

    Reproducer for the off-by-design bug: the initial reservation is
    seeded 10 minutes out; a heartbeat at ~60 s with a short extension
    (e.g. 5 min) must NOT reset the deadline to "now + 5 min" — that
    would shorten the originally-granted slot by ~5 min and *cause*
    the very reclaim race the heartbeat is supposed to prevent.

    The fix uses ``CASE WHEN expires_at < candidate THEN candidate
    ELSE expires_at END`` so the deadline moves monotonically. The
    function still returns ``True`` (slot is still ours) so the
    heartbeat loop keeps running.
    """
    tool, key, token = "heartbeat_tool", "hb-monotonic", "tok-aaaaaaaaaaaaaaaaaaaaaaaaaa"
    # Seed a generous 30-minute initial reservation.
    long_expiry = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=30)
    async with get_session() as session:
        session.add(
            IdempotencyCacheModel(
                tool_name=tool,
                idempotency_key=key,
                request_hash="hash",
                reservation_token=token,
                response_json=_PENDING_SENTINEL,
                created_at=dt.datetime.now(dt.UTC),
                expires_at=long_expiry,
            )
        )

    # Heartbeat tick with a SHORT extension (5 min). Must not move the
    # existing deadline backward.
    still_ours = await _extend_reservation_ttl(tool, key, token, extra_seconds=300)
    assert still_ours is True, "matched row → caller keeps heartbeating"

    after = await _read_expiry(tool, key)
    assert after >= long_expiry - dt.timedelta(seconds=1), (
        "heartbeat must not shorten an existing-and-later expiry"
    )


@pytest.mark.asyncio
async def test_extend_reservation_no_op_when_token_does_not_match() -> None:
    """A stale owner cannot bump the slot of the newer reservation.

    Models the post-reclaim case: the token from the original slow
    operation no longer matches the row's ``reservation_token`` (the
    retry winner has a fresh UUID), so the heartbeat update must
    affect zero rows and the function returns ``False``.
    """
    tool, key, token = "heartbeat_tool", "hb-stale", "tok-aaaaaaaaaaaaaaaaaaaaaaaaaa"
    await _seed_pending_reservation(tool, key, token)

    still_ours = await _extend_reservation_ttl(
        tool, key, "tok-different-token-zzzzzzzzzzzz", extra_seconds=600
    )
    assert still_ours is False


@pytest.mark.asyncio
async def test_extend_reservation_skips_finalized_rows() -> None:
    """A row already carrying a real response must not have its TTL bumped.

    The pending sentinel guard is load-bearing: without it, the
    heartbeat could indefinitely extend a finalized response row past
    the documented 24h TTL, which would silently change the cache
    contract for replays.
    """
    tool, key, token = "heartbeat_tool", "hb-final", "tok-aaaaaaaaaaaaaaaaaaaaaaaaaa"
    finalized_expiry = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
    async with get_session() as session:
        session.add(
            IdempotencyCacheModel(
                tool_name=tool,
                idempotency_key=key,
                request_hash="hash",
                reservation_token=token,
                response_json='{"success": true}',
                created_at=dt.datetime.now(dt.UTC),
                expires_at=finalized_expiry,
            )
        )

    bumped = await _extend_reservation_ttl(tool, key, token, extra_seconds=600)
    assert bumped is False


@pytest.mark.asyncio
async def test_extend_active_reservation_uses_contextvar() -> None:
    """``extend_active_reservation`` reads the bound :class:`_ActiveReservation`."""
    tool, key, token = "heartbeat_tool", "hb-ctx", "tok-aaaaaaaaaaaaaaaaaaaaaaaaaa"
    await _seed_pending_reservation(tool, key, token)

    handle = _active_reservation.set(
        _ActiveReservation(tool_name=tool, idempotency_key=key, reservation_token=token)
    )
    try:
        result = await extend_active_reservation(extra_seconds=900)
    finally:
        _active_reservation.reset(handle)
    assert result is True


@pytest.mark.asyncio
async def test_extend_active_reservation_outside_context_is_noop() -> None:
    """No bound reservation → no DB write, returns ``False``."""
    assert _active_reservation.get() is None
    assert await extend_active_reservation(extra_seconds=600) is False


@pytest.mark.asyncio
async def test_heartbeat_extends_slot_during_slow_apply_idempotency() -> None:
    """Slow executor + heartbeat → expires_at moves forward; finalize succeeds.

    Reproduces the documented test strategy: a slow fake executor +
    a heartbeat that ticks on a sub-second cadence (test-only override).
    Before the heartbeat, the pending row's ``expires_at`` would have
    elapsed inside the synthetic ``reservation_ttl_seconds=1``; after
    the heartbeat, the row's TTL is pushed past the executor's
    completion time and the response finalizes normally.
    """
    captured: dict[str, dt.datetime] = {}

    async def slow_executor(_session: AsyncSession) -> dict[str, Any]:
        # Confirm the heartbeat actually fires while we're "computing".
        async with reservation_heartbeat(period_seconds=0.05, extension_seconds=10.0):
            await asyncio.sleep(0.3)
            captured["mid_run_expiry"] = await _read_expiry("slow_tool", "slow-1")
        return {"success": True, "ran": True}

    response = await apply_idempotency(
        tool_name="slow_tool",
        idempotency_key="slow-1",
        request_payload={"v": 1},
        executor=slow_executor,
        reservation_ttl_seconds=1,
    )
    assert response["success"] is True
    assert response.get("ran") is True
    # The mid-run snapshot must be in the future relative to the
    # original 1-second reservation TTL; the heartbeat pushed it well
    # past wall-clock + reservation_ttl.
    assert captured["mid_run_expiry"] > dt.datetime.now(dt.UTC) - dt.timedelta(seconds=5)


@pytest.mark.asyncio
async def test_heartbeat_is_noop_outside_idempotency_context() -> None:
    """Calling the context manager without an active reservation must not crash.

    Direct call sites of ``_generate_via_backend`` (no idempotency
    wrapper) hit the heartbeat context unconditionally. The contract
    is: silently no-op when the contextvar is unset so legacy callers
    are not affected.
    """
    assert _active_reservation.get() is None
    async with reservation_heartbeat(period_seconds=0.05, extension_seconds=10.0):
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_heartbeat_runtime_cap_counts_elapsed_not_extension_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The heartbeat bound is elapsed runtime (one cadence per beat), not Σ extension.

    ``_extend_reservation_ttl`` applies a monotonic ``max(current, now+ext)``,
    so a beat can match the row (return ``True``) without moving the deadline.
    Charging the requested ``extension`` against the bound would let a single
    large (or early no-op) extension burn the whole budget in one beat and stop
    the heartbeat while the compute is still running. With a runtime bound a
    huge per-beat ``extension`` is irrelevant — the heartbeat still runs
    ``cap / cadence`` beats. (Under the old extension-amount accounting the 10 s
    extension below would stop the beat after a *single* tick.)
    """
    import bo_mcp_server.idempotency as idem

    extend_calls = 0

    async def counting_extend(_extra_seconds: float) -> bool:
        nonlocal extend_calls
        extend_calls += 1
        return True

    monkeypatch.setattr(idem, "extend_active_reservation", counting_extend)
    monkeypatch.setattr(
        idem,
        "get_idempotency_heartbeat_max_total_extension_seconds",
        lambda: 0.05,
    )

    tool, key, token = "hb", "runtime-cap", "tok-cccccccccccccccccccccccccc"
    handle = _active_reservation.set(
        _ActiveReservation(tool_name=tool, idempotency_key=key, reservation_token=token)
    )
    try:
        async with reservation_heartbeat(period_seconds=0.01, extension_seconds=10.0):
            # Sleep far longer than the bound needs; the runtime cap, not the
            # sleep, must be what stops the beat.
            await asyncio.sleep(0.3)
    finally:
        _active_reservation.reset(handle)

    # cap 0.05 / cadence 0.01 = 5 beats, independent of the 10 s extension.
    assert extend_calls == 5


@pytest.mark.asyncio
async def test_heartbeat_runs_through_noop_extensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Early no-op extensions (deadline already beyond ``now+ext``) don't stop it early.

    Reproduces the regression against a *real* cache row: a reservation whose
    deadline sits far beyond ``now + extension`` makes every extend a monotonic
    no-op (it matches the row but leaves ``expires_at`` untouched). With the
    runtime bound the heartbeat keeps beating through the no-ops instead of
    stopping after ``cap / extension`` beats — the pre-fix behaviour that would
    expire a still-running 12–20 minute compute the compute timeout still
    allows. The previous mock-based test could not catch this because it forced
    every extend to be "effective".
    """
    import bo_mcp_server.idempotency as idem

    tool, key, token = "hb", "noop-extend", "tok-dddddddddddddddddddddddddd"
    # Deadline far beyond ``now + extension`` so every extend is a monotonic
    # no-op (matches the row, deadline unchanged).
    far_future = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=3600)
    async with get_session() as session:
        session.add(
            IdempotencyCacheModel(
                tool_name=tool,
                idempotency_key=key,
                request_hash="hash",
                reservation_token=token,
                response_json=_PENDING_SENTINEL,
                created_at=dt.datetime.now(dt.UTC),
                expires_at=far_future,
            )
        )

    real_extend = idem.extend_active_reservation
    calls = 0

    async def spy_extend(extra_seconds: float | None = None) -> bool:
        nonlocal calls
        calls += 1
        return await real_extend(extra_seconds)

    monkeypatch.setattr(idem, "extend_active_reservation", spy_extend)
    monkeypatch.setattr(
        idem,
        "get_idempotency_heartbeat_max_total_extension_seconds",
        lambda: 0.05,
    )

    handle = _active_reservation.set(
        _ActiveReservation(tool_name=tool, idempotency_key=key, reservation_token=token)
    )
    try:
        # extension (1 s) exceeds the 0.05 s cap; under the old extension-amount
        # accounting the first no-op extend would burn the whole budget and
        # stop after one beat.
        async with reservation_heartbeat(period_seconds=0.01, extension_seconds=1.0):
            await asyncio.sleep(0.3)
    finally:
        _active_reservation.reset(handle)

    # The heartbeat kept beating through the no-ops: ~cap/cadence = 5 beats.
    assert calls == 5
