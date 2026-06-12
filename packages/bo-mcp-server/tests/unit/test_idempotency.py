"""Idempotency cache.

Background: state-mutating MCP tools accept an optional ``idempotency_
key``. Retries with the same key + payload return the cached response
(``idempotency_replay=True``) instead of re-executing; retries with a
mismatched payload return a ``VALIDATION_FAILED`` conflict envelope.
Concurrent retries with the same key + payload must not both run the
operation — the reservation pattern in
:func:`bo_mcp_server.idempotency.apply_idempotency` is the load-bearing
invariant.

This suite exercises the in-process store directly, isolated from any
specific tool, and includes a concurrency test that fires two retries
under :func:`asyncio.gather` and asserts exactly one side effect.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from bo_mcp_server.idempotency import (
    apply_idempotency,
    canonical_request_hash,
    digest_large_field,
)

pytestmark = pytest.mark.usefixtures("setup_database")


def test_canonical_hash_is_order_independent() -> None:
    """Two dicts with the same logical keys hash equal regardless of insertion order."""
    a = {"alpha": 1, "beta": [1, 2, {"nested": True}]}
    b = {"beta": [1, 2, {"nested": True}], "alpha": 1}
    assert canonical_request_hash(a) == canonical_request_hash(b)


def test_canonical_hash_changes_for_different_payload() -> None:
    a = {"alpha": 1}
    b = {"alpha": 2}
    assert canonical_request_hash(a) != canonical_request_hash(b)


@pytest.mark.asyncio
async def test_apply_idempotency_caches_first_response_and_replays() -> None:
    """First call executes the operation; second call returns the cached payload."""
    from sqlalchemy.ext.asyncio import AsyncSession

    payload = {"key": "value"}
    n_calls = 0

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal n_calls
        n_calls += 1
        return {"success": True, "result": n_calls}

    first = await apply_idempotency(
        tool_name="test_tool",
        idempotency_key="key-1",
        request_payload=payload,
        executor=run,
    )
    second = await apply_idempotency(
        tool_name="test_tool",
        idempotency_key="key-1",
        request_payload=payload,
        executor=run,
    )

    assert n_calls == 1, "second call must not re-execute"
    assert first["result"] == 1
    assert second["result"] == 1
    assert second["idempotency_replay"] is True


@pytest.mark.asyncio
async def test_replay_refreshes_metadata_trace_id() -> None:
    """Cached replays must reattach the *current* trace_id, not the cached one.

    Background: the cached payload was assembled during the first call
    and its ``_metadata.trace_id`` reflects whichever workflow id was
    bound then. A retry under a different workflow (or with no trace
    at all) must see its own id — otherwise distributed-tracing tools
    stitch the replay onto the original workflow and lose the actual
    causal chain.

    Reference: W3C trace-context recommends one trace id per logical
    workflow; replays of a stored response are *not* part of the
    original workflow. See https://www.w3.org/TR/trace-context/#trace-id.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    from bo_mcp_server.trace_context import bind_trace_id

    async def run(_session: AsyncSession) -> dict[str, Any]:
        # Simulate an operation that goes through ``with_response_metadata``
        # by attaching the metadata block directly. Real callers reach
        # this through :func:`response_formatter.attach_response_metadata`.
        from bo_mcp_server.response_formatter import attach_response_metadata

        return attach_response_metadata({"success": True})

    # First call: bind a workflow id so the cached payload carries it.
    with bind_trace_id("workflow-original"):
        first = await apply_idempotency(
            tool_name="metadata_replay",
            idempotency_key="meta-1",
            request_payload={"k": "v"},
            executor=run,
        )
    assert first["_metadata"]["trace_id"] == "workflow-original"

    # Replay under a *different* workflow id must echo the new id.
    with bind_trace_id("workflow-retry"):
        replay_new = await apply_idempotency(
            tool_name="metadata_replay",
            idempotency_key="meta-1",
            request_payload={"k": "v"},
            executor=run,
        )
    assert replay_new["idempotency_replay"] is True
    assert replay_new["_metadata"]["trace_id"] == "workflow-retry"

    # Replay with NO workflow bound must drop the field entirely so the
    # metadata envelope stays compact for one-off retries.
    replay_unbound = await apply_idempotency(
        tool_name="metadata_replay",
        idempotency_key="meta-1",
        request_payload={"k": "v"},
        executor=run,
    )
    assert replay_unbound["idempotency_replay"] is True
    assert "trace_id" not in replay_unbound["_metadata"]


@pytest.mark.asyncio
async def test_short_circuit_envelopes_carry_trace_metadata() -> None:
    """Conflict / in-progress / stale envelopes also echo ``_metadata.trace_id``.

    Cached successful replays already refresh the trace id via
    :func:`_refresh_response_metadata_trace_id`; the error-path
    short-circuits (payload conflict, in-flight reservation, stale
    owner) are built locally inside the idempotency module and never
    go through the operation-level
    ``with_response_metadata`` decorator. Without an explicit attach
    step those envelopes would be the only tool returns that drop the
    trace echo a debugging operator needs most.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    from bo_mcp_server.trace_context import bind_trace_id

    async def run(_session: AsyncSession) -> dict[str, Any]:
        return {"success": True}

    # Seed the cache with one payload, then retry with a mismatched
    # payload to trigger the conflict short-circuit.
    await apply_idempotency(
        tool_name="trace_short_circuit",
        idempotency_key="meta-conflict",
        request_payload={"k": "first"},
        executor=run,
    )
    with bind_trace_id("trace-conflict"):
        conflict = await apply_idempotency(
            tool_name="trace_short_circuit",
            idempotency_key="meta-conflict",
            request_payload={"k": "second"},
            executor=run,
        )
    # The envelope is an error response — under a bound trace it must
    # still echo the id under ``_metadata``.
    assert conflict.get("success") is False
    assert conflict.get("_metadata", {}).get("trace_id") == "trace-conflict"


@pytest.mark.asyncio
async def test_in_progress_envelope_carries_trace_metadata() -> None:
    """A second retry that hits the in-progress reservation echoes its own trace id.

    The first call holds the reservation via an ``asyncio.Event``
    barrier so the second call observes a pending row and gets the
    ``IDEMPOTENCY_IN_PROGRESS`` envelope. Under a bound trace that
    envelope must still echo ``_metadata.trace_id`` — without the
    attach step inside ``_lookup_to_short_circuit`` the in-progress
    short-circuit would be the only tool return that silently drops
    the workflow id.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    from bo_mcp_server.trace_context import bind_trace_id

    barrier = asyncio.Event()
    started = asyncio.Event()
    first_run = False

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal first_run
        if not first_run:
            first_run = True
            started.set()
            await barrier.wait()
        return {"success": True}

    async def winner() -> dict[str, Any]:
        return await apply_idempotency(
            tool_name="trace_in_progress",
            idempotency_key="meta-inflight",
            request_payload={"k": "v"},
            executor=run,
        )

    async def retry_with_trace() -> dict[str, Any]:
        await started.wait()
        with bind_trace_id("trace-in-progress"):
            result = await apply_idempotency(
                tool_name="trace_in_progress",
                idempotency_key="meta-inflight",
                request_payload={"k": "v"},
                executor=run,
            )
        barrier.set()
        return result

    _, retry_result = await asyncio.gather(winner(), retry_with_trace())

    # The retry observed either the in-progress envelope (the load-
    # bearing case for this test) or, if timing landed after
    # finalize, the cached replay. Both code paths run the metadata
    # attach / refresh, so either response must echo the bound id.
    assert retry_result.get("_metadata", {}).get("trace_id") == "trace-in-progress"


@pytest.mark.asyncio
async def test_stale_owner_envelope_carries_trace_metadata() -> None:
    """The stale-owner short-circuit also echoes ``_metadata.trace_id``.

    Mirrors ``test_stale_owner_session_aware_writes_roll_back`` but
    binds a workflow trace around the call so the resulting
    ``stale_owner`` envelope's metadata can be asserted. Without the
    attach step inside ``_run_session_aware`` this envelope would be
    the only path that silently drops the trace echo a debugging
    operator needs most.
    """
    from uuid import uuid4

    from sqlalchemy import update
    from sqlalchemy.ext.asyncio import AsyncSession

    from bo_mcp_server.idempotency import apply_idempotency
    from bo_mcp_server.storage.models import IdempotencyCacheModel
    from bo_mcp_server.trace_context import bind_trace_id

    tool = f"stale_owner_trace_{uuid4().hex[:8]}"
    key = "stale-owner-trace-key"

    async def slow_executor(db: AsyncSession) -> dict[str, Any]:
        # Model a concurrent reclaim by overwriting the reservation
        # token on the same session — finalize will see 0 affected
        # rows and the stale-owner branch fires.
        await db.execute(
            update(IdempotencyCacheModel)
            .where(
                IdempotencyCacheModel.tool_name == tool,
                IdempotencyCacheModel.idempotency_key == key,
            )
            .values(reservation_token="different-owner-token-32-chars-aa")  # noqa: S106
        )
        return {"success": True, "executed": True}

    with bind_trace_id("trace-stale-owner"):
        response = await apply_idempotency(
            tool_name=tool,
            idempotency_key=key,
            request_payload={"v": 1},
            executor=slow_executor,
        )

    assert response["success"] is False
    assert response["error"]["details"]["stale_owner"] is True
    assert response.get("_metadata", {}).get("trace_id") == "trace-stale-owner"


@pytest.mark.asyncio
async def test_apply_idempotency_conflict_on_payload_mismatch() -> None:
    """Reusing a key with a different payload returns a conflict envelope."""
    from sqlalchemy.ext.asyncio import AsyncSession

    async def run(_session: AsyncSession) -> dict[str, Any]:
        return {"success": True}

    await apply_idempotency(
        tool_name="test_tool",
        idempotency_key="key-2",
        request_payload={"value": 1},
        executor=run,
    )
    conflict = await apply_idempotency(
        tool_name="test_tool",
        idempotency_key="key-2",
        request_payload={"value": 2},  # Different payload, same key
        executor=run,
    )

    assert conflict["success"] is False
    # IDEMPOTENCY_CONFLICT (E015) is distinct from generic VALIDATION_FAILED
    # so clients can programmatically detect the key-reuse case.
    assert conflict["error"]["code"] == "E015"
    assert conflict["error"]["details"]["idempotency_conflict"] is True
    # Conflict envelopes must not be marked retryable — retrying the same
    # call with the same payload will keep colliding.
    assert conflict["error"]["retryable"] is False


@pytest.mark.asyncio
async def test_apply_idempotency_none_key_always_executes() -> None:
    """``None`` key disables caching — every call runs the operation."""
    from sqlalchemy.ext.asyncio import AsyncSession

    n_calls = 0

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal n_calls
        n_calls += 1
        return {"success": True}

    await apply_idempotency(
        tool_name="test_tool",
        idempotency_key=None,
        request_payload={"x": 1},
        executor=run,
    )
    await apply_idempotency(
        tool_name="test_tool",
        idempotency_key=None,
        request_payload={"x": 1},
        executor=run,
    )

    assert n_calls == 2


@pytest.mark.asyncio
async def test_apply_idempotency_drops_reservation_on_exception() -> None:
    """If the executor raises, the reservation is dropped — retries can win the race."""
    from sqlalchemy.ext.asyncio import AsyncSession

    n_calls = 0

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal n_calls
        n_calls += 1
        if n_calls == 1:
            msg = "simulated failure"
            raise RuntimeError(msg)
        return {"success": True, "attempt": n_calls}

    with pytest.raises(RuntimeError):
        await apply_idempotency(
            tool_name="test_tool",
            idempotency_key="key-retry",
            request_payload={"x": 1},
            executor=run,
        )

    result = await apply_idempotency(
        tool_name="test_tool",
        idempotency_key="key-retry",
        request_payload={"x": 1},
        executor=run,
    )

    assert result["attempt"] == 2
    # Not a replay — the failed first attempt dropped its reservation.
    assert result.get("idempotency_replay") is not True


@pytest.mark.asyncio
async def test_apply_idempotency_drops_reservation_on_cancelled_error() -> None:
    """An executor raising ``asyncio.CancelledError`` still frees the slot.

    ``CancelledError`` is a ``BaseException`` subclass, so the broad
    ``except Exception`` cleanup does not catch it. Before the fix the
    reservation stayed ``pending`` until its TTL and the immediate retry
    saw ``IDEMPOTENCY_IN_PROGRESS`` though nothing was running. The
    dedicated ``except asyncio.CancelledError`` clause must drop the slot
    so the retry executes.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    n_calls = 0

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal n_calls
        n_calls += 1
        if n_calls == 1:
            raise asyncio.CancelledError
        return {"success": True, "attempt": n_calls}

    with pytest.raises(asyncio.CancelledError):
        await apply_idempotency(
            tool_name="test_tool",
            idempotency_key="key-cancel",
            request_payload={"x": 1},
            executor=run,
        )

    result = await apply_idempotency(
        tool_name="test_tool",
        idempotency_key="key-cancel",
        request_payload={"x": 1},
        executor=run,
    )

    # The retry executed instead of returning IDEMPOTENCY_IN_PROGRESS.
    assert result["attempt"] == 2
    assert result.get("idempotency_replay") is not True


@pytest.mark.asyncio
async def test_apply_idempotency_drops_reservation_on_task_cancel() -> None:
    """A real ``task.cancel()`` mid-execution frees the slot for retry.

    Mirrors the production failure mode: an MCP client disconnects and the
    server cancels the in-flight tool task. The reservation must not be
    left ``pending``; a follow-up call with the same key must execute.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    started = asyncio.Event()
    n_calls = 0

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal n_calls
        n_calls += 1
        if n_calls == 1:
            started.set()
            # Block until cancelled by the outer task.cancel().
            await asyncio.sleep(3600)
        return {"success": True, "attempt": n_calls}

    task = asyncio.create_task(
        apply_idempotency(
            tool_name="test_tool",
            idempotency_key="key-task-cancel",
            request_payload={"x": 1},
            executor=run,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    result = await apply_idempotency(
        tool_name="test_tool",
        idempotency_key="key-task-cancel",
        request_payload={"x": 1},
        executor=run,
    )

    assert result["attempt"] == 2
    assert result.get("idempotency_replay") is not True


@pytest.mark.asyncio
async def test_double_cancellation_during_cleanup_still_drops_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second cancel while the reservation drop is awaiting must not leak the slot.

    The cancellation handler drops the reservation as a *shielded* task and
    awaits it to completion in ``finally`` even when its own wait is cancelled
    again (server shutdown, an impatient client). Without the shield the second
    cancel would interrupt the drop mid-flight and the slot would stay
    ``pending`` until its TTL — the exact failure M36 closes. We drive the
    race deterministically by blocking the drop on an event so the second
    cancel lands while it is in progress.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    import bo_mcp_server.idempotency as idem

    original_drop = idem._drop_reservation
    entered = asyncio.Event()
    release = asyncio.Event()
    dropped = asyncio.Event()

    async def blocking_drop(tool_name: str, idempotency_key: str, reservation_token: str) -> None:
        entered.set()
        await release.wait()
        await original_drop(tool_name, idempotency_key, reservation_token)
        dropped.set()

    monkeypatch.setattr(idem, "_drop_reservation", blocking_drop)

    started = asyncio.Event()
    n_calls = 0

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal n_calls
        n_calls += 1
        if n_calls == 1:
            started.set()
            await asyncio.sleep(3600)
        return {"success": True, "attempt": n_calls}

    task = asyncio.create_task(
        apply_idempotency(
            tool_name="test_tool",
            idempotency_key="key-double-cancel",
            request_payload={"x": 1},
            executor=run,
        )
    )
    await started.wait()
    task.cancel()  # first cancel → enters the shielded drop
    await entered.wait()
    task.cancel()  # second cancel → interrupts our shielded wait
    release.set()  # let the (shielded) drop finish
    with pytest.raises(asyncio.CancelledError):
        await task
    await dropped.wait()

    # The reservation was dropped despite the double cancel, so a retry runs.
    result = await apply_idempotency(
        tool_name="test_tool",
        idempotency_key="key-double-cancel",
        request_payload={"x": 1},
        executor=run,
    )
    assert result["attempt"] == 2
    assert result.get("idempotency_replay") is not True


@pytest.mark.asyncio
async def test_canonical_hash_accepts_large_payload() -> None:
    """Large payloads hash without raising — SHA256 is O(n) and cheap.

    Regression test for the review pass: the old 1 MiB
    safety cap raised :class:`ValueError` before the tool could return
    a structured envelope, which broke file uploads that bundled the
    raw CSV in the payload. The fix removes the cap and tools digest
    large fields up front (see ``digest_large_field``).
    """
    huge = "x" * (2 * 1024 * 1024)
    digest = canonical_request_hash({"blob": huge})
    assert len(digest) == 64  # SHA256 hex length


def test_digest_large_field_is_stable_and_size_independent() -> None:
    """Pre-digest helper produces a 64-char hex string regardless of input size."""
    short = digest_large_field("hello")
    long_value = digest_large_field("x" * (5 * 1024 * 1024))
    assert len(short) == 64
    assert len(long_value) == 64
    # Same input ⇒ same digest (deterministic / no random salt).
    assert digest_large_field("hello") == short
    # Different inputs ⇒ different digests.
    assert short != long_value


@pytest.mark.asyncio
async def test_apply_idempotency_keys_by_tool() -> None:
    """The same key under different tools does not collide."""
    from sqlalchemy.ext.asyncio import AsyncSession

    a_calls = 0
    b_calls = 0

    async def run_a(_session: AsyncSession) -> dict[str, Any]:
        nonlocal a_calls
        a_calls += 1
        return {"tool": "a", "n": a_calls}

    async def run_b(_session: AsyncSession) -> dict[str, Any]:
        nonlocal b_calls
        b_calls += 1
        return {"tool": "b", "n": b_calls}

    await apply_idempotency(
        tool_name="tool_a",
        idempotency_key="shared",
        request_payload={"x": 1},
        executor=run_a,
    )
    b_result = await apply_idempotency(
        tool_name="tool_b",
        idempotency_key="shared",
        request_payload={"x": 1},
        executor=run_b,
    )

    assert b_result["tool"] == "b"
    assert b_calls == 1
    # ``a`` was not invoked twice and remains cached under its own tool name.
    assert a_calls == 1


@pytest.mark.asyncio
async def test_stale_reservation_is_reclaimed_when_pending_ttl_elapses() -> None:
    """An abandoned pending reservation can be reclaimed once it expires.

    Regression test for the review-pass finding: previously a worker
    death between mutation-commit and cache-finalize would poison the
    slot for the full ``DEFAULT_IDEMPOTENCY_TTL_SECONDS`` (24 hours).
    The fix splits the row lifetime so a pending row gets a short TTL
    (``DEFAULT_RESERVATION_TTL_SECONDS``); once it expires, the next
    retry re-enters the reservation race and can win.

    Here we simulate the failure mode by:

    1. Issuing a call with a very short ``reservation_ttl_seconds``
       whose executor raises after delaying past the TTL — except we
       cheat with a sleep that is shorter than the TTL but the second
       call asks for an even shorter TTL. The cleaner approach is to
       just leak a pending row directly with the cache helpers and
       verify the reclaim, which is what we do below.
    """
    from bo_mcp_server.idempotency import _read_existing, _try_reserve

    tool = "stale_tool"
    key = "stale-key"
    request_hash = canonical_request_hash({"x": 1})

    # Pretend a previous call crashed after reserving with a 1 ms
    # pending TTL: the reservation row is in the DB, response_json="",
    # expires_at already past.
    token = await _try_reserve(tool, key, request_hash, reservation_ttl_seconds=0)
    assert token is not None
    assert len(token) == 32

    # Sanity-check: the row is "pending" the instant we insert it.
    # (We can't easily observe that without racing the purge, so we
    # focus on the post-expiry behaviour.)

    # The next read should treat the row as expired (purge + miss).
    lookup = await _read_existing(tool, key, request_hash, ttl_seconds=60)
    assert lookup.cached_response is None
    assert lookup.conflict_response is None
    assert lookup.in_progress is False

    # And a retry can win a *fresh* reservation race for the same key.
    from sqlalchemy.ext.asyncio import AsyncSession

    n_calls = 0

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal n_calls
        n_calls += 1
        return {"success": True, "attempt": n_calls}

    result = await apply_idempotency(
        tool_name=tool,
        idempotency_key=key,
        request_payload={"x": 1},
        executor=run,
    )
    assert result["success"] is True
    assert result["attempt"] == 1
    assert n_calls == 1


@pytest.mark.asyncio
async def test_stale_finalize_does_not_overwrite_newer_reservation() -> None:
    """A reservation-token mismatch on finalize never touches the live row.

    Regression test for the Medium review-pass finding: previously
    ``_finalize_reservation`` matched only ``(tool, key, request_hash)``,
    so an original operation whose reservation was reclaimed by a newer
    retry (same payload) would silently overwrite the newer row's
    response on its late finalize. The token guarantees that only the
    original owner can complete its row.
    """
    from sqlalchemy import delete, select

    from bo_mcp_server.idempotency import _finalize_reservation, _try_reserve
    from bo_mcp_server.storage import get_session
    from bo_mcp_server.storage.models import IdempotencyCacheModel

    tool = "stale_finalize_tool"
    key = "stale-finalize-key"
    request_hash = canonical_request_hash({"x": 1})

    # 1) Original operation reserves with token_A and is "slow".
    token_a = await _try_reserve(tool, key, request_hash, reservation_ttl_seconds=0)
    assert token_a is not None

    # 2) Reservation expires; a retry purges + reclaims with token_B.
    async with get_session() as session:
        await session.execute(
            delete(IdempotencyCacheModel).where(
                IdempotencyCacheModel.tool_name == tool,
                IdempotencyCacheModel.idempotency_key == key,
            )
        )
    token_b = await _try_reserve(tool, key, request_hash, reservation_ttl_seconds=60)
    assert token_b is not None
    assert token_b != token_a

    # 3) The original operation finally finishes and tries to finalize
    #    with its stale token. The mismatch must produce a no-op.
    await _finalize_reservation(
        tool_name=tool,
        key=key,
        request_hash=request_hash,
        reservation_token=token_a,
        response={"success": True, "from": "original"},
        ttl_seconds=24 * 60 * 60,
    )

    # 4) The newer row should still carry the pending sentinel — the
    #    stale finalize was rejected because tokens didn't match.
    async with get_session() as session:
        result = await session.execute(
            select(IdempotencyCacheModel).where(
                IdempotencyCacheModel.tool_name == tool,
                IdempotencyCacheModel.idempotency_key == key,
            )
        )
        row = result.scalar_one()
        assert row.reservation_token == token_b
        assert row.response_json == "", (
            "Stale token finalize must not overwrite the newer reservation"
        )


@pytest.mark.asyncio
async def test_stale_owner_session_aware_writes_roll_back() -> None:
    """A session-aware executor that loses its reservation must not commit writes.

    Regression test for the High review-pass finding: previously,
    ``_run_session_aware`` would log a warning when
    ``finalize_reservation_in_session`` returned 0 affected rows, but
    still let the surrounding ``async with get_session()`` commit the
    executor's writes. A slow operation that ran past its pending
    reservation TTL could therefore commit its DB writes even though
    the cache slot had been reclaimed by a concurrent retry, producing
    duplicate side effects.

    The fix raises ``_StaleReservationError`` from inside the session
    block, which triggers ``get_session``'s rollback handler and
    discards the executor's writes. The caller receives a retryable
    ``IDEMPOTENCY_IN_PROGRESS`` envelope with ``details.stale_owner=
    True``.

    Test scenario:

    1. Caller reserves the slot; the executor receives that session.
    2. The executor writes a sentinel ``UserModel`` row on the supplied
       session — that row stands in for any production-grade side
       effect (campaign, suggestion, result row).
    3. The executor then UPDATEs the reservation row's
       ``reservation_token`` on the *same* session, modelling a
       concurrent retry that reclaimed the slot. Doing the race on the
       same session avoids the SQLite-in-memory ``StaticPool`` quirk
       where a sibling session's commit would prematurely persist the
       outer session's pending writes — production uses Postgres with
       independent connections, where this is not a concern.
    4. The executor returns a successful response.
    5. ``apply_idempotency``'s finalize matches on the original token
       and affects 0 rows. The stale-ownership branch raises, the
       session rolls back, the sentinel ``User`` row never lands, and
       the caller gets the structured envelope.
    """
    from uuid import uuid4

    from sqlalchemy import select, update
    from sqlalchemy.ext.asyncio import AsyncSession

    from bo_mcp_server.domain import User
    from bo_mcp_server.domain.utils import utcnow
    from bo_mcp_server.idempotency import apply_idempotency
    from bo_mcp_server.storage import UserRepository, get_session
    from bo_mcp_server.storage.models import IdempotencyCacheModel, UserModel

    tool = "stale_owner_tool"
    key = "stale-owner-key"
    sentinel_user_id = uuid4()
    sentinel_email = f"stale-{uuid4().hex}@example.test"

    async def slow_executor(db: AsyncSession) -> dict[str, Any]:
        # 1) Write a sentinel User row on the supplied session. If the
        #    rollback works, this row will never reach the DB.
        user_repo = UserRepository(db)
        await user_repo.save(
            User(
                id=sentinel_user_id,
                name="Stale Owner Sentinel",
                email=sentinel_email,
                api_key_hash="stale-owner-hash",
                created_at=utcnow(),
            )
        )

        # 2) Race: model the concurrent reclaim by overwriting the
        #    reservation token via the same session. In production a
        #    different connection would have already committed the new
        #    token; the same-session model produces the *same* outcome
        #    (finalize matches 0 rows) without exposing the test to the
        #    sibling-session-on-StaticPool corner case.
        await db.execute(
            update(IdempotencyCacheModel)
            .where(
                IdempotencyCacheModel.tool_name == tool,
                IdempotencyCacheModel.idempotency_key == key,
            )
            .values(reservation_token="different-owner-token-32-chars-aa")  # noqa: S106
        )

        return {"success": True, "executed": True}

    response = await apply_idempotency(
        tool_name=tool,
        idempotency_key=key,
        request_payload={"v": 1},
        executor=slow_executor,
    )

    # The retryable in-progress envelope is returned.
    assert response["success"] is False
    assert response["error"]["code"] == "E014"
    assert response["error"]["details"]["stale_owner"] is True

    # The executor's sentinel row must NOT have committed.
    async with get_session() as session:
        result = await session.execute(
            select(UserModel).where(UserModel.id == str(sentinel_user_id))
        )
        assert result.scalar_one_or_none() is None, (
            "Stale-owner session-aware writes must be rolled back; "
            "otherwise a slow operation can commit duplicate side effects."
        )


@pytest.mark.asyncio
async def test_session_aware_executor_finalizes_in_same_session() -> None:
    """The session-aware path delivers a usable session to the executor.

    The contract is "the executor writes its side effect on the session
    handed in, and ``apply_idempotency`` finalizes the cache row on the
    same session before commit". This test verifies the session is
    actually passed through and the cache row carries the executor's
    response after the call returns.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession

    from bo_mcp_server.idempotency import apply_idempotency
    from bo_mcp_server.storage import get_session
    from bo_mcp_server.storage.models import IdempotencyCacheModel

    observed: list[AsyncSession] = []

    async def session_aware(db: AsyncSession) -> dict[str, Any]:
        observed.append(db)
        return {"success": True, "value": "via-session"}

    response = await apply_idempotency(
        tool_name="session_aware_tool",
        idempotency_key="sa-key-1",
        request_payload={"x": 1},
        executor=session_aware,
    )

    assert response == {"success": True, "value": "via-session"}
    assert len(observed) == 1

    # Replay: same key + payload returns the cached response with the
    # replay flag set, AND the row carries the original response.
    replay = await apply_idempotency(
        tool_name="session_aware_tool",
        idempotency_key="sa-key-1",
        request_payload={"x": 1},
        executor=session_aware,
    )
    assert replay["idempotency_replay"] is True
    assert replay["value"] == "via-session"
    # And the executor did not run again.
    assert len(observed) == 1

    # The row really has the response committed.
    async with get_session() as session:
        result = await session.execute(
            select(IdempotencyCacheModel).where(
                IdempotencyCacheModel.tool_name == "session_aware_tool",
                IdempotencyCacheModel.idempotency_key == "sa-key-1",
            )
        )
        row = result.scalar_one()
        assert "via-session" in row.response_json


@pytest.mark.asyncio
async def test_session_aware_executor_failure_rolls_back_writes() -> None:
    """When the session-aware executor raises, its writes and the cache row both roll back.

    This is the durability property the review pass asked for: a
    process death (modelled here as an exception) between the
    operation's commit and the cache finalize cannot leave the
    operation persisted with the cache row stuck pending. With the
    session-aware path, both the operation writes and the finalize ride
    a single transaction, so a raise rolls back everything.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession

    from bo_mcp_server.idempotency import apply_idempotency
    from bo_mcp_server.storage import get_session
    from bo_mcp_server.storage.models import IdempotencyCacheModel

    async def crasher(db: AsyncSession) -> dict[str, Any]:
        _ = db
        msg = "operation died mid-flight"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError):
        await apply_idempotency(
            tool_name="crashy_tool",
            idempotency_key="crash-key",
            request_payload={"x": 1},
            executor=crasher,
        )

    # The reservation must have been dropped (not stuck pending) so a
    # future retry can win the race instead of seeing E014 for 10
    # minutes.
    async with get_session() as session:
        result = await session.execute(
            select(IdempotencyCacheModel).where(
                IdempotencyCacheModel.tool_name == "crashy_tool",
                IdempotencyCacheModel.idempotency_key == "crash-key",
            )
        )
        assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_concurrent_retries_execute_once() -> None:
    """Two simultaneous retries with the same key produce exactly one side effect.

    Regression test for the lookup-then-execute-then-store race that
    motivated the reservation refactor: without the unique-key
    reservation insert, both callers would miss the cache, both would
    execute, and both would commit duplicate state.

    The executor sleeps briefly to widen the race window so the test
    reliably observes the in-progress branch.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    execute_count = 0
    barrier = asyncio.Event()
    started = asyncio.Event()
    first_started = False

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal execute_count, first_started
        execute_count += 1
        if not first_started:
            first_started = True
            started.set()
            # Hold the reservation until the other call has had a chance
            # to observe it. The barrier is released by the test once
            # the in-progress branch has been hit.
            await barrier.wait()
        return {"success": True, "executed": execute_count}

    async def call() -> dict[str, Any]:
        return await apply_idempotency(
            tool_name="concurrent_tool",
            idempotency_key="shared-key",
            request_payload={"v": 1},
            executor=run,
        )

    async def release_after_observation() -> dict[str, Any]:
        # Wait until the winner has entered the executor, then make one
        # more attempt. With the reservation in place, this must observe
        # the in-progress envelope.
        await started.wait()
        result = await apply_idempotency(
            tool_name="concurrent_tool",
            idempotency_key="shared-key",
            request_payload={"v": 1},
            executor=run,
        )
        # Release the winner so the gather can finish.
        barrier.set()
        return result

    winner_result, runner_up_result = await asyncio.gather(call(), release_after_observation())

    assert execute_count == 1, "executor must run exactly once across concurrent retries"
    assert winner_result["success"] is True
    assert winner_result["executed"] == 1
    # The second concurrent call either saw the in-progress envelope or,
    # if the timing landed after finalize, the cached replay. Both are
    # acceptable — the load-bearing assertion is the single execution.
    assert runner_up_result["success"] is False or (
        runner_up_result.get("idempotency_replay") is True
    )
    if runner_up_result["success"] is False:
        assert runner_up_result["error"]["code"] == "E014"
        assert runner_up_result["error"]["details"]["idempotency_in_progress"] is True

    # After both calls finish, a third retry sees the cached response.
    third = await apply_idempotency(
        tool_name="concurrent_tool",
        idempotency_key="shared-key",
        request_payload={"v": 1},
        executor=run,
    )
    assert third["idempotency_replay"] is True
    assert execute_count == 1


@pytest.mark.asyncio
async def test_apply_idempotency_does_not_cache_retryable_error_envelopes() -> None:
    """Retryable error envelopes (any code, not just CONCURRENT_MODIFICATION)
    must not be cached — caching would block retries for the full 24h TTL.

    Pre-fix: ``_is_transient_error`` only special-cased
    ``CONCURRENT_MODIFICATION`` (E010). A ``BACKEND_TRANSIENT_ERROR``
    (E105) would therefore be finalized
    into the cache and every retry would replay the failure instead
    of re-executing the operation against the (now-recovered) backend.

    Post-fix: the cache consults the envelope's ``retryable`` flag
    (set by ``make_error_response`` from the central
    ``ERROR_CODE_RETRY_HINTS`` table) so every newly-added retryable
    code automatically skips finalization without re-editing
    ``idempotency.py``.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    from bo_mcp_server.errors import ErrorCode, make_error_response

    execute_count = 0

    async def flaky(_session: AsyncSession) -> dict[str, Any]:
        nonlocal execute_count
        execute_count += 1
        # First call: transient backend flake. Second call: success.
        if execute_count == 1:
            return make_error_response(
                ErrorCode.BACKEND_TRANSIENT_ERROR,
                message="optimizer flake (run 1)",
            )
        return {"success": True, "executed": execute_count}

    first = await apply_idempotency(
        tool_name="flaky_tool",
        idempotency_key="flaky-key",
        request_payload={"v": 1},
        executor=flaky,
    )
    assert first["success"] is False
    assert first["error"]["code"] == ErrorCode.BACKEND_TRANSIENT_ERROR.value
    assert first["error"]["retryable"] is True

    # The retry must re-execute (not replay the cached failure).
    second = await apply_idempotency(
        tool_name="flaky_tool",
        idempotency_key="flaky-key",
        request_payload={"v": 1},
        executor=flaky,
    )
    assert second["success"] is True
    assert second["executed"] == 2, "executor must run again on retry, not replay"
    assert execute_count == 2
