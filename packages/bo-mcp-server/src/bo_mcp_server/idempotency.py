"""Idempotency-key support for state-mutating MCP tools.

Agents retry mutating MCP calls after timeouts or transient network
errors, and without server-side deduplication those retries silently
duplicate state (a second campaign, a second result row, a second
suggestion-status flip). The MCP spec leaves the retry policy to the
client, so the safe path is server-side: each mutating tool accepts an
optional ``idempotency_key``; on the first call the server persists the
response keyed by ``(tool_name, idempotency_key)``; on later calls with
the same key the server returns the cached response instead of
re-executing.

Atomicity / race-safety (TODO 1.46 review pass)
-----------------------------------------------

A naive look-up-then-execute-then-store pattern is racy: two concurrent
retries with the same key both miss the cache, both run the executor,
and both commit duplicate side effects before either cache row is
written. This module uses a three-step reservation pattern instead:

1. **Reserve.** Try to ``INSERT`` a placeholder row (``response_json=
   ""``) in its own short transaction. The primary key
   ``(tool_name, idempotency_key)`` is unique, so exactly one caller
   wins; every other concurrent retry sees an :class:`IntegrityError`
   and falls through to a re-read.

2. **Execute.** The reservation winner runs the operation in its own
   transaction. The reservation row is already committed, so any other
   retry that arrives during execution sees the placeholder and gets a
   structured :class:`ErrorCode.IDEMPOTENCY_IN_PROGRESS` envelope.

3. **Finalize.** On success, ``UPDATE`` the reservation row with the
   actual response. On executor failure, ``DELETE`` the reservation so
   a future retry can win the race instead of being blocked forever.

The implementation runs every cache touch in a dedicated short-lived
session (separate from the operation's session) so the cache write
order does not depend on the operation's transaction lifetime. The
operation's commit is still distinct from the cache finalize, but the
reservation row already prevents duplicate execution before the
operation runs — that is the load-bearing invariant.

Conflict handling: re-using the same key with a different request
payload is a client bug. We surface it as
``ErrorCode.VALIDATION_FAILED`` with ``details.idempotency_conflict=
True`` so the agent can resolve the key collision or fetch the prior
response.

This module is transport-neutral; both MCP tools and the REST routes
can reuse the same store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.domain.utils import utcnow
from bo_mcp_server.errors import ErrorCode, make_error_response, retry_hint_for
from bo_mcp_server.storage import get_session
from bo_mcp_server.storage.models import IdempotencyCacheModel

ToolExecutor = Callable[[AsyncSession], Awaitable[dict[str, Any]]]
"""Mutating-tool executor signature.

Receives the :class:`AsyncSession` opened by :func:`apply_idempotency`
and uses it for every DB write. The cache finalize runs on that same
session before commit so the side effect and the cached response
commit atomically — closing the post-commit/pre-finalize window that a
zero-arg executor would otherwise leave open.
"""

logger = logging.getLogger(__name__)

# 24 hours. Chosen to cover same-day agent retry windows without
# letting the cache grow unbounded. Override via the constructor only
# in tests.
DEFAULT_IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60

# How long a reservation can stay in the "pending" state before a
# retrying caller is allowed to reclaim the slot. Without this bound, a
# worker that dies between mutation-commit and cache-finalize would
# poison the slot for the entire ``DEFAULT_IDEMPOTENCY_TTL_SECONDS``
# window — every retry would see ``E014 IDEMPOTENCY_IN_PROGRESS`` until
# the 24h TTL elapsed, at which point retries could finally execute
# again and *duplicate* the mutation. With a short pending lifetime
# (10 min, well above any normal BO operation), abandoned reservations
# clear within minutes and retries can re-enter the reservation race.
#
# Trade-off: an operation that genuinely runs longer than this timeout
# may have its slot reclaimed by a concurrent retry — leading to two
# executions of the same logical call. Tune up only if you intentionally
# run very slow operations and prefer a longer block to a possible
# duplicate. Document the choice for callers.
DEFAULT_RESERVATION_TTL_SECONDS = 10 * 60

# Sentinel stored in ``response_json`` while the reservation winner is
# executing the operation. Distinguishable from any real JSON response
# because no JSON document is ever the empty string.
_PENDING_SENTINEL = ""


def _ensure_aware(value: datetime) -> datetime:
    """Coerce a naive datetime into a UTC-aware one.

    SQLite's ``DateTime`` column type drops the tzinfo on round-trip, so
    rows we wrote with ``utcnow()`` come back naive even though they
    were stored as UTC. PostgreSQL (the production target) preserves
    tzinfo. Normalizing here lets comparisons with ``utcnow()`` work in
    both environments without special-casing the driver.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass
class IdempotencyLookup:
    """Outcome of an idempotency cache lookup.

    Attributes:
        cached_response: A cached response dict to return instead of
            executing, or ``None`` when no usable cache entry exists.
        conflict_response: An error envelope to return when the supplied
            key was reused with a different payload, or ``None``
            otherwise. When set, the caller must short-circuit with this
            response and must not execute the operation.
        in_progress: True iff a reservation row exists with the same
            payload hash but no response yet — another retry holds the
            slot. The caller must short-circuit with the in-progress
            envelope.
    """

    cached_response: dict[str, Any] | None = None
    conflict_response: dict[str, Any] | None = None
    in_progress: bool = False


def canonical_request_hash(payload: dict[str, Any]) -> str:
    """Hash a tool call payload deterministically.

    Sorted keys + UTF-8 + SHA256 produces the same digest for two calls
    with the same logical arguments regardless of dict insertion order.
    Non-JSON-serializable values are coerced via ``default=str`` so a
    Pydantic model instance, a UUID, or a datetime do not crash the
    hash; the trade-off is that two distinct objects whose ``str``
    representations match are treated as equivalent, which is fine here
    because the upstream tool layer has already validated the payload
    against its typed schema.

    No size cap is applied: SHA256 is O(n) and even multi-megabyte
    payloads hash in milliseconds. Tools that carry genuinely large
    fields (e.g. CSV uploads) should still pre-digest those fields
    before building the idempotency payload (see
    :func:`digest_large_field`) so the cache row stays small.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def digest_large_field(value: str) -> str:
    """Return a stable, compact SHA256 digest of a large string field.

    Use when an otherwise-large field (a CSV file, a long string blob)
    needs to participate in the idempotency hash without inflating the
    cached response payload. Pre-digesting:

    - Keeps the canonical payload small enough to round-trip through the
      cache table.
    - Still detects payload mismatch — two distinct file contents
      collide only on a SHA256 collision, which is computationally
      infeasible for the lifetimes this cache covers.

    The digest is the literal SHA256 hex of the UTF-8 bytes — no salt,
    no prefix — so the caller can compute the same value client-side
    when debugging idempotency conflicts.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _in_progress_envelope(tool_name: str, key: str) -> dict[str, Any]:
    return make_error_response(
        ErrorCode.IDEMPOTENCY_IN_PROGRESS,
        message=(
            f"Operation {tool_name} with idempotency_key {key!r} is still "
            "executing; retry shortly to receive the cached response."
        ),
        details={
            "idempotency_in_progress": True,
            "tool_name": tool_name,
            "idempotency_key": key,
        },
    )


def _conflict_envelope(tool_name: str, key: str) -> dict[str, Any]:
    """Build the structured envelope for an idempotency-key collision.

    Uses :class:`ErrorCode.IDEMPOTENCY_CONFLICT` so callers can
    programmatically distinguish the key-reuse case from generic
    validation failures. ``details.idempotency_conflict=True`` is
    retained for backward compatibility with existing clients that
    grep the details field.
    """
    return make_error_response(
        ErrorCode.IDEMPOTENCY_CONFLICT,
        message=(
            f"idempotency_key {key!r} was previously used by "
            f"{tool_name} with a different payload; refusing to "
            "return a stale response. Either re-issue the call "
            "with a fresh idempotency_key or look up the prior "
            "response shape."
        ),
        details={
            "idempotency_conflict": True,
            "tool_name": tool_name,
            "idempotency_key": key,
        },
    )


def _project_row(
    row: IdempotencyCacheModel | None,
    tool_name: str,
    key: str,
    request_hash: str,
) -> IdempotencyLookup:
    """Classify a fetched cache row into the appropriate lookup outcome.

    Split out so :func:`_read_existing` can return the result of one
    decision rather than threading seven independent ``return``
    statements through the lookup body.
    """
    if row is None or _ensure_aware(row.expires_at) <= utcnow():
        # Either no row, or it expired between the purge and the
        # select. Treat as a miss; the next reservation attempt will
        # replace it.
        return IdempotencyLookup()

    if row.request_hash != request_hash:
        return IdempotencyLookup(conflict_response=_conflict_envelope(tool_name, key))

    if row.response_json == _PENDING_SENTINEL:
        return IdempotencyLookup(in_progress=True)

    try:
        cached = json.loads(row.response_json)
    except json.JSONDecodeError:
        logger.exception("Corrupted idempotency cache row for %s/%s; ignoring", tool_name, key)
        return IdempotencyLookup()

    if not isinstance(cached, dict):
        logger.error(
            "Unexpected non-dict idempotency cache row for %s/%s; ignoring",
            tool_name,
            key,
        )
        return IdempotencyLookup()

    return IdempotencyLookup(cached_response=cached)


async def _read_existing(
    tool_name: str,
    key: str,
    request_hash: str,
    ttl_seconds: int,
) -> IdempotencyLookup:
    """Inspect the cache row for ``(tool_name, key)``.

    Runs in its own session because the surrounding operation owns its
    own transaction; mixing reads across sessions keeps both isolated
    from each other.
    """
    _ = ttl_seconds  # TTL is enforced on write; read just respects expires_at
    async with get_session() as session:
        # Lazy GC of an expired row for the specific key being queried.
        # Scoping the purge to one row keeps it O(1) and avoids a global
        # scan that would contend with concurrent writers.
        await session.execute(
            delete(IdempotencyCacheModel).where(
                IdempotencyCacheModel.tool_name == tool_name,
                IdempotencyCacheModel.idempotency_key == key,
                IdempotencyCacheModel.expires_at <= utcnow(),
            )
        )

        result = await session.execute(
            select(IdempotencyCacheModel).where(
                IdempotencyCacheModel.tool_name == tool_name,
                IdempotencyCacheModel.idempotency_key == key,
            )
        )
        row = result.scalar_one_or_none()
        # Project inside the session: ``expire_on_commit=True`` expires
        # every ORM attribute on the way out of this block, so reading
        # ``row.expires_at`` after the ``async with`` would raise
        # ``DetachedInstanceError``.
        return _project_row(row, tool_name, key, request_hash)


async def _try_reserve(
    tool_name: str,
    key: str,
    request_hash: str,
    reservation_ttl_seconds: int,
) -> str | None:
    """Try to insert a placeholder row.

    Returns the freshly-minted reservation token on success. ``None``
    means another caller already inserted a row for
    ``(tool_name, key)``. The fresh session is critical: a unique-key
    violation on a shared session would corrupt the surrounding
    transaction.

    The token is a UUID4 string. It is matched by
    :func:`_finalize_reservation` and :func:`_drop_reservation` so a
    slow operation whose reservation was already reclaimed by a
    concurrent retry cannot accidentally finalize over (or delete) the
    retry's newer row.

    ``reservation_ttl_seconds`` is the *pending* lifetime: if the
    reservation winner never finalizes (process death, network drop),
    the row expires after this window and a retry can re-enter the
    race. The full response TTL is applied by
    :func:`_finalize_reservation` once the operation succeeds.
    """
    now = utcnow()
    token = uuid.uuid4().hex
    try:
        async with get_session() as session:
            session.add(
                IdempotencyCacheModel(
                    tool_name=tool_name,
                    idempotency_key=key,
                    request_hash=request_hash,
                    reservation_token=token,
                    response_json=_PENDING_SENTINEL,
                    created_at=now,
                    expires_at=now + timedelta(seconds=reservation_ttl_seconds),
                )
            )
            await session.flush()  # IntegrityError surfaces here
    except IntegrityError:
        return None
    return token


def _finalize_statement(
    tool_name: str,
    key: str,
    request_hash: str,
    reservation_token: str,
    response: dict[str, Any],
    ttl_seconds: int,
) -> Any:
    """Build the parameterised UPDATE used by both finalize helpers.

    Centralising the statement keeps the ``WHERE`` clause in one place
    so the token + hash predicates cannot drift between the
    same-session and own-session finalize paths.
    """
    now = utcnow()
    return (
        update(IdempotencyCacheModel)
        .where(
            IdempotencyCacheModel.tool_name == tool_name,
            IdempotencyCacheModel.idempotency_key == key,
            IdempotencyCacheModel.request_hash == request_hash,
            IdempotencyCacheModel.reservation_token == reservation_token,
        )
        .values(
            response_json=json.dumps(response, default=str),
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
    )


async def _finalize_reservation(
    tool_name: str,
    key: str,
    request_hash: str,
    reservation_token: str,
    response: dict[str, Any],
    ttl_seconds: int,
) -> None:
    """Promote the placeholder row into a finalized response.

    Two changes happen atomically: ``response_json`` gets the actual
    response, and ``expires_at`` is extended from the short reservation
    TTL to the full response TTL. The WHERE clause includes both
    ``request_hash`` *and* ``reservation_token`` so a slow operation
    whose reservation was already reclaimed by a concurrent retry
    cannot accidentally overwrite that retry's row. When the token
    fails to match, ``rowcount`` is 0 and we log a stale-reservation
    warning instead of touching the newer row.
    """
    async with get_session() as session:
        result = await session.execute(
            _finalize_statement(
                tool_name, key, request_hash, reservation_token, response, ttl_seconds
            )
        )
    if result.rowcount == 0:  # ty: ignore[unresolved-attribute]
        logger.warning(
            "Stale reservation finalize ignored for %s/%s (token=%s)",
            tool_name,
            key,
            reservation_token,
        )


async def finalize_reservation_in_session(
    session: AsyncSession,
    tool_name: str,
    key: str,
    request_hash: str,
    reservation_token: str,
    response: dict[str, Any],
    ttl_seconds: int,
) -> int:
    """Run the finalize UPDATE inside the caller's session.

    The session-aware path of :func:`apply_idempotency` uses this so the
    cache row commits atomically with the operation's writes. Returns
    the affected ``rowcount`` so the caller can log a stale-reservation
    warning if the token failed to match. The session is not committed
    here — the caller controls commit boundary.
    """
    result = await session.execute(
        _finalize_statement(tool_name, key, request_hash, reservation_token, response, ttl_seconds)
    )
    return int(result.rowcount or 0)  # ty: ignore[unresolved-attribute]


async def _drop_reservation(
    tool_name: str,
    key: str,
    reservation_token: str,
) -> None:
    """Delete the placeholder row owned by ``reservation_token``.

    Matching on the token prevents a stale owner (whose reservation
    expired and was reclaimed) from deleting the newer reservation. If
    ``rowcount`` is 0, the token no longer matches and we log instead
    of touching the row.
    """
    async with get_session() as session:
        result = await session.execute(
            delete(IdempotencyCacheModel).where(
                IdempotencyCacheModel.tool_name == tool_name,
                IdempotencyCacheModel.idempotency_key == key,
                IdempotencyCacheModel.reservation_token == reservation_token,
            )
        )
    if result.rowcount == 0:  # ty: ignore[unresolved-attribute]
        logger.warning(
            "Stale reservation drop ignored for %s/%s (token=%s)",
            tool_name,
            key,
            reservation_token,
        )


@dataclass
class _ReservationOutcome:
    """Internal: the result of the lookup + reserve phase.

    Returned to the caller of :func:`_reserve_or_short_circuit` so the
    two execution paths (own-session and session-aware) share the same
    pre-execution machinery.
    """

    short_circuit: dict[str, Any] | None = None
    reservation_token: str | None = None


def _lookup_to_short_circuit(
    lookup: IdempotencyLookup,
    tool_name: str,
    idempotency_key: str,
    *,
    pending_short_circuits: bool,
) -> dict[str, Any] | None:
    """Convert a lookup result to the envelope that should short-circuit.

    Centralises the cached / conflict / in-progress dispatch used by
    both the initial cache read and the post-reservation re-read in
    :func:`_reserve_or_short_circuit`. ``pending_short_circuits``
    controls the in-progress branch: ``True`` for the initial lookup
    (an existing pending row blocks us); on the post-reservation re-
    read it is also ``True`` (the row we see belongs to the winner).
    Returns ``None`` when there is no usable response and the caller
    should proceed to reserve.
    """
    if lookup.conflict_response is not None:
        return _attach_metadata(lookup.conflict_response)
    if lookup.cached_response is not None:
        replay = {**lookup.cached_response, "idempotency_replay": True}
        _refresh_response_metadata_trace_id(replay)
        return replay
    if pending_short_circuits and lookup.in_progress:
        return _attach_metadata(_in_progress_envelope(tool_name, idempotency_key))
    return None


def _attach_metadata(response: dict[str, Any]) -> dict[str, Any]:
    """Splice ``_metadata`` (including any active ``trace_id``) onto an envelope.

    Idempotency short-circuit envelopes (conflict, in-progress,
    stale-owner) are built locally by ``make_error_response`` and never
    pass through the operation-level
    :func:`bo_mcp_server.response_formatter.with_response_metadata`
    decorator, so without an explicit attach step they drop the
    ``_metadata.trace_id`` echo the cookbook promises for *every* tool
    return. Lazy import avoids the
    ``idempotency → response_formatter → idempotency`` cycle that the
    heavy imports inside ``response_formatter`` would otherwise force.
    """
    from bo_mcp_server.response_formatter import (  # noqa: PLC0415
        attach_response_metadata,
    )

    return attach_response_metadata(response)


def _refresh_response_metadata_trace_id(response: dict[str, Any]) -> None:
    """Reattach the current ``trace_id`` to a cached idempotency replay.

    The cached payload was assembled during the FIRST call and the
    ``_metadata.trace_id`` field reflects whatever workflow id was
    bound *then*. A retry from a different workflow (or with no trace
    at all) must see its own id — or the absence of the key — so
    distributed-tracing tools don't stitch the replay back onto the
    original workflow. Backend / protocol / server_version fields stay
    cached because they are stable across the retry window.

    Local imports avoid the
    ``idempotency → response_formatter → idempotency`` import cycle
    (response_formatter pulls in :mod:`bo_mcp_server.backend`, which
    transitively imports operations that depend on this module).
    """
    from bo_mcp_server.trace_context import get_trace_id  # noqa: PLC0415

    metadata = response.get("_metadata")
    if not isinstance(metadata, dict):
        return
    trace_id = get_trace_id()
    if trace_id is None:
        metadata.pop("trace_id", None)
    else:
        metadata["trace_id"] = trace_id


async def _reserve_or_short_circuit(
    tool_name: str,
    idempotency_key: str,
    request_hash: str,
    ttl_seconds: int,
    reservation_ttl_seconds: int,
) -> _ReservationOutcome:
    """Look up the cache; if no usable row, try to reserve atomically.

    Splits out so the own-session and session-aware execution paths in
    :func:`apply_idempotency` share the identical lookup + reservation
    semantics. Returns either:

    - ``short_circuit`` envelope (cached / conflict / in-progress), in
      which case the caller must return it without running the
      executor.
    - ``reservation_token``, meaning we hold the slot and the caller
      must execute the operation, then finalize with this token.
    """
    lookup = await _read_existing(tool_name, idempotency_key, request_hash, ttl_seconds)
    short_circuit = _lookup_to_short_circuit(
        lookup, tool_name, idempotency_key, pending_short_circuits=True
    )
    if short_circuit is not None:
        return _ReservationOutcome(short_circuit=short_circuit)

    token = await _try_reserve(tool_name, idempotency_key, request_hash, reservation_ttl_seconds)
    if token is not None:
        return _ReservationOutcome(reservation_token=token)

    # Lost the reservation race; re-read to figure out whether the
    # winner is done, mid-flight, or running a different payload.
    lookup = await _read_existing(tool_name, idempotency_key, request_hash, ttl_seconds)
    short_circuit = _lookup_to_short_circuit(
        lookup, tool_name, idempotency_key, pending_short_circuits=False
    )
    if short_circuit is None:
        short_circuit = _attach_metadata(_in_progress_envelope(tool_name, idempotency_key))
    return _ReservationOutcome(short_circuit=short_circuit)


@asynccontextmanager
async def session_scope(session: AsyncSession | None):
    """Provide a session: re-use the caller's if given, else open a new one.

    Operation-layer callers can hand a session through (the same-
    transaction path used by :func:`apply_idempotency`'s session-aware
    branch) or omit it to keep the legacy "operation owns its own
    transaction" behaviour. Hoisted into the idempotency module
    because the session-aware contract is what makes durable
    exactly-once work.
    """
    if session is not None:
        yield session
        return
    async with get_session() as new_session:
        yield new_session


async def apply_idempotency(
    tool_name: str,
    idempotency_key: str | None,
    request_payload: dict[str, Any],
    executor: ToolExecutor,
    *,
    ttl_seconds: int | None = None,
    reservation_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    """Run an executor at most once per ``(tool_name, idempotency_key)``.

    Behavioural contract:

    - ``idempotency_key is None`` → executor always runs, no caching.
    - Cache hit (response present, payload hash matches) → cached
      response returned with ``idempotency_replay=True`` injected.
    - Payload hash mismatch on the same key → ``VALIDATION_FAILED``
      envelope with ``details.idempotency_conflict=True``; executor
      does **not** run.
    - Another retry is mid-flight on the same key + payload →
      ``IDEMPOTENCY_IN_PROGRESS`` envelope (HTTP 409); executor does
      not run, the caller is expected to retry after a brief backoff.
    - No prior row → atomic ``INSERT`` reservation, executor runs,
      ``UPDATE`` with response. Executor failure ``DELETE``s the
      reservation so future retries are not permanently blocked.

    Two TTLs apply to the cache row:

    - ``reservation_ttl_seconds`` is the *pending* lifetime. If the
      reservation winner dies (process kill, network drop) between
      acquiring the slot and finalizing the row, the row expires after
      this window and a retry can re-enter the race. Default 10 min,
      well above any normal BO operation.
    - ``ttl_seconds`` is the *finalized* lifetime. Once the executor
      returns and the row carries a real response, the row lives for
      this window. Default 24 hours.

    Executor contract: ``executor`` receives the
    :class:`AsyncSession` opened by ``apply_idempotency`` and uses it
    for all DB writes. The cache finalize runs on the same session
    before commit, so the operation's side effect and the cached
    response commit atomically — there is no window in which the
    write exists without a cached response. Executors that do not
    touch the DB can ignore the session argument.
    """
    if idempotency_key is None:
        async with get_session() as session:
            return await executor(session)

    ttl = ttl_seconds or DEFAULT_IDEMPOTENCY_TTL_SECONDS
    reservation_ttl = reservation_ttl_seconds or DEFAULT_RESERVATION_TTL_SECONDS
    request_hash = canonical_request_hash(request_payload)

    outcome = await _reserve_or_short_circuit(
        tool_name, idempotency_key, request_hash, ttl, reservation_ttl
    )
    if outcome.short_circuit is not None:
        return outcome.short_circuit
    assert outcome.reservation_token is not None
    token = outcome.reservation_token

    return await _run_session_aware(
        executor,
        tool_name,
        idempotency_key,
        request_hash,
        token,
        ttl,
    )


def _is_transient_error(response: dict[str, Any]) -> bool:
    """Whether the executor returned a retryable-error envelope.

    Any envelope whose error is marked ``retryable=True`` must not be
    cached: caching it would block all future retries until the 24h
    TTL elapsed even though a successful outcome is possible
    immediately. This covers the historical ``CONCURRENT_MODIFICATION``
    case and every newer addition to the typed backend hierarchy
    (notably ``BACKEND_TRANSIENT_ERROR``) introduced in TODO 8.13/8.14
    without having to re-list each code here.

    Other operation errors (validation failures, missing campaign,
    incompatible backend, etc.) are deterministic given the same
    payload, so caching them is correct and avoids re-running an
    operation that will always fail the same way.

    The lookup first consults the envelope's own ``retryable`` flag
    (always populated by :func:`make_error_response` since 8.14) so
    custom envelopes that set the flag are honoured; falls back to
    the central :data:`ERROR_CODE_RETRY_HINTS` table when the field
    is missing (legacy callers that built the envelope by hand).
    """
    if response.get("success") is not False:
        return False
    error = response.get("error")
    if not isinstance(error, dict):
        return False
    if "retryable" in error:
        return bool(error["retryable"])
    code_value = error.get("code")
    if not code_value:
        return False
    try:
        code = ErrorCode(code_value)
    except ValueError:
        return False
    retryable, _ = retry_hint_for(code)
    return retryable


class _StaleReservationError(Exception):
    """Raised when finalize finds our token no longer owns the cache row.

    Internal-only signal between the finalize step and the outer
    handler in :func:`_run_session_aware`. Triggers a session rollback
    so the operation's writes never commit — without this, a slow
    operation whose reservation TTL elapsed (and was reclaimed by a
    concurrent retry) would otherwise still commit its DB writes after
    losing the cache slot, producing duplicate side effects.
    """


def _stale_reservation_envelope(tool_name: str, key: str) -> dict[str, Any]:
    """Envelope returned to a stale session-aware owner.

    Reuses :class:`ErrorCode.IDEMPOTENCY_IN_PROGRESS` because the
    agent-facing semantics are identical: another caller holds the
    slot and the caller should retry with the same key. On retry the
    normal lookup will either return the new owner's cached response
    or the in-progress envelope until it finishes.
    """
    return make_error_response(
        ErrorCode.IDEMPOTENCY_IN_PROGRESS,
        message=(
            f"Operation {tool_name} lost its idempotency reservation for "
            f"key {key!r} while running (slow execution past the pending "
            "TTL). The transaction has been rolled back; retry with the "
            "same key to see the active owner's result."
        ),
        details={
            "idempotency_in_progress": True,
            "stale_owner": True,
            "tool_name": tool_name,
            "idempotency_key": key,
        },
    )


async def _run_session_aware(
    executor: ToolExecutor,
    tool_name: str,
    idempotency_key: str,
    request_hash: str,
    reservation_token: str,
    ttl_seconds: int,
) -> dict[str, Any]:
    """Durable path: operation writes + cache finalize commit atomically.

    Opens one session for both the executor's work and the cache
    finalize; commits at the end of the ``async with`` block. If
    anything raises, the entire transaction rolls back and the
    reservation is dropped in a separate transaction so a future retry
    can win the slot. If the operation commits, the cache is already
    finalized — there is no window in which the side effect exists
    without a cached response.

    Transient error contract (review pass): the executor may catch a
    retryable failure (e.g. ``ConcurrentModificationError``) and return
    a structured envelope rather than raise. In that case the
    operation is expected to have rolled back its partial writes on
    the supplied session, and we *skip* the cache finalize and drop
    the reservation outside the transaction so future retries can re-
    enter the race. Otherwise an optimistic-lock conflict would cache
    a permanent error response under a key that should remain
    retryable.

    Stale ownership contract (review pass): if the finalize UPDATE
    affects zero rows, our reservation has been reclaimed by another
    retry (the operation outran its pending TTL). The token-matched
    update is a no-op, so just logging would let our operation's
    writes commit anyway — that is the bug. Instead we raise
    :class:`_StaleReservationError` so ``get_session`` rolls the
    session back, then return a retryable structured envelope to the
    caller. The retry will hit the new owner's slot through the
    normal lookup path.
    """
    try:
        async with get_session() as session:
            response = await executor(session)
            transient = _is_transient_error(response)
            if not transient:
                affected = await finalize_reservation_in_session(
                    session,
                    tool_name,
                    idempotency_key,
                    request_hash,
                    reservation_token,
                    response,
                    ttl_seconds,
                )
                if affected == 0:
                    logger.warning(
                        "Stale reservation finalize for %s/%s (token=%s): "
                        "the slot is now owned by a different caller. "
                        "Rolling back the operation's writes.",
                        tool_name,
                        idempotency_key,
                        reservation_token,
                    )
                    raise _StaleReservationError
    except _StaleReservationError:
        # The session rolled back via get_session's exception handler,
        # so no operation writes survive. The cache row is owned by
        # someone else now — do NOT drop the reservation (the
        # token-matched drop would no-op anyway, but skipping the call
        # makes the intent explicit). Return a retryable envelope.
        return _attach_metadata(_stale_reservation_envelope(tool_name, idempotency_key))
    except Exception:
        # The async with already rolled back the session; drop our
        # reservation so a future retry can claim the slot.
        await _drop_reservation(tool_name, idempotency_key, reservation_token)
        raise

    if transient:
        # Operation rolled back its writes; do not cache the transient
        # error response — drop the reservation so a retry with the
        # same key can race for a fresh slot.
        await _drop_reservation(tool_name, idempotency_key, reservation_token)

    return response
