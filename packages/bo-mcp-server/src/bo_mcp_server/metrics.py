"""Domain Prometheus instruments for bo-mcp-server.

Defined on the server layer (rather than only in ``api.metrics``) so
non-HTTP entry points — the MCP transport, scheduled jobs, tests —
can record the same metrics without depending on the REST package.
The REST surface in :mod:`api.metrics` imports these instruments and
adds HTTP-specific request/latency counters around them.

Naming follows Prometheus conventions (``bo_mcp_*`` namespace,
``_total`` suffix on counters, base SI units in histogram buckets).

Reference: https://prometheus.io/docs/practices/naming/.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from prometheus_client import Counter, Histogram
from sqlalchemy import event
from sqlalchemy.engine import Connection
from sqlalchemy.exc import InvalidRequestError

logger = logging.getLogger(__name__)


CAMPAIGNS_CREATED = Counter(
    "bo_mcp_campaigns_created_total",
    "Campaigns persisted via the REST or MCP transport.",
    labelnames=("backend",),
)
SUGGESTION_LATENCY = Histogram(
    "bo_mcp_suggestion_generation_duration_seconds",
    "Wall-clock latency of generate_suggestions, end-to-end.",
    labelnames=("backend",),
    buckets=(0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)
CACHE_EVENTS = Counter(
    "bo_mcp_diagnostics_cache_events_total",
    "Diagnostics-cache lookup outcomes.",
    labelnames=("outcome",),
)
AUDIT_FAILURES = Counter(
    "bo_mcp_audit_failures_total",
    "Audit-event persistence failures, regardless of whether the parent tool succeeded.",
    labelnames=("tool",),
)
IDEMPOTENCY_GC_PURGED = Counter(
    "bo_mcp_idempotency_cache_gc_rows_total",
    "Expired idempotency_cache rows removed by the background GC sweep.",
)
SUBSCRIPTION_SEND_RETRIES = Counter(
    "bo_mcp_subscription_send_retries_total",
    "Resource-update deliveries that succeeded on a retry attempt.",
)
SUBSCRIPTION_DROPPED = Counter(
    "bo_mcp_subscription_dropped_total",
    "Subscriptions unregistered after exhausting the retry budget on push delivery.",
)
PROGRESS_NOTIFY_FAILURES = Counter(
    "bo_mcp_progress_notify_failures_total",
    "Progress events that could not be forwarded to the MCP session. "
    "A sustained non-zero rate means clients waiting on progress for "
    "ETAs or cancellation are hanging — the poll fallback "
    "(bo_check_progress) is the supported workaround.",
    labelnames=("reason",),
)


def record_campaign_created(backend: str | None) -> None:
    """Bump the per-backend campaign counter (``unknown`` when missing)."""
    CAMPAIGNS_CREATED.labels(backend or "unknown").inc()


def record_campaign_created_after_commit(session: Any, backend: str | None) -> None:
    """Defer :func:`record_campaign_created` until ``session`` commits.

    The operation can be invoked with an externally-owned session
    (the ``apply_idempotency`` session-aware path): in that case
    ``session_scope(session).__aexit__`` does **not** commit — the
    outer caller does. Calling :func:`record_campaign_created`
    immediately after the ``async with session_scope(...)`` block
    would therefore bump the counter while the row is still
    uncommitted, and an outer rollback (idempotency reservation
    conflict, downstream optimistic-lock race) would leave the metric
    inflated relative to persisted state.

    The SQLAlchemy ``after_commit`` event fires whenever
    ``Session.commit()`` returns — including for a ``begin_nested()``
    SAVEPOINT release inside an externally managed outer transaction
    (the ``tests/conftest_postgres.py`` test-isolation pattern). In
    that shape the data isn't yet durable: the outer transaction may
    still roll back at teardown. We therefore guard the bump with
    :meth:`Session.in_nested_transaction` — when the just-completed
    commit was a SAVEPOINT release the session still reports a
    nested transaction (autobegin opens a fresh inner SAVEPOINT
    immediately on the way back into application code), so we skip
    the bump and let the outer commit/rollback decide. Pure root
    commits — the only shape that occurs in production — see
    ``in_nested_transaction() is False`` and fire the metric.

    The hook is also **transaction-scoped**, not session-scoped: an
    ``after_rollback`` listener disarms the commit listener for the
    rolled-back transaction so a subsequent unrelated commit on the
    same long-lived session cannot resurrect the original campaign's
    counter bump.

    Both listeners are explicitly removed (via a deferred
    ``asyncio`` task — SQLAlchemy raises ``RuntimeError: deque mutated
    during iteration`` if a listener is removed inline during
    dispatch) after either fires so genuinely long-lived
    externally-owned sessions do not accumulate inert callbacks. The
    ``armed`` flag still gates the commit handler in case a second
    transaction completes before the cleanup task has a chance to
    run.

    Known limitation: when an externally managed outer transaction
    *does* commit (a hypothetical "host owns the transaction" pattern
    we do not currently use), the metric stays unfired — the listener
    is detached on the inner SAVEPOINT release and never sees the
    outer commit. Production paths open the session at the root, so
    this case does not occur. If a future caller introduces an outer
    "real commit" pattern, register the metric on that owner's commit
    boundary directly.
    """
    sync_session = session.sync_session
    bind = sync_session.bind
    # ``isinstance(bind, Connection)`` flags the connection-bound
    # patterns where the session may be sharing an externally owned
    # transaction. We can't decide durability *here* (after autobegin
    # has run, both the durable and joined shapes look the same), but
    # we capture the bind so the after-commit listener can inspect
    # the connection's transaction state at commit time — the two
    # shapes split cleanly there (see :func:`_commit_is_durable`).
    external_connection: Connection | None = bind if isinstance(bind, Connection) else None
    state = _ArmingState()

    def _detach() -> None:
        # Listener removal must run outside the SQLAlchemy event
        # dispatch — calling ``event.remove`` while ``dispatch.X`` is
        # iterating its listener deque trips a runtime error.
        for evt, listener in (
            ("after_commit", _on_after_commit),
            ("after_rollback", _on_after_rollback),
        ):
            try:
                event.remove(sync_session, evt, listener)
            except InvalidRequestError:
                pass

    def _schedule_detach() -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — extremely unusual for our async code
            # paths, but fall back to inline removal so the listener
            # still doesn't leak. The dispatch has already returned
            # by the time the synchronous _on_after_* path finishes,
            # so inline removal is safe here.
            _detach()
            return
        loop.call_soon(_detach)

    def _on_after_commit(committed_session: Any) -> None:
        if not state.armed:
            _schedule_detach()
            return
        if not _commit_is_durable(committed_session, external_connection):
            # Non-durable commit — either a SAVEPOINT release or a
            # sub-commit inside an externally owned outer transaction.
            # We CANNOT keep the arming live: the outer transaction may
            # be committed or rolled back through the Connection
            # directly, which fires neither ``after_commit`` nor
            # ``after_rollback`` on the session. Leaving the listener
            # armed would let a later, unrelated durable commit on the
            # same session retroactively bump the counter for the
            # rolled-back campaign. Disarm + detach instead — the host
            # owning the outer transaction is now responsible for the
            # metric (call :func:`record_campaign_created` after their
            # real commit boundary if metric tracking is required).
            state.armed = False
            _schedule_detach()
            return
        state.armed = False
        record_campaign_created(backend)
        _schedule_detach()

    def _on_after_rollback(_: Any) -> None:
        # Rollback disables the commit listener for this arming so a
        # later, unrelated commit on the same session does not bump
        # the counter retroactively.
        state.armed = False
        _schedule_detach()

    event.listen(sync_session, "after_commit", _on_after_commit)
    event.listen(sync_session, "after_rollback", _on_after_rollback)


@dataclass
class _ArmingState:
    """Mutable arming flag for the deferred metric hook.

    Stored on a small dataclass instead of a closure ``nonlocal`` so
    the nested commit / rollback listeners can flip it without
    inflating their cognitive-complexity bookkeeping.
    """

    armed: bool = True


def _commit_is_durable(
    committed_session: Any,
    external_connection: Connection | None,
) -> bool:
    """Return True when the just-fired commit actually persists state.

    Two non-durable shapes need to skip the metric bump:

    * ``Session.in_nested_transaction()`` is True → the commit just
      released a SAVEPOINT inside an outer transaction
      (test-isolation pattern from ``tests/conftest_postgres.py``).
    * The session is connection-bound and the connection is *still*
      in a transaction after ``Session.commit()`` → the session was
      joined to an externally owned outer transaction that hasn't
      committed yet. In the legitimate connection-bound-durable
      shape (session autobegan its own root transaction on a bare
      connection) ``connection.in_transaction()`` is False here.

    Introspection exceptions are conservatively treated as "looks
    durable" so the metric path doesn't silently swallow itself when
    a future SQLAlchemy release changes the introspection surface.
    """
    try:
        if bool(committed_session.in_nested_transaction()):
            return False
    except Exception:  # noqa: BLE001, S110 - introspection is best-effort
        logger.debug("Session.in_nested_transaction() raised", exc_info=True)
    if external_connection is None:
        return True
    try:
        return not bool(external_connection.in_transaction())
    except Exception:  # noqa: BLE001 - introspection is best-effort
        logger.debug("Connection.in_transaction() raised", exc_info=True)
        return True


def observe_suggestion_latency(backend: str | None, seconds: float) -> None:
    """Record the wall-clock latency of a generate_suggestions call."""
    SUGGESTION_LATENCY.labels(backend or "unknown").observe(seconds)


def record_audit_failure(tool: str) -> None:
    """Bump the audit-failure counter for ``tool``.

    Always called from the audit logger's failure branch — dashboards
    can alarm on ``rate(bo_mcp_audit_failures_total[5m]) > 0`` even when
    ``AUDIT_FAILURES_FATAL`` is unset and the parent tool kept running.
    """
    AUDIT_FAILURES.labels(tool or "unknown").inc()


def record_idempotency_gc(rows_removed: int) -> None:
    """Bump the idempotency-cache GC counter by ``rows_removed``.

    Always called from the background sweep; a zero sweep is recorded
    as a no-op so the counter still exists in ``/metrics`` once the
    sweep has run at least once.
    """
    if rows_removed > 0:
        IDEMPOTENCY_GC_PURGED.inc(rows_removed)


def record_diagnostics_cache(outcome: str) -> None:
    """Bump the diagnostics-cache counter labelled by ``hit`` or ``miss``.

    Any string value is accepted so callers can extend the label set
    (e.g. ``"skip"`` when caching is disabled) without code changes.
    """
    CACHE_EVENTS.labels(outcome).inc()


def snapshot_db_pool() -> dict[str, float]:
    """Return a snapshot of the SQLAlchemy connection-pool utilization.

    Reads ``checked_out`` / ``checked_in`` / ``overflow`` / ``size`` off
    the current async engine's pool, if one exists. The snapshot is
    explicitly read-only: it inspects the engine slot directly rather
    than going through ``_get_engine()`` (which would lazily construct
    an engine), so a ``/metrics`` scrape that happens before the first
    real DB request does not have the side effect of opening a
    connection pool. Exposed through the client facade so REST / MCP
    transports can publish pool gauges without reaching into
    :mod:`bo_mcp_server.storage` directly. Returns ``{}`` when no
    engine has been created yet or when the pool implementation does
    not expose the relevant counters.
    """
    try:
        from bo_mcp_server.storage import database  # noqa: PLC0415

        # Read the slot directly — do *not* call ``_get_engine``,
        # which would lazily construct an engine. A metrics scrape
        # should observe state, not create it.
        engine = getattr(database, "_engine", None)
        if engine is None:
            return {}
        pool = engine.pool
    except Exception:  # noqa: BLE001 - metric snapshot is best-effort
        return {}
    snapshot: dict[str, float] = {}
    for state, getter in (
        ("checked_out", "checkedout"),
        ("checked_in", "checkedin"),
        ("overflow", "overflow"),
        ("size", "size"),
    ):
        accessor: Any = getattr(pool, getter, None)
        if accessor is None:
            continue
        try:
            snapshot[state] = float(accessor())
        except Exception:  # noqa: BLE001, S112 - per-pool method may raise on non-pooled engines
            logger.debug("DB pool accessor %s raised", state, exc_info=True)
            continue
    return snapshot


def reset_for_test() -> None:
    """Reset every domain metric — for unit tests only.

    ``prometheus_client.Counter`` does not expose a public reset, so we
    walk the internal sample children and zero them. The reset is
    idempotent and safe to call between tests; production code must not
    call this — it would erase running counters.
    """
    for instrument in (
        CAMPAIGNS_CREATED,
        SUGGESTION_LATENCY,
        CACHE_EVENTS,
        AUDIT_FAILURES,
    ):
        children: dict[Any, Any] = instrument._metrics  # type: ignore[attr-defined]  # noqa: SLF001
        children.clear()
    # Labelless counters bypass the ``_metrics`` child dict and store the
    # running total directly on ``_value``; reset that slot to its zero
    # equivalent so tests start from a clean baseline.
    if hasattr(IDEMPOTENCY_GC_PURGED, "_value"):
        IDEMPOTENCY_GC_PURGED._value.set(0)  # type: ignore[attr-defined]  # noqa: SLF001
