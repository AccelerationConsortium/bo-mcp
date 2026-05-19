"""Resource subscription registry for ``campaign://{id}`` URIs.

The MCP ``resources/subscribe`` capability lets a client register
interest in a resource URI so the server can push
``notifications/resources/updated`` whenever the underlying state
changes -- agents stop polling, terminal-state transitions surface
within one round-trip, and multi-agent orchestration loops save the
turns they would otherwise spend on backoff polling.

This module owns the per-session subscription registry and the wire
encoding used by lifecycle callers when they emit notifications. The
session integration (registering subscribe / unsubscribe handlers and
patching the lowlevel server's hardcoded ``subscribe=False`` capability
flag) lives in :mod:`bo_mcp_server.server`.

Design notes
------------

* Sessions are weakly tracked: when a client disconnects, its
  ``_ResourceSubscriber`` is dropped from the registry so we never try to
  push to a dead transport. We attach a finalizer rather than rely on
  unsubscribe alone because clients are not obligated to send
  ``resources/unsubscribe`` before tearing down.
* Notifications are best-effort: a failed ``send_resource_updated``
  removes the subscription rather than propagating the error up to the
  caller. State changes must succeed even when push delivery is
  flaky.
* Two URI shapes are accepted -- the canonical ``campaign://{id}`` and
  the legacy bare ``{id}`` -- so SDK quirks around URI normalization do
  not silently break subscription matching.

References:
----------
- MCP resource-subscription spec
  https://modelcontextprotocol.io/specification/2025-06-18/server/resources#subscriptions
- MCP server capabilities reference (subscribe flag)
  https://modelcontextprotocol.io/specification/2025-06-18/server#capabilities
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel import Server as LowlevelServer
from mcp.server.lowlevel.server import NotificationOptions
from mcp.types import ServerCapabilities
from pydantic import AnyUrl
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from bo_mcp_server.metrics import SUBSCRIPTION_DROPPED, SUBSCRIPTION_SEND_RETRIES

logger = logging.getLogger(__name__)

# Retry budget for a single resource-update delivery. The wire-level
# spec does not guarantee delivery, but a single transient hiccup
# (transport reconnect, momentary backpressure) should not be enough
# to silently unsubscribe a long-running agent. Three attempts with
# exponential backoff is enough to ride out brief flakes while
# bounding worst-case latency for permanent failures.
SUBSCRIPTION_SEND_MAX_ATTEMPTS = 3
# Base backoff in seconds before the first retry; doubles on each
# subsequent attempt (0.05s, 0.10s). Kept small because the receiving
# session is in the same process — anything longer would just stall
# unrelated notifications behind this one.
SUBSCRIPTION_SEND_BACKOFF_BASE_SECONDS = 0.05


@runtime_checkable
class _ResourceSubscriber(Protocol):
    """Structural protocol matching ``mcp.server.session._ResourceSubscriber``.

    Using a Protocol here lets unit tests substitute a fake session
    without depending on the real MCP transport, while still letting
    ``ty`` enforce that production callers pass a session that knows
    how to send the ``notifications/resources/updated`` payload.
    """

    async def send_resource_updated(self, uri: AnyUrl, /) -> None: ...


CAMPAIGN_URI_SCHEME = "campaign"


def campaign_uri(campaign_id: UUID | str) -> str:
    """Build the canonical ``campaign://{id}`` URI for a campaign id."""
    return f"{CAMPAIGN_URI_SCHEME}://{campaign_id}"


def _normalize_uri(raw: str | AnyUrl) -> str | None:
    """Map any client-supplied URI shape to the canonical form.

    Returns ``None`` when the URI does not address a campaign. The
    server should treat that as an unsupported subscription target;
    we'd rather drop it than silently subscribe to a no-op channel.
    """
    text = str(raw)
    if text.startswith(f"{CAMPAIGN_URI_SCHEME}://"):
        return text
    # Tolerate the bare ``{uuid}`` shape some SDKs emit when the scheme
    # is part of the resource template definition.
    try:
        UUID(text)
    except ValueError:
        return None
    return campaign_uri(text)


class _SubscriptionRegistry:
    """Maps canonical URIs to weakly-held subscriber sessions.

    Notes:
        * The inner sets store ``weakref.ref`` objects so dropped
          sessions are eligible for GC the moment the transport tears
          down -- without forcing the registry to learn about every
          disconnect path.
        * All mutations take an asyncio lock so concurrent
          subscribe / unsubscribe / notify do not race with each other
          (tested via :func:`tests/integration/test_resource_subscriptions.py`).
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, set[weakref.ref[_ResourceSubscriber]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, uri: str | AnyUrl, session: _ResourceSubscriber) -> bool:
        """Register ``session`` for ``uri``. Returns True iff the URI was accepted."""
        canonical = _normalize_uri(uri)
        if canonical is None:
            return False
        async with self._lock:
            self._subscribers.setdefault(canonical, set()).add(weakref.ref(session))
        logger.debug("Subscribed session to %s", canonical)
        return True

    async def unsubscribe(self, uri: str | AnyUrl, session: _ResourceSubscriber) -> bool:
        """Drop ``session`` from ``uri``. Returns True iff the entry existed."""
        canonical = _normalize_uri(uri)
        if canonical is None:
            return False
        target_id = id(session)
        async with self._lock:
            holders = self._subscribers.get(canonical)
            if not holders:
                return False
            removed = False
            survivors: set[weakref.ref[_ResourceSubscriber]] = set()
            for ref in holders:
                obj = ref()
                if obj is None:
                    continue
                if id(obj) == target_id:
                    removed = True
                    continue
                survivors.add(ref)
            if survivors:
                self._subscribers[canonical] = survivors
            else:
                self._subscribers.pop(canonical, None)
            return removed

    async def snapshot(self, uri: str | AnyUrl) -> list[_ResourceSubscriber]:
        """Return live sessions subscribed to ``uri``.

        Performs a sweep of dead weakrefs so notifier callers do not
        keep paying GC tax on disconnected sessions.
        """
        canonical = _normalize_uri(uri)
        if canonical is None:
            return []
        async with self._lock:
            holders = self._subscribers.get(canonical)
            if not holders:
                return []
            live: list[_ResourceSubscriber] = []
            survivors: set[weakref.ref[_ResourceSubscriber]] = set()
            for ref in holders:
                obj = ref()
                if obj is None:
                    continue
                live.append(obj)
                survivors.add(ref)
            if survivors:
                self._subscribers[canonical] = survivors
            else:
                self._subscribers.pop(canonical, None)
            return live

    async def total_subscriptions(self) -> int:
        """Return the number of (uri, session) pairs currently held.

        Test-only helper; included here so tests do not have to reach
        into the private attribute.
        """
        async with self._lock:
            return sum(len(refs) for refs in self._subscribers.values())


_registry = _SubscriptionRegistry()

# Strong refs to fire-and-forget notification tasks. ``loop.create_task``
# only holds a weak reference, so without anchoring the task can be GC'd
# before it runs. ``add_done_callback(_BACKGROUND_TASKS.discard)`` removes
# the ref once the task completes so this set stays bounded.
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


def get_registry() -> _SubscriptionRegistry:
    """Module-level accessor used by server wiring and tests."""
    return _registry


async def _deliver_with_retry(
    session: _ResourceSubscriber,
    uri: str,
    parsed: AnyUrl,
) -> bool:
    """Send a resource-update with bounded exponential-backoff retries.

    Returns ``True`` if delivery succeeded (possibly on a retry),
    ``False`` after the full retry budget is exhausted. Centralising
    the retry loop here keeps :func:`notify_campaign_updated` focused
    on fan-out and lets us emit a single ``SUBSCRIPTION_SEND_RETRIES``
    bump on each recovered delivery so dashboards distinguish
    "wire-blip ridden out" from "subscription wedged".

    The retry strategy is intentionally small: a long-running agent
    that holds a subscription open for hours is the workload we want
    to protect, and three attempts ride out the transient hiccups
    that previously dropped the subscriber on the very first error.
    Permanent transport failures (closed session, decoder mismatch)
    still surface as an unsubscribe after the budget is exhausted.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, SUBSCRIPTION_SEND_MAX_ATTEMPTS + 1):
        try:
            await session.send_resource_updated(parsed)
        except Exception as exc:  # noqa: BLE001 -- transport errors are retried then escalated
            last_exc = exc
            if attempt >= SUBSCRIPTION_SEND_MAX_ATTEMPTS:
                break
            backoff = SUBSCRIPTION_SEND_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.debug(
                "Resource-update delivery for %s failed on attempt %d/%d; retrying after %.3fs.",
                uri,
                attempt,
                SUBSCRIPTION_SEND_MAX_ATTEMPTS,
                backoff,
                exc_info=True,
            )
            await asyncio.sleep(backoff)
            continue
        if attempt > 1:
            SUBSCRIPTION_SEND_RETRIES.inc()
        return True

    logger.warning(
        "Failed to deliver resource update for %s after %d attempts; dropping subscriber.",
        uri,
        SUBSCRIPTION_SEND_MAX_ATTEMPTS,
        exc_info=last_exc,
    )
    return False


async def notify_campaign_updated(campaign_id: UUID | str) -> None:
    """Broadcast a ``notifications/resources/updated`` for ``campaign://{id}``.

    Delivery is bounded-retry: each subscriber's
    ``send_resource_updated`` call is attempted up to
    :data:`SUBSCRIPTION_SEND_MAX_ATTEMPTS` times with exponential
    backoff before the subscription is unregistered. The retry budget
    keeps a transient wire blip from silently severing a long-running
    agent's subscription; the unsubscribe-after-exhaustion behaviour
    keeps a permanently-broken transport from re-firing on every
    notification. The function never raises -- lifecycle callers must
    not see push failures bubble up into their main flow.
    """
    uri = campaign_uri(campaign_id)
    sessions = await _registry.snapshot(uri)
    if not sessions:
        return
    parsed = AnyUrl(uri)
    for session in sessions:
        delivered = await _deliver_with_retry(session, uri, parsed)
        if not delivered:
            SUBSCRIPTION_DROPPED.inc()
            await _registry.unsubscribe(uri, session)


def notify_campaign_updated_after_commit(
    session: AsyncSession,
    campaign_id: UUID | str,
) -> None:
    """Schedule a campaign-update notification to fire after ``session`` commits.

    Why a deferred hook instead of a direct call: operations that
    accept an external session (the idempotency-aware path) hand the
    session to the executor and let the *outer* transaction handler
    commit it. Calling :func:`notify_campaign_updated` from inside the
    operation would push notifications that:

    1. Subscribers see *before* the row is durable -- a re-read of
       ``campaign://{id}`` at that point still returns the pre-commit
       state.
    2. May be objectively wrong: an outer rollback (idempotency
       reservation conflict, optimistic-lock race detected
       post-executor) would leave the row at its old value while
       subscribers were already told it changed.

    Using SQLAlchemy's ``after_commit`` event sidesteps both: the
    callback only fires once the underlying transaction has hit the
    log, and never fires if the transaction rolls back. The same hook
    works for the operation-owns-the-session path (commit at
    ``async with`` exit) and the idempotency-owns-the-session path
    (commit at ``_run_session_aware``'s ``async with`` exit).

    Notes:
        * The hook is one-shot per ``(session, callback)`` pair: we
          remove the listener inside itself so a long-lived session
          that commits multiple times only notifies for the
          transactions that explicitly armed a hook.
        * ``after_commit`` fires synchronously inside SQLAlchemy's
          internal callback machinery, so the actual ``send_resource_updated``
          call has to be scheduled on the running event loop via
          ``asyncio.create_task``. Failures are caught and logged in
          :func:`notify_campaign_updated`; we do not propagate.
    """
    sync_session = session.sync_session
    fired = False

    def _on_after_commit(_: Session) -> None:
        nonlocal fired
        if fired:
            # The hook is one-shot per arming: once a listener has
            # fired we keep it attached but make it a no-op. We do not
            # call ``event.remove`` from inside the listener because
            # SQLAlchemy's event subsystem may be iterating over its
            # listener list at that time -- removing during iteration
            # raises ``InvalidRequestError`` from the post-commit
            # state-machine assertion. Sessions in our codebase are
            # short-lived (one per operation) so the residual listener
            # is GC'd with the session.
            return
        fired = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop -- best effort, run synchronously via
            # asyncio.run. Realistic operation paths always have a
            # running loop, but this keeps the helper robust under
            # synchronous test fixtures.
            asyncio.run(notify_campaign_updated(campaign_id))
            return
        task = loop.create_task(notify_campaign_updated(campaign_id))
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)

    event.listen(sync_session, "after_commit", _on_after_commit)


async def reset_for_tests() -> None:
    """Clear all subscriptions. Test-only helper."""
    async with _registry._lock:
        _registry._subscribers.clear()


def _patch_subscribe_capability(server: LowlevelServer) -> None:
    """Force ``ResourcesCapability.subscribe = True`` on the lowlevel server.

    The MCP lowlevel server hardcodes ``subscribe=False`` in
    ``Server.get_capabilities`` even when a SubscribeRequest handler is
    registered (see ``mcp/server/lowlevel/server.py:212``). Without
    this patch, clients would never advertise interest in subscribing
    because the initialize handshake claims the server does not
    support it. We wrap ``get_capabilities`` rather than monkey-patch
    ``ResourcesCapability`` so the override is scoped to this server
    instance and survives library upgrades that change the default.
    """
    original = server.get_capabilities

    def patched(
        notification_options: NotificationOptions,
        experimental_capabilities: dict[str, dict[str, Any]],
    ) -> ServerCapabilities:
        capabilities = original(notification_options, experimental_capabilities)
        if capabilities.resources is not None:
            capabilities.resources = capabilities.resources.model_copy(update={"subscribe": True})
        return capabilities

    server.get_capabilities = patched  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]


def register_subscription_handlers(mcp_instance: FastMCP) -> None:
    """Wire subscribe / unsubscribe handlers into a FastMCP instance.

    Registers the lowlevel handlers and patches the capability flag so
    the initialize handshake advertises subscription support. Idempotent
    -- registering twice replaces the prior handler, which is the
    behaviour the lowlevel server already provides via
    ``request_handlers[...] = handler``.
    """
    lowlevel = mcp_instance._mcp_server
    registry = get_registry()

    @lowlevel.subscribe_resource()
    async def _on_subscribe(uri: AnyUrl) -> None:
        # The handler does not get the session, so we pull it from the
        # request context. This is the same pattern FastMCP uses for
        # its own session-aware handlers (see
        # ``mcp/server/fastmcp/server.py``'s call_tool path).
        session = lowlevel.request_context.session
        accepted = await registry.subscribe(uri, session)
        if not accepted:
            logger.info("Ignoring subscribe request for unsupported URI: %s", uri)

    @lowlevel.unsubscribe_resource()
    async def _on_unsubscribe(uri: AnyUrl) -> None:
        session = lowlevel.request_context.session
        await registry.unsubscribe(uri, session)

    _patch_subscribe_capability(lowlevel)
