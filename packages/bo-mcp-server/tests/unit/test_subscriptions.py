"""Unit tests for the campaign-resource subscription registry.

The MCP resource-subscription spec
(https://modelcontextprotocol.io/specification/2025-06-18/server/resources#subscriptions)
documents three guarantees the server must uphold:

1. ``resources/subscribe`` registers interest in a specific URI.
2. ``resources/unsubscribe`` cleanly removes that interest.
3. ``notifications/resources/updated`` is delivered to every subscriber
   when the underlying resource changes -- and to nobody else.

These tests exercise the registry against a fake ServerSession so we
do not need a live MCP transport; the integration with the FastMCP
instance is covered separately in
``test_subscriptions_capability_and_registration``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import pytest

from bo_mcp_server.subscriptions import (
    campaign_uri,
    get_registry,
    notify_campaign_updated,
    reset_for_tests,
)


class _FakeSession:
    """Minimal stand-in for ``mcp.server.session.ServerSession``.

    Only ``send_resource_updated`` is exercised by the registry.
    """

    def __init__(self) -> None:
        self.delivered: list[str] = []
        self.fail_on_send = False

    async def send_resource_updated(self, uri: Any) -> None:
        if self.fail_on_send:
            raise RuntimeError("simulated transport failure")
        self.delivered.append(str(uri))


@pytest.fixture(autouse=True)
async def _reset() -> AsyncGenerator[None]:
    """Drop registry state between tests."""
    await reset_for_tests()
    yield
    await reset_for_tests()


class TestSubscribeUnsubscribe:
    """Round-trip subscribe + unsubscribe across the public registry API."""

    @pytest.mark.asyncio
    async def test_subscribe_accepts_canonical_uri(self) -> None:
        cid = uuid4()
        session = _FakeSession()
        registry = get_registry()
        accepted = await registry.subscribe(campaign_uri(cid), session)
        assert accepted is True
        assert await registry.total_subscriptions() == 1

    @pytest.mark.asyncio
    async def test_subscribe_normalizes_bare_uuid(self) -> None:
        """Some SDKs strip the ``campaign://`` prefix on URI normalization."""
        cid = uuid4()
        session = _FakeSession()
        registry = get_registry()
        assert await registry.subscribe(str(cid), session) is True
        # Notification keyed by canonical URI must still find this session.
        await notify_campaign_updated(cid)
        assert session.delivered == [campaign_uri(cid)]

    @pytest.mark.asyncio
    async def test_subscribe_rejects_unknown_scheme(self) -> None:
        registry = get_registry()
        accepted = await registry.subscribe("widget://not-a-uuid", _FakeSession())
        assert accepted is False
        assert await registry.total_subscriptions() == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_drops_the_correct_session(self) -> None:
        cid = uuid4()
        a = _FakeSession()
        b = _FakeSession()
        registry = get_registry()
        await registry.subscribe(campaign_uri(cid), a)
        await registry.subscribe(campaign_uri(cid), b)
        assert await registry.unsubscribe(campaign_uri(cid), a) is True

        await notify_campaign_updated(cid)
        # Only the surviving subscriber receives the notification.
        assert a.delivered == []
        assert b.delivered == [campaign_uri(cid)]

    @pytest.mark.asyncio
    async def test_unsubscribe_unknown_uri_is_a_noop(self) -> None:
        registry = get_registry()
        assert await registry.unsubscribe(f"campaign://{uuid4()}", _FakeSession()) is False


class TestNotifyDelivery:
    """Notification fanout + drop-on-failure behaviour."""

    @pytest.mark.asyncio
    async def test_only_subscribed_sessions_get_notified(self) -> None:
        cid_a = uuid4()
        cid_b = uuid4()
        registry = get_registry()
        sub_a = _FakeSession()
        sub_b = _FakeSession()
        await registry.subscribe(campaign_uri(cid_a), sub_a)
        await registry.subscribe(campaign_uri(cid_b), sub_b)

        await notify_campaign_updated(cid_a)
        assert sub_a.delivered == [campaign_uri(cid_a)]
        assert sub_b.delivered == []

    @pytest.mark.asyncio
    async def test_failed_send_drops_subscriber_silently(self) -> None:
        cid = uuid4()
        registry = get_registry()
        flaky = _FakeSession()
        flaky.fail_on_send = True
        await registry.subscribe(campaign_uri(cid), flaky)

        # The first delivery raises internally; the registry must catch
        # and unsubscribe so the next notification finds nobody.
        await notify_campaign_updated(cid)
        assert await registry.total_subscriptions() == 0

    @pytest.mark.asyncio
    async def test_no_subscribers_no_op(self) -> None:
        # Should not raise regardless of how many notifications fire
        # without a registered subscriber.
        await notify_campaign_updated(uuid4())
        registry = get_registry()
        assert await registry.total_subscriptions() == 0


class TestCapabilityWiring:
    """``create_mcp_server`` advertises and wires the subscription surface."""

    def test_subscribe_capability_is_advertised(self) -> None:
        from mcp import types
        from mcp.server.lowlevel.server import NotificationOptions

        from bo_mcp_server.server import create_mcp_server

        mcp = create_mcp_server()
        lowlevel = mcp._mcp_server
        # Both handlers registered.
        assert types.SubscribeRequest in lowlevel.request_handlers
        assert types.UnsubscribeRequest in lowlevel.request_handlers
        # The MCP lowlevel hardcodes ``subscribe=False``; our wiring
        # patches ``get_capabilities`` to flip it on so the handshake
        # advertises the capability.
        caps = lowlevel.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        )
        assert caps.resources is not None
        assert caps.resources.subscribe is True
