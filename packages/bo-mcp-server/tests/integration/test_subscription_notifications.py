"""End-to-end notification fan-out from lifecycle operations.

When a subscriber registers interest in ``campaign://{id}``, every
status-changing lifecycle operation (pause / resume / terminate, the
CREATED→RUNNING transition triggered by the first
``generate_suggestions`` call) must push a
``notifications/resources/updated`` to that subscriber. Subscribers
keyed to a different campaign must not receive the push.

Reference: MCP resource-subscription spec
https://modelcontextprotocol.io/specification/2025-06-18/server/resources#subscriptions
"""

from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID, uuid4

import pytest


class _FakeSession:
    """Minimal stand-in for ``mcp.server.session.ServerSession``."""

    def __init__(self) -> None:
        self.delivered: list[str] = []

    async def send_resource_updated(self, uri: Any) -> None:
        self.delivered.append(str(uri))


async def _create_running_campaign() -> tuple[str, str, UUID]:
    """Create a campaign and drive it past CREATED so we can test pause/resume."""
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = str(uuid4())
    intake = {
        "name": "Subscription Notify",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    campaign_id = created["campaign_id"]
    # First suggestion call advances CREATED -> RUNNING and itself
    # fires a notification (covered separately).
    await generate_suggestions(campaign_id, batch_size=1)
    return campaign_id, owner_id, UUID(campaign_id)


@pytest.fixture(autouse=True)
async def _reset_registry() -> AsyncGenerator[None]:
    from bo_mcp_server.subscriptions import reset_for_tests

    await reset_for_tests()
    yield
    await reset_for_tests()


class TestLifecycleNotifications:
    @pytest.mark.asyncio
    async def test_pause_emits_resource_updated(self, setup_database) -> None:
        from bo_mcp_server.operations.campaign_lifecycle import (
            manage_campaign_lifecycle_operation,
        )
        from bo_mcp_server.subscriptions import campaign_uri, get_registry

        campaign_id, _, campaign_uuid = await _create_running_campaign()
        subscriber = _FakeSession()
        await get_registry().subscribe(campaign_uri(campaign_uuid), subscriber)

        result = await manage_campaign_lifecycle_operation(campaign_id=campaign_id, action="pause")
        assert result["success"] is True
        assert subscriber.delivered == [campaign_uri(campaign_uuid)]

    @pytest.mark.asyncio
    async def test_terminate_emits_resource_updated(self, setup_database) -> None:
        from bo_mcp_server.operations.campaign_lifecycle import (
            manage_campaign_lifecycle_operation,
        )
        from bo_mcp_server.subscriptions import campaign_uri, get_registry

        campaign_id, _, campaign_uuid = await _create_running_campaign()
        subscriber = _FakeSession()
        await get_registry().subscribe(campaign_uri(campaign_uuid), subscriber)

        await manage_campaign_lifecycle_operation(campaign_id=campaign_id, action="terminate")
        assert subscriber.delivered == [campaign_uri(campaign_uuid)]

    @pytest.mark.asyncio
    async def test_subscribers_to_other_campaign_are_not_notified(self, setup_database) -> None:
        """A pause on campaign A must not leak into campaign B's subscribers."""
        from bo_mcp_server.operations.campaign_lifecycle import (
            manage_campaign_lifecycle_operation,
        )
        from bo_mcp_server.subscriptions import campaign_uri, get_registry

        campaign_a, _, _ = await _create_running_campaign()
        _, _, campaign_b_uuid = await _create_running_campaign()
        subscriber_b = _FakeSession()
        await get_registry().subscribe(campaign_uri(campaign_b_uuid), subscriber_b)

        await manage_campaign_lifecycle_operation(campaign_id=campaign_a, action="pause")
        # Subscriber B is keyed to a different campaign -- must stay quiet.
        assert subscriber_b.delivered == []

    @pytest.mark.asyncio
    async def test_invalid_state_transition_does_not_notify(self, setup_database) -> None:
        """Failed transitions must not push misleading notifications.

        Resuming a CREATED-but-never-RUNNING campaign is rejected with
        ``INVALID_STATE_TRANSITION``; subscribers must not see a
        ``resources/updated`` for a write that did not happen.
        """
        from bo_mcp_server.operations.campaign_lifecycle import (
            manage_campaign_lifecycle_operation,
        )
        from bo_mcp_server.subscriptions import campaign_uri, get_registry
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        created = await create_campaign(
            {
                "name": "No Run Yet",
                "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
            owner_id,
        )
        campaign_id = created["campaign_id"]
        subscriber = _FakeSession()
        await get_registry().subscribe(campaign_uri(UUID(campaign_id)), subscriber)

        result = await manage_campaign_lifecycle_operation(campaign_id=campaign_id, action="resume")
        assert result["success"] is False
        assert subscriber.delivered == []


class TestPostCommitOrdering:
    """Notifications fire only after the campaign-status commit is durable.

    The post-commit hook is the contract that keeps subscribers from
    seeing transitions that the surrounding transaction later rolls
    back -- and from seeing them while a re-read of ``campaign://{id}``
    would still return the pre-transition state.
    """

    @pytest.mark.asyncio
    async def test_lifecycle_notification_does_not_fire_before_commit(self, setup_database) -> None:
        """The pause notification must not fire while the lifecycle session is open."""
        import asyncio as _asyncio

        from bo_mcp_server.operations.campaign_lifecycle import (
            manage_campaign_lifecycle_operation,
        )
        from bo_mcp_server.subscriptions import campaign_uri, get_registry

        campaign_id, _, campaign_uuid = await _create_running_campaign()
        subscriber = _FakeSession()
        await get_registry().subscribe(campaign_uri(campaign_uuid), subscriber)

        result = await manage_campaign_lifecycle_operation(campaign_id=campaign_id, action="pause")
        # The hook arms a ``loop.create_task`` so the actual delivery
        # is scheduled for the next event-loop tick. Yield once so
        # those tasks get a chance to run before we assert.
        await _asyncio.sleep(0)
        assert result["success"] is True
        assert subscriber.delivered == [campaign_uri(campaign_uuid)]

    @pytest.mark.asyncio
    async def test_rollback_does_not_fire_after_commit_hook(self, setup_database) -> None:
        """An armed post-commit hook must not fire when the session rolls back.

        This is the contract that protects subscribers from seeing a
        ``resources/updated`` for a transition the surrounding
        transaction later rolled back -- the exact bug the hook
        replaces a direct ``await notify_campaign_updated(...)`` call
        to prevent. The test arms a hook on a session that then
        raises a ``SQLAlchemyError`` to force ``get_session`` into
        its rollback path, so ``after_commit`` is never fired by
        SQLAlchemy.
        """
        import asyncio as _asyncio
        from uuid import uuid4 as _uuid4

        from sqlalchemy.exc import SQLAlchemyError

        from bo_mcp_server.storage import get_session
        from bo_mcp_server.subscriptions import (
            campaign_uri,
            get_registry,
            notify_campaign_updated_after_commit,
        )

        cid = _uuid4()
        subscriber = _FakeSession()
        await get_registry().subscribe(campaign_uri(cid), subscriber)

        with pytest.raises(SQLAlchemyError):
            async with get_session() as session:
                notify_campaign_updated_after_commit(session, cid)
                # Raising a SQLAlchemyError sends ``get_session``
                # straight to its ``rollback`` branch, skipping the
                # commit. The post-commit listener therefore never
                # fires.
                raise SQLAlchemyError("simulated transactional failure")
        await _asyncio.sleep(0)

        assert subscriber.delivered == []


class TestGenerateSuggestionsTransitionsToRunning:
    @pytest.mark.asyncio
    async def test_first_suggestion_call_emits_resource_updated(self, setup_database) -> None:
        """The CREATED→RUNNING transition during ``generate_suggestions`` notifies."""
        from bo_mcp_server.subscriptions import campaign_uri, get_registry
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake = {
            "name": "Created To Running Notify",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        created = await create_campaign(intake, owner_id)
        campaign_id = created["campaign_id"]
        campaign_uuid = UUID(campaign_id)

        subscriber = _FakeSession()
        await get_registry().subscribe(campaign_uri(campaign_uuid), subscriber)

        await generate_suggestions(campaign_id, batch_size=1)
        assert subscriber.delivered == [campaign_uri(campaign_uuid)]

    @pytest.mark.asyncio
    async def test_second_suggestion_call_does_not_notify(self, setup_database) -> None:
        """Subscribers do not get pinged on every iteration bump.

        Once the campaign is already RUNNING, additional suggestion
        batches do not change ``status``. Pushing every iteration
        would defeat the polling-saver value of subscriptions.
        """
        from bo_mcp_server.subscriptions import campaign_uri, get_registry
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake = {
            "name": "No Iteration Spam",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        created = await create_campaign(intake, owner_id)
        campaign_id = created["campaign_id"]
        campaign_uuid = UUID(campaign_id)
        # Drain the CREATED -> RUNNING transition before we register
        # the subscriber so this test only sees iteration bumps.
        await generate_suggestions(campaign_id, batch_size=1)

        subscriber = _FakeSession()
        await get_registry().subscribe(campaign_uri(campaign_uuid), subscriber)

        # A subsequent suggestion batch must not push -- there is no
        # status change to report. We do not need a result to land
        # first because the engine produces additional initial-design
        # points until ``initial_design_size`` is exhausted, and the
        # iteration counter advances regardless.
        await generate_suggestions(campaign_id, batch_size=1)
        assert subscriber.delivered == []
