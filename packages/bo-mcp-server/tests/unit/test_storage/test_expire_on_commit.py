"""Stale-instance reliance guards for the production session factory.

Reference: SQLAlchemy's default ``expire_on_commit=True`` expires every
ORM attribute on commit so the next access reloads fresh state. The
production session factory restores this default to avoid silent stale
reads after a commit; this test pins both invariants.

See https://docs.sqlalchemy.org/en/20/orm/session_api.html#sqlalchemy.orm.Session.params.expire_on_commit
for the SQLAlchemy reference behaviour: post-commit attribute access on
an expired instance triggers a refresh (or, on async sessions without
an awaitable context, ``DetachedInstanceError`` /
``MissingGreenlet``).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import DetachedInstanceError

from bo_mcp_server.domain import User
from bo_mcp_server.domain.utils import utcnow
from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    UserRepository,
    close_database,
    get_session,
    init_database,
)
from bo_mcp_server.storage.database import _get_session_factory
from bo_mcp_server.storage.models import (
    IdempotencyCacheModel,
)


@pytest_asyncio.fixture
async def fresh_database() -> AsyncGenerator[None]:
    """Reset the engine + session factory between tests.

    The production session-factory singleton is the unit under test;
    closing and re-initialising the database here exercises the same
    bootstrap path production uses.
    """
    await close_database()
    await init_database()
    try:
        yield
    finally:
        await close_database()


@pytest.mark.asyncio
async def test_session_factory_uses_expire_on_commit_true(fresh_database: None) -> None:
    """Production sessions opt into SQLAlchemy's default expire-on-commit.

    Stale ORM-instance reads are silent — they return cached field
    values that diverge from durable truth. The fix is to keep the
    SQLAlchemy default, which expires every loaded instance on commit
    so subsequent attribute access goes back to the database.
    """
    factory = _get_session_factory()
    # ``kw`` carries the keyword arguments forwarded to ``Session``;
    # we want to pin the explicit ``expire_on_commit=True`` choice.
    assert factory.kw.get("expire_on_commit") is True


@pytest.mark.asyncio
async def test_second_session_observes_fresh_state_after_commit(fresh_database: None) -> None:
    """Two sessions see committed state, not pre-commit cached values.

    Reproducer for the stale-read hazard called out in TODO 8.10: a
    second reader after a first session's commit must observe the
    persisted value, not whatever the first session held before
    flushing. With ``expire_on_commit=True`` this is the SQLAlchemy
    default behaviour; the test pins it so a regression to ``False``
    surfaces immediately.
    """
    user_id = uuid4()
    async with get_session() as session:
        user_repo = UserRepository(session)
        user = await user_repo.save(
            User(
                id=user_id,
                name="Initial",
                email=f"u-{user_id}@example.com",
                api_key_hash=f"hash-{user_id}",
            )
        )

    async with get_session() as second_session:
        fresh = await UserRepository(second_session).get(user.id)
    assert fresh is not None
    assert fresh.name == "Initial"


@pytest.mark.asyncio
async def test_orm_attribute_access_after_session_close_raises(
    fresh_database: None,
) -> None:
    """Reading an ORM attribute outside its session is a programming bug.

    With ``expire_on_commit=True`` an instance loaded inside a session
    is expired on the way out of ``async with get_session()``; touching
    its columns afterwards raises ``DetachedInstanceError`` rather than
    silently returning stale data. This is the exact failure mode that
    surfaced the bug fixed inside ``_read_existing``: that code used
    to project ``row.expires_at`` outside the session, returning stale
    values silently with ``expire_on_commit=False``.
    """
    now = utcnow()
    row_id = uuid4()
    async with get_session() as session:
        session.add(
            IdempotencyCacheModel(
                tool_name="diag-tool",
                idempotency_key=f"key-{row_id}",
                request_hash=f"hash-{row_id}",
                reservation_token=None,
                response_json="{}",
                created_at=now,
                expires_at=now + timedelta(seconds=60),
            )
        )

    async with get_session() as session:
        from sqlalchemy import select  # noqa: PLC0415

        result = await session.execute(
            select(IdempotencyCacheModel).where(
                IdempotencyCacheModel.idempotency_key == f"key-{row_id}"
            )
        )
        row = result.scalar_one()

    with pytest.raises(DetachedInstanceError):
        _ = row.expires_at


@pytest.mark.asyncio
async def test_apply_idempotency_replays_after_session_recycle(
    fresh_database: None,
) -> None:
    """Cache hit on a retry must work even after the writer's session expired.

    Regression test for the ``DetachedInstanceError`` that
    ``_read_existing`` triggered once production flipped to
    ``expire_on_commit=True``: the second call here goes through
    ``_read_existing`` → ``_project_row(row, ...)`` on a row that was
    written by the first call's session (now expired). If projection
    happens outside the read session the second call raises.
    """
    calls = 0

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"success": True, "iteration": calls}

    payload = {"campaign_id": "abc", "n": 1}
    first = await apply_idempotency(
        tool_name="diag_replay",
        idempotency_key="replay-key",
        request_payload=payload,
        executor=run,
    )
    second = await apply_idempotency(
        tool_name="diag_replay",
        idempotency_key="replay-key",
        request_payload=payload,
        executor=run,
    )

    assert calls == 1
    assert first["iteration"] == 1
    assert second["iteration"] == 1
    assert second.get("idempotency_replay") is True


@pytest.mark.asyncio
async def test_campaign_repository_save_enforces_optimistic_concurrency(
    fresh_database: None,
) -> None:
    """Mutating-entity repositories must guard against lost updates.

    The audit follow-up to TODO 8.10 asks for a contract test that
    every repository with a mutation path wraps the write with version
    optimistic concurrency (or an equivalent guard). ``Campaign`` is
    currently the only entity with a mutating ``save`` path:
    ``Suggestion``/``Result`` are write-once and their lifecycle goes
    through state-machine-validated operations, ``User`` is rewritten
    via ``merge`` which is last-write-wins by design. This test pins
    the OCC contract for campaigns; if a new mutating entity ships
    without a similar guard it should grow its own contract test
    here.
    """
    from bo_mcp_server.domain import (  # noqa: PLC0415
        Campaign,
        CampaignSpec,
        CampaignStatus,
        InputParameter,
        Objective,
        ParameterType,
        User,
    )
    from bo_mcp_server.storage import ConcurrentModificationError  # noqa: PLC0415

    user = User(name="Owner", email=f"owner-{uuid4()}@example.com", api_key_hash="hash")
    spec = CampaignSpec(
        name="OCC Test",
        parameters=(
            InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
        ),
        objectives=(Objective(name="y", direction="minimize"),),
    )
    spec_id = uuid4()

    async with get_session() as session:
        await UserRepository(session).save(user)
        await CampaignSpecRepository(session).save(spec, spec_id=spec_id)
        campaign = Campaign(
            spec_id=spec_id,
            owner_id=user.id,
            status=CampaignStatus.CREATED,
            version=1,
            iteration=0,
        )
        repo = CampaignRepository(session)
        await repo.save(campaign)

    # First update: bumps version 1 -> 2.
    async with get_session() as session:
        repo = CampaignRepository(session)
        current = await repo.get(campaign.id)
        assert current is not None
        updated = current.with_status(CampaignStatus.RUNNING)
        await repo.save(updated, expected_version=current.version)

    # Second update against the original (now stale) version must lose.
    async with get_session() as session:
        repo = CampaignRepository(session)
        with pytest.raises(ConcurrentModificationError):
            stale = campaign.with_status(CampaignStatus.PAUSED)
            await repo.save(stale, expected_version=1)


# Pin the asyncio event loop scope so the fresh-database fixture's
# teardown runs on the same loop as the per-test sessions.
asyncio_default_fixture_loop_scope = "function"
