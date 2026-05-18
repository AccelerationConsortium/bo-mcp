"""Atomic suggestion-status transition contract.

TODO 8.11 follow-up: status updates used to read the row, validate
the transition in Python, then write through ``SuggestionRepository.save``.
Two concurrent transitions from the same source state (e.g.
``PENDING -> ACCEPTED`` and ``PENDING -> REJECTED``) could both pass
the in-Python validation, then race to overwrite each other —
last-writer-wins with no signal to the loser. ``transition_status``
folds the expected-source-status guard into the SQL ``UPDATE``, so
only one of the racing callers succeeds and the other observes
``rowcount == 0``.

Reference: classic CAS pattern applied at the SQL layer. See
https://martinfowler.com/eaaCatalog/optimisticOfflineLock.html for
the design motivation.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bo_mcp_server.domain import (
    Campaign,
    CampaignStatus,
    Suggestion,
    SuggestionProvenance,
    SuggestionStatus,
    User,
)
from bo_mcp_server.storage.models import (
    Base,
    CampaignSpecModel,
    SuggestionModel,
)
from bo_mcp_server.storage.repositories import (
    CampaignRepository,
    SuggestionRepository,
    UserRepository,
)

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def session() -> AsyncGenerator[AsyncSession]:
    """Per-test session with FK enforcement enabled."""
    from sqlalchemy import event  # noqa: PLC0415

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_fks(dbapi_connection, _record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=True)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed_pending(session: AsyncSession) -> Suggestion:
    user = User(name="Owner", email=f"o-{uuid4()}@example.com", api_key_hash="h")
    spec_id = uuid4()
    await UserRepository(session).save(user)
    session.add(
        CampaignSpecModel(
            id=str(spec_id),
            name="Status transition test",
            parameters_json='[{"name":"x","type":"continuous","bounds":[0,1]}]',
            objectives_json='[{"name":"y","direction":"minimize"}]',
        )
    )
    await session.flush()
    campaign = Campaign(
        spec_id=spec_id,
        owner_id=user.id,
        status=CampaignStatus.RUNNING,
        version=1,
        iteration=0,
    )
    await CampaignRepository(session).save(campaign)
    suggestion = Suggestion(
        campaign_id=campaign.id,
        parameter_values={"x": 0.5},
        provenance=SuggestionProvenance(iteration=1, batch_index=0),
        status=SuggestionStatus.PENDING,
    )
    await SuggestionRepository(session).save(suggestion)
    await session.commit()
    return suggestion


@pytest.mark.asyncio
async def test_first_transition_wins_concurrent_loser_reports_false(
    session: AsyncSession,
) -> None:
    """Two transitions from ``PENDING`` — only the first updates the row.

    Both callers' in-Python validation would accept the transition
    (the row is ``PENDING`` at read time), but the SQL ``WHERE
    status = PENDING`` predicate is enforced atomically at execution
    time. The first ``UPDATE`` flips the status; the second's
    predicate no longer matches and it returns ``False``.
    """
    suggestion = await _seed_pending(session)
    repo = SuggestionRepository(session)

    first = await repo.transition_status(
        suggestion.id, SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED
    )
    second = await repo.transition_status(
        suggestion.id, SuggestionStatus.PENDING, SuggestionStatus.REJECTED
    )
    await session.commit()

    assert first is True
    assert second is False

    raw = await session.execute(
        select(SuggestionModel).where(SuggestionModel.id == str(suggestion.id))
    )
    model = raw.scalar_one()
    assert model.status == SuggestionStatus.ACCEPTED


@pytest.mark.asyncio
async def test_transition_status_accepts_set_of_source_states(
    session: AsyncSession,
) -> None:
    """``submit_results`` completes either PENDING or ACCEPTED atomically."""
    suggestion = await _seed_pending(session)
    repo = SuggestionRepository(session)

    # PENDING → COMPLETED via the multi-source predicate.
    ok = await repo.transition_status(
        suggestion.id,
        (SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED),
        SuggestionStatus.COMPLETED,
    )
    await session.commit()
    assert ok is True

    raw = await session.execute(
        select(SuggestionModel).where(SuggestionModel.id == str(suggestion.id))
    )
    model = raw.scalar_one()
    assert model.status == SuggestionStatus.COMPLETED

    # A repeat transition cannot fire (status is now COMPLETED).
    again = await repo.transition_status(
        suggestion.id,
        (SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED),
        SuggestionStatus.COMPLETED,
    )
    assert again is False


@pytest.mark.asyncio
async def test_transition_status_refuses_soft_deleted_rows(
    session: AsyncSession,
) -> None:
    """The ``deleted_at IS NULL`` predicate is part of the conditional UPDATE."""
    suggestion = await _seed_pending(session)
    repo = SuggestionRepository(session)

    await repo.delete(suggestion.id)
    await session.commit()

    ok = await repo.transition_status(
        suggestion.id, SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED
    )
    await session.commit()
    assert ok is False

    raw = await session.execute(
        select(SuggestionModel).where(SuggestionModel.id == str(suggestion.id))
    )
    model = raw.scalar_one()
    assert model.deleted_at is not None
    assert model.status == SuggestionStatus.PENDING
