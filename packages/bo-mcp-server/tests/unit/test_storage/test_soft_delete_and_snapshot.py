"""Soft-delete + suggestion-provenance snapshot contracts.

Pins the data-integrity guarantees:

* ``CampaignRepository.delete`` (and the suggestion / result equivalents)
  now flips a ``deleted_at`` timestamp instead of physically removing the
  row; standard reads hide soft-deleted rows but
  ``include_deleted=True`` keeps them queryable for audit / forensics.

* ``hard_delete`` physically removes the row. Because the FKs are
  ``ON DELETE RESTRICT`` (suggestions / results / events → campaigns),
  the hard delete fails with an integrity error if children still
  exist. SQLite test runs enable ``PRAGMA foreign_keys=ON`` so this is
  exercised here.

* ``ResultRepository`` captures the originating suggestion's
  parameter values + provenance into ``suggestion_snapshot_json`` so
  the BO context survives a later hard delete of the suggestion (the
  ``ON DELETE SET NULL`` FK on ``results.suggestion_id`` keeps the
  result row alive without losing audit data).

Reference for the soft-delete pattern:
https://www.cosmicpython.com/book/chapter_07_aggregate.html (DDD
aggregate boundary with explicit deletion semantics).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bo_mcp_server.domain import (
    Campaign,
    CampaignStatus,
    Result,
    ResultSource,
    Suggestion,
    SuggestionProvenance,
    SuggestionStatus,
    User,
)
from bo_mcp_server.storage.models import (
    Base,
    CampaignModel,
    CampaignSpecModel,
    ResultModel,
    SuggestionModel,
    UserModel,
)
from bo_mcp_server.storage.repositories import (
    CampaignRepository,
    ResultRepository,
    SuggestionRepository,
    UserRepository,
)

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def session() -> AsyncGenerator[AsyncSession]:
    """Per-test session with FK enforcement enabled.

    The production session factory deliberately leaves SQLite FK
    enforcement off (see :func:`bo_mcp_server.storage.database._create_engine_with_options`
    for the reasoning). This fixture enables the pragma locally so
    ``ON DELETE RESTRICT`` is exercised in the cascade-contract tests
    below; PostgreSQL enforces FKs unconditionally in production.
    """
    from sqlalchemy import event

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_fks(dbapi_connection, _record) -> None:
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


async def _seed_owner(session: AsyncSession) -> User:
    """Insert a baseline user via the production repository path."""
    user = User(
        name="Owner",
        email=f"owner-{uuid4()}@example.com",
        api_key_hash=f"hash-{uuid4()}",
    )
    await UserRepository(session).save(user)
    await session.commit()
    return user


async def _seed_campaign(session: AsyncSession, owner: User) -> Campaign:
    """Insert a baseline campaign (and matching spec) via the repos."""
    spec_id = uuid4()
    session.add(
        CampaignSpecModel(
            id=str(spec_id),
            name="Soft-delete spec",
            parameters_json='[{"name":"x","type":"continuous","bounds":[0,1]}]',
            objectives_json='[{"name":"y","direction":"minimize"}]',
        )
    )
    await session.flush()
    campaign = Campaign(
        spec_id=spec_id,
        owner_id=owner.id,
        status=CampaignStatus.RUNNING,
        version=1,
        iteration=0,
    )
    await CampaignRepository(session).save(campaign)
    await session.commit()
    return campaign


async def _seed_suggestion(session: AsyncSession, campaign: Campaign) -> Suggestion:
    """Insert a baseline pending suggestion."""
    suggestion = Suggestion(
        campaign_id=campaign.id,
        parameter_values={"x": 0.5},
        provenance=SuggestionProvenance(iteration=1, batch_index=0),
        status=SuggestionStatus.PENDING,
    )
    saved = await SuggestionRepository(session).save(suggestion)
    await session.commit()
    return saved


@pytest.mark.asyncio
async def test_soft_delete_hides_campaign_from_default_reads(session: AsyncSession) -> None:
    """A soft-deleted campaign disappears from ``get`` / ``list_*`` by default.

    ``delete()`` flips ``deleted_at`` instead of physically
    removing the row, so the campaign is gone for application reads
    but still reconstructable through ``include_deleted=True``.
    """
    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)

    repo = CampaignRepository(session)
    soft_deleted = await repo.delete(campaign.id)
    await session.commit()
    assert soft_deleted is True

    # Default reads hide it.
    assert await repo.get(campaign.id) is None
    assert campaign.id not in {c.id for c in await repo.list_by_owner(owner.id)}

    # Forensics path can still load it.
    raw = await repo.get(campaign.id, include_deleted=True)
    assert raw is not None
    assert raw.id == campaign.id


@pytest.mark.asyncio
async def test_soft_delete_keeps_audit_history_recoverable(session: AsyncSession) -> None:
    """Soft-deleted rows remain physically present for admin / audit queries.

    Verifies the underlying ``deleted_at`` column is populated rather
    than the row being removed: the same query without the soft-delete
    filter still returns the model.
    """
    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    await CampaignRepository(session).delete(campaign.id)
    await session.commit()

    raw = await session.execute(select(CampaignModel).where(CampaignModel.id == str(campaign.id)))
    model = raw.scalar_one()
    assert model.deleted_at is not None


@pytest.mark.asyncio
async def test_hard_delete_blocked_by_restrict_when_children_exist(
    session: AsyncSession,
) -> None:
    """Hard-deleting a campaign with live suggestions fails on RESTRICT.

    This change swaps the previous ``ON DELETE CASCADE`` for ``RESTRICT``
    on ``suggestions.campaign_id``: a forgotten cleanup that drops
    the parent without first removing the children now fails fast
    with an integrity error instead of silently destroying history.
    """
    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    await _seed_suggestion(session, campaign)

    async def _hard_delete_and_commit() -> None:
        await CampaignRepository(session).hard_delete(campaign.id)
        await session.commit()

    with pytest.raises(IntegrityError):
        await _hard_delete_and_commit()


@pytest.mark.asyncio
async def test_result_snapshot_persists_suggestion_provenance(
    session: AsyncSession,
) -> None:
    """The result row carries the originating suggestion's snapshot.

    Pins the snapshot wiring in ``ResultRepository.save``: the
    domain entity's ``suggestion_snapshot`` round-trips through the
    JSON column so a later hard-delete of the suggestion cannot
    erase the BO context.
    """
    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    suggestion = await _seed_suggestion(session, campaign)

    snapshot = {
        "suggestion_id": str(suggestion.id),
        "parameter_values": dict(suggestion.parameter_values),
        "provenance": suggestion.provenance.model_dump(mode="json"),
        "suggestion_created_at": suggestion.created_at.isoformat(),
    }
    result = Result(
        campaign_id=campaign.id,
        suggestion_id=suggestion.id,
        parameter_values={"x": 0.5},
        objective_values={"y": 0.123},
        source=ResultSource.API,
        submitted_by=owner.id,
        suggestion_snapshot=snapshot,
    )
    await ResultRepository(session).save(result)
    await session.commit()

    reloaded = await ResultRepository(session).get(result.id)
    assert reloaded is not None
    assert reloaded.suggestion_snapshot == snapshot


@pytest.mark.asyncio
async def test_snapshot_survives_suggestion_hard_delete(session: AsyncSession) -> None:
    """``ON DELETE SET NULL`` is exercised end-to-end, leaving the snapshot intact.

    Hard-deletes the suggestion directly. The result row's
    ``suggestion_id`` FK is declared ``ON DELETE SET NULL``
    (``packages/bo-mcp-server/src/bo_mcp_server/storage/models.py``)
    so the database clears the pointer automatically when the parent
    is removed; the test would fail if the FK had been mis-declared
    ``RESTRICT``, ``NO ACTION``, or omitted. The
    ``suggestion_snapshot_json`` column remains intact on the
    result row so the BO context is reconstructable from the
    surviving snapshot alone.
    """
    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    suggestion = await _seed_suggestion(session, campaign)

    snapshot = {
        "suggestion_id": str(suggestion.id),
        "parameter_values": dict(suggestion.parameter_values),
        "provenance": suggestion.provenance.model_dump(mode="json"),
        "suggestion_created_at": suggestion.created_at.isoformat(),
    }
    result = Result(
        campaign_id=campaign.id,
        suggestion_id=suggestion.id,
        parameter_values={"x": 0.5},
        objective_values={"y": 0.123},
        source=ResultSource.API,
        submitted_by=owner.id,
        suggestion_snapshot=snapshot,
    )
    await ResultRepository(session).save(result)
    await session.commit()

    # Hard-delete the parent. With PRAGMA foreign_keys=ON the
    # database honours the ``ON DELETE SET NULL`` clause and clears
    # the result's ``suggestion_id`` for us.
    deleted = await SuggestionRepository(session).hard_delete(suggestion.id)
    assert deleted is True
    await session.commit()

    reloaded = await ResultRepository(session).get(result.id)
    assert reloaded is not None
    assert reloaded.suggestion_id is None
    assert reloaded.suggestion_snapshot == snapshot


@pytest.mark.asyncio
async def test_suggestion_soft_delete_removes_from_actionable_list(
    session: AsyncSession,
) -> None:
    """A soft-deleted suggestion no longer occupies an actionable slot.

    ``list_actionable_by_campaign`` powers both the pending-budget
    reservation in ``generate_suggestions`` and the duplicate check in
    ``submit_results``; soft-deleted suggestions must not keep
    reserving capacity.
    """
    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    suggestion = await _seed_suggestion(session, campaign)

    repo = SuggestionRepository(session)
    actionable_before = await repo.list_actionable_by_campaign(campaign.id)
    assert suggestion.id in {s.id for s in actionable_before}

    await repo.delete(suggestion.id)
    await session.commit()

    actionable_after = await repo.list_actionable_by_campaign(campaign.id)
    assert suggestion.id not in {s.id for s in actionable_after}


@pytest.mark.asyncio
async def test_stale_suggestion_save_raises_and_preserves_row(
    session: AsyncSession,
) -> None:
    """A save against a tombstoned suggestion raises ``ConcurrentModificationError``.

    Catches both failure modes of the soft-delete contract:
    a naive ``session.merge`` would (a) clear ``deleted_at`` and
    resurrect the row, *and* (b) overwrite the historical ``status``
    and ``provenance_json`` columns that audit / forensics callers
    rely on. The fixed ``SuggestionRepository.save`` raises
    :class:`ConcurrentModificationError` when the existing row is
    soft-deleted so caller operations can surface a structured
    conflict envelope instead of reporting a phantom success; the
    persisted row is byte-identical to its pre-tombstone snapshot.
    """
    from bo_mcp_server.storage import ConcurrentModificationError

    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    suggestion = await _seed_suggestion(session, campaign)
    repo = SuggestionRepository(session)

    pre_delete = await session.execute(
        select(SuggestionModel).where(SuggestionModel.id == str(suggestion.id))
    )
    pre_delete_model = pre_delete.scalar_one()
    pre_status = pre_delete_model.status
    pre_provenance_json = pre_delete_model.provenance_json
    pre_parameter_values_json = pre_delete_model.parameter_values_json

    await repo.delete(suggestion.id)
    await session.commit()
    assert await repo.get(suggestion.id) is None

    # Stale save with a *different* status + provenance to make a
    # silent overwrite immediately visible if it happened.
    mutated = suggestion.with_status(SuggestionStatus.ACCEPTED).model_copy(
        update={
            "parameter_values": {"x": 0.999},
            "provenance": SuggestionProvenance(iteration=999, batch_index=42),
        }
    )
    with pytest.raises(ConcurrentModificationError):
        await repo.save(mutated)
    await session.rollback()

    raw = await session.execute(
        select(SuggestionModel).where(SuggestionModel.id == str(suggestion.id))
    )
    model = raw.scalar_one()
    assert model.deleted_at is not None
    assert model.status == pre_status
    assert model.provenance_json == pre_provenance_json
    assert model.parameter_values_json == pre_parameter_values_json


@pytest.mark.asyncio
async def test_stale_result_save_raises_and_preserves_row(
    session: AsyncSession,
) -> None:
    """The same raise-and-preserve contract applies to ``Result``."""
    from bo_mcp_server.storage import ConcurrentModificationError

    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    result = Result(
        campaign_id=campaign.id,
        parameter_values={"x": 0.5},
        objective_values={"y": 0.0},
        source=ResultSource.API,
        submitted_by=owner.id,
        metadata={"note": "original"},
    )
    repo = ResultRepository(session)
    await repo.save(result)
    await session.commit()

    pre_delete = await session.execute(select(ResultModel).where(ResultModel.id == str(result.id)))
    pre_delete_model = pre_delete.scalar_one()
    pre_objective_values_json = pre_delete_model.objective_values_json
    pre_metadata_json = pre_delete_model.metadata_json

    await repo.delete(result.id)
    await session.commit()
    assert await repo.get(result.id) is None

    mutated = result.model_copy(
        update={
            "objective_values": {"y": 99.0},
            "metadata": {"note": "tampered"},
        }
    )
    with pytest.raises(ConcurrentModificationError):
        await repo.save(mutated)
    await session.rollback()

    raw = await session.execute(select(ResultModel).where(ResultModel.id == str(result.id)))
    model = raw.scalar_one()
    assert model.deleted_at is not None
    assert model.objective_values_json == pre_objective_values_json
    assert model.metadata_json == pre_metadata_json


@pytest.mark.asyncio
async def test_suggestion_save_is_atomic_under_concurrent_delete(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A soft-delete that lands mid-save raises CMR atomically.

    Without the atomic ``UPDATE … WHERE deleted_at IS NULL``, a
    concurrent ``delete()`` between the existence check and the
    write could silently resurrect the tombstone. The race is
    modelled by intercepting the existence-check ``SELECT`` and
    issuing an inline soft-delete on the same session before the
    save's UPDATE runs. Statements in SQLite serialize within one
    connection, so the subsequent
    ``UPDATE … WHERE deleted_at IS NULL`` sees the tombstone and
    reports ``rowcount == 0`` — the exact race the friend's audit
    called out.
    """
    from sqlalchemy import text

    from bo_mcp_server.domain.utils import utcnow
    from bo_mcp_server.storage import ConcurrentModificationError

    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    suggestion = await _seed_suggestion(session, campaign)
    repo = SuggestionRepository(session)

    original_execute = AsyncSession.execute
    injected = {"done": False}

    async def racing_execute(self, stmt, *args, **kwargs):
        result = await original_execute(self, stmt, *args, **kwargs)
        compiled = str(stmt).upper()
        if not injected["done"] and "SELECT" in compiled and "SUGGESTIONS.ID" in compiled:
            injected["done"] = True
            await original_execute(
                self,
                text("UPDATE suggestions SET deleted_at = :ts WHERE id = :id"),
                {"ts": utcnow(), "id": str(suggestion.id)},
            )
        return result

    monkeypatch.setattr(AsyncSession, "execute", racing_execute)
    with pytest.raises(ConcurrentModificationError):
        await repo.save(suggestion.with_status(SuggestionStatus.ACCEPTED))
    monkeypatch.undo()

    raw = await session.execute(
        select(SuggestionModel).where(SuggestionModel.id == str(suggestion.id))
    )
    model = raw.scalar_one()
    assert model.deleted_at is not None
    assert model.status == SuggestionStatus.PENDING


@pytest.mark.asyncio
async def test_result_save_is_atomic_under_concurrent_delete(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same atomic-guard contract for ``ResultRepository.save``."""
    from sqlalchemy import text

    from bo_mcp_server.domain.utils import utcnow
    from bo_mcp_server.storage import ConcurrentModificationError

    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    result = Result(
        campaign_id=campaign.id,
        parameter_values={"x": 0.5},
        objective_values={"y": 1.0},
        source=ResultSource.API,
        submitted_by=owner.id,
    )
    repo = ResultRepository(session)
    await repo.save(result)
    await session.commit()

    original_execute = AsyncSession.execute
    injected = {"done": False}

    async def racing_execute(self, stmt, *args, **kwargs):
        response = await original_execute(self, stmt, *args, **kwargs)
        compiled = str(stmt).upper()
        if not injected["done"] and "SELECT" in compiled and "RESULTS.ID" in compiled:
            injected["done"] = True
            await original_execute(
                self,
                text("UPDATE results SET deleted_at = :ts WHERE id = :id"),
                {"ts": utcnow(), "id": str(result.id)},
            )
        return response

    monkeypatch.setattr(AsyncSession, "execute", racing_execute)
    mutated = result.model_copy(update={"objective_values": {"y": 999.0}})
    with pytest.raises(ConcurrentModificationError):
        await repo.save(mutated)
    monkeypatch.undo()

    raw = await session.execute(select(ResultModel).where(ResultModel.id == str(result.id)))
    model = raw.scalar_one()
    assert model.deleted_at is not None
    assert model.objective_values_json == '{"y": 1.0}'


@pytest.mark.asyncio
async def test_stale_campaign_save_raises_and_preserves_tombstone(
    session: AsyncSession,
) -> None:
    """A save against a tombstoned campaign raises ``ConcurrentModificationError``.

    Mirrors the suggestion / result tombstone-save contract for
    :class:`CampaignRepository.save`. Without the ``deleted_at IS NULL``
    clause on the OCC update, a long-running operation could read an
    active campaign, then commit fresh status / iteration / backend
    state into a tombstoned row after a concurrent
    :meth:`CampaignRepository.delete`. The persisted row must be
    byte-identical to its pre-tombstone snapshot.
    """
    from bo_mcp_server.storage import ConcurrentModificationError

    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    repo = CampaignRepository(session)

    pre_delete = await session.execute(
        select(CampaignModel).where(CampaignModel.id == str(campaign.id))
    )
    pre_delete_model = pre_delete.scalar_one()
    pre_status = pre_delete_model.status
    pre_iteration = pre_delete_model.iteration
    pre_version = pre_delete_model.version

    await repo.delete(campaign.id)
    await session.commit()
    assert await repo.get(campaign.id) is None

    mutated = campaign.with_status(CampaignStatus.PAUSED).model_copy(
        update={"iteration": 42, "version": pre_version + 1}
    )
    with pytest.raises(ConcurrentModificationError):
        await repo.save(mutated, expected_version=pre_version)
    await session.rollback()

    raw = await session.execute(select(CampaignModel).where(CampaignModel.id == str(campaign.id)))
    model = raw.scalar_one()
    assert model.deleted_at is not None
    assert model.status == pre_status
    assert model.iteration == pre_iteration
    assert model.version == pre_version


@pytest.mark.asyncio
async def test_db_constraint_permits_admin_replacement_after_result_soft_delete(
    session: AsyncSession,
) -> None:
    """DB constraint permits a replacement result for a soft-deleted row.

    Narrow scope: this pins the partial-unique-index predicate change
    (``suggestion_id IS NOT NULL AND deleted_at IS NULL``) shipped by
    migration ``014_results_unique_active``. The replacement INSERT
    is exercised directly through ``ResultRepository.save``, so the
    test passes when the *database* allows the second row.

    Application-level re-submission is a separate concern and is
    *not* covered here: the normal ``submit_results`` path also
    marks the underlying suggestion ``COMPLETED`` and later rejects
    non-actionable suggestions, so a real client cannot resubmit
    against the same suggestion without an additional admin
    restore-the-suggestion path. The migration deliberately fixes
    only the constraint so the admin / forensics replacement story
    works; the application story is intentionally unchanged.
    """
    owner = await _seed_owner(session)
    campaign = await _seed_campaign(session, owner)
    suggestion = await _seed_suggestion(session, campaign)

    first = Result(
        campaign_id=campaign.id,
        suggestion_id=suggestion.id,
        parameter_values={"x": 0.5},
        objective_values={"y": 1.0},
        source=ResultSource.API,
        submitted_by=owner.id,
    )
    repo = ResultRepository(session)
    await repo.save(first)
    await session.commit()

    await repo.delete(first.id)
    await session.commit()

    replacement = Result(
        campaign_id=campaign.id,
        suggestion_id=suggestion.id,
        parameter_values={"x": 0.5},
        objective_values={"y": 2.0},
        source=ResultSource.API,
        submitted_by=owner.id,
    )
    await repo.save(replacement)
    await session.commit()

    reloaded = await repo.get(replacement.id)
    assert reloaded is not None
    assert reloaded.objective_values == {"y": 2.0}


@pytest.mark.asyncio
async def test_user_repository_still_uses_hard_delete(session: AsyncSession) -> None:
    """Users have no cascade exposure; ``delete()`` remains a hard remove.

    Pins the contract noted in the soft-delete audit: only
    ``Campaign`` / ``Suggestion`` / ``Result`` entities switched to
    soft delete because they are part of the cascade-exposed graph.
    ``User`` deletion already uses ``ON DELETE RESTRICT`` on the
    parent FK, so a user with active campaigns cannot be removed and
    a user with no campaigns is safely removed outright.
    """
    user = User(
        name="To delete",
        email=f"to-delete-{uuid4()}@example.com",
        api_key_hash=f"hash-{uuid4()}",
    )
    saved = await UserRepository(session).save(user)
    await session.commit()

    removed = await UserRepository(session).delete(saved.id)
    await session.commit()
    assert removed is True

    row = await session.execute(select(UserModel).where(UserModel.id == str(saved.id)))
    assert row.scalar_one_or_none() is None


# ``UUID`` is exported for parity with sibling test modules even though
# the tests above do not construct one directly.
_ = UUID
