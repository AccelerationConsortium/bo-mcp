"""SQLite engine contract: pragmas, UTC timestamps, and index mirroring.

The default deployment (no ``DATABASE_URL`` set) runs on SQLite, so the
engine must provide the same guarantees PostgreSQL gives production
unconditionally:

* ``PRAGMA foreign_keys=ON`` — SQLite ships with FK enforcement OFF per
  connection; without the pragma the schema's ``ON DELETE RESTRICT`` /
  ``SET NULL`` contracts silently do not hold and ``hard_delete`` can
  orphan suggestions/results.
  Reference: https://www.sqlite.org/foreignkeys.html#fk_enable ("Foreign
  key constraints are disabled by default … must be enabled separately
  for each database connection").
* ``PRAGMA journal_mode=WAL`` + ``PRAGMA busy_timeout`` — concurrent
  writers on the rollback-journal default hit immediate ``database is
  locked`` errors that never occur on PostgreSQL.
  Reference: https://www.sqlite.org/wal.html and
  https://www.sqlite.org/pragma.html#pragma_busy_timeout.
* Timestamps must round-trip timezone-aware: SQLite's
  ``DateTime(timezone=True)`` returns naive datetimes, so without the
  shared ``UTCDateTime`` decorator every ``isoformat()`` emit site flips
  wire format between SQLite and PostgreSQL deployments and offset-less
  strings get parsed as *local* time by RFC-3339/JS consumers.
  Reference: SQLAlchemy DateTime docs — "The timezone flag has no effect
  on the SQLite dialect" —
  https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.DateTime.
* The ORM metadata must mirror every index its migrations create, or the
  next ``alembic --autogenerate`` proposes dropping a load-bearing index
  and SQLite ``create_all`` deployments never get it.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from bo_mcp_server.domain import (
    CampaignSpec,
    InputParameter,
    Objective,
    ParameterType,
    Suggestion,
    SuggestionProvenance,
    User,
)
from bo_mcp_server.domain.campaign import Campaign, CampaignStatus
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    SuggestionRepository,
    UserRepository,
    close_database,
    get_session,
    init_database,
)
from bo_mcp_server.storage.models import CampaignModel, IdempotencyCacheModel

pytestmark = pytest.mark.usefixtures("fresh_database")


@pytest_asyncio.fixture
async def fresh_database() -> AsyncGenerator[None]:
    """Reset the engine + session factory between tests."""
    await close_database()
    await init_database()
    try:
        yield
    finally:
        await close_database()


async def _seed_user_and_spec() -> tuple[User, str]:
    """Persist a user and spec, returning them for campaign FK wiring."""
    async with get_session() as session:
        unique = str(uuid4())
        user = await UserRepository(session).save(
            User(
                name="Owner",
                email=f"owner-{unique}@example.com",
                api_key_hash=hashlib.sha256(unique.encode()).hexdigest(),
            )
        )
        spec = CampaignSpec(
            name="Engine Contract",
            parameters=(
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ),
            objectives=(Objective(name="y", direction="maximize"),),
        )
        spec_id = uuid4()
        await CampaignSpecRepository(session).save(spec, spec_id=spec_id)
    return user, str(spec_id)


# ---------------------------------------------------------------------------
# Pragmas on the shared (default) engine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_engine_enables_foreign_keys_and_busy_timeout() -> None:
    """The default engine applies FK enforcement and the busy timeout."""
    from bo_mcp_server.settings import get_sqlite_busy_timeout_ms

    async with get_session() as session:
        fk = (await session.execute(text("PRAGMA foreign_keys"))).scalar_one()
        busy = (await session.execute(text("PRAGMA busy_timeout"))).scalar_one()
    assert fk == 1
    assert busy == get_sqlite_busy_timeout_ms()


@pytest.mark.asyncio
async def test_file_backed_engine_uses_wal(tmp_path, monkeypatch) -> None:
    """A file-backed SQLite engine runs in WAL journal mode.

    In-memory databases ignore the WAL pragma (journal_mode stays
    ``memory``), so the WAL assertion needs a real file.
    """
    db_file = tmp_path / "wal_probe.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    await close_database()
    try:
        await init_database()
        async with get_session() as session:
            mode = (await session.execute(text("PRAGMA journal_mode"))).scalar_one()
        assert mode == "wal"
    finally:
        await close_database()


async def _insert_dangling_campaign() -> None:
    """Insert a campaign row whose parent FKs reference nothing."""
    async with get_session() as session:
        session.add(
            CampaignModel(
                id=str(uuid4()),
                spec_id=str(uuid4()),
                owner_id=str(uuid4()),
                status=CampaignStatus.CREATED,
            )
        )
        await session.flush()


@pytest.mark.asyncio
async def test_dangling_campaign_fk_rejected_on_default_engine() -> None:
    """Inserting a campaign with synthetic parent UUIDs fails loudly.

    Before the pragma promotion this silently succeeded, which is why
    fixtures could get away with dangling ``owner_id`` / ``spec_id``
    values — and why real orphaning went unnoticed.
    """
    with pytest.raises(IntegrityError):
        await _insert_dangling_campaign()


@pytest.mark.asyncio
async def test_hard_delete_with_children_raises_integrity_error() -> None:
    """``hard_delete`` honors ``ON DELETE RESTRICT`` on the default engine.

    Pins the contract stated in the ``CampaignRepository.hard_delete``
    docstring: child rows must block the physical delete instead of
    being orphaned.
    """
    user, spec_id = await _seed_user_and_spec()
    async with get_session() as session:
        campaign = Campaign(
            spec_id=spec_id,  # ty: ignore[invalid-argument-type]
            owner_id=user.id,
            status=CampaignStatus.RUNNING,
        )
        await CampaignRepository(session).save(campaign)
        await SuggestionRepository(session).save(
            Suggestion(
                campaign_id=campaign.id,
                parameter_values={"x": 0.5},
                provenance=SuggestionProvenance(
                    iteration=1, batch_index=0, generation_method="initial_design"
                ),
            )
        )

    async def _hard_delete_with_children() -> None:
        async with get_session() as session:
            await CampaignRepository(session).hard_delete(campaign.id)
            await session.flush()

    with pytest.raises(IntegrityError):
        await _hard_delete_with_children()


# ---------------------------------------------------------------------------
# UTC-aware timestamp round-trips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timestamps_round_trip_timezone_aware() -> None:
    """Entities read back from SQLite carry ``tzinfo`` on their timestamps."""
    user, spec_id = await _seed_user_and_spec()
    async with get_session() as session:
        campaign = Campaign(
            spec_id=spec_id,  # ty: ignore[invalid-argument-type]
            owner_id=user.id,
            status=CampaignStatus.RUNNING,
        )
        await CampaignRepository(session).save(campaign)

    async with get_session() as session:
        loaded_user = await UserRepository(session).get(user.id)
        loaded_campaign = await CampaignRepository(session).get(campaign.id)

    assert loaded_user is not None
    assert loaded_campaign is not None
    assert loaded_user.created_at.tzinfo is not None
    assert loaded_campaign.created_at.tzinfo is not None
    assert loaded_campaign.updated_at.tzinfo is not None
    # The rendered wire format now always carries a UTC designator.
    assert loaded_campaign.created_at.isoformat().endswith("+00:00")


@pytest.mark.asyncio
async def test_non_utc_aware_timestamp_preserves_instant() -> None:
    """An aware non-UTC input is stored as UTC, not as bare wall-clock.

    Without bind-side normalization SQLite would strip the ``+02:00``
    offset and store the wall-clock digits, silently shifting the
    instant by the offset on read-back.
    """
    user, spec_id = await _seed_user_and_spec()
    plus_two = timezone(timedelta(hours=2))
    created_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=plus_two)
    async with get_session() as session:
        campaign = Campaign(
            spec_id=spec_id,  # ty: ignore[invalid-argument-type]
            owner_id=user.id,
            status=CampaignStatus.CREATED,
            created_at=created_at,
            updated_at=created_at,
        )
        await CampaignRepository(session).save(campaign)

    async with get_session() as session:
        loaded = await CampaignRepository(session).get(campaign.id)

    assert loaded is not None
    assert loaded.created_at == created_at.astimezone(UTC)


# ---------------------------------------------------------------------------
# Metadata mirrors migration-created indexes
# ---------------------------------------------------------------------------


def test_idempotency_cache_model_mirrors_expires_at_index() -> None:
    """The ORM metadata declares the GC-sweep index its migration created.

    Without the mirror, ``alembic --autogenerate`` proposes dropping
    ``ix_idempotency_cache_expires_at`` and SQLite ``create_all``
    deployments run the hourly GC as a full-table scan.
    """
    table = IdempotencyCacheModel.metadata.tables["idempotency_cache"]
    index_names = {index.name for index in table.indexes}
    assert "ix_idempotency_cache_expires_at" in index_names
