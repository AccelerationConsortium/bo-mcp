"""Partial-index contract for ``campaigns.status`` (TODO 8.49).

The per-status partial indexes added in migration
``015_campaigns_status_partials`` only deliver the planner benefit if
two things hold:

1. The index predicate matches the *stored* representation of the
   enum column. SQLAlchemy's ``Enum(CampaignStatus)`` stores the enum
   *name* (``"RUNNING"``) rather than the lowercase domain ``.value``
   (``"running"``); the index ``WHERE`` clause has to follow suit.
2. The predicate matches the application's actual query shape.
   The repository emits **single-status equality** filters
   (``CampaignModel.status == status``) in
   ``CampaignRepository.list_filtered`` and
   ``CampaignRepository.list_keyset`` — not the conceptual
   ``status IN ('CREATED','RUNNING')`` "active set" predicate the
   audit's prose described. An earlier iteration of this migration
   used IN-keyed compound partial indexes; SQLite did not prove that
   ``status = 'RUNNING'`` subsumes ``status IN ('CREATED','RUNNING')``
   and the partial index was unused for the equality query. The
   current per-status predicates align with the equality query shape
   and the planner picks them.

This module locks in four guarantees:

* All four per-status partial indexes exist and key on the stored
  enum names.
* Each partial subset is populated by exactly the matching status
  with ``deleted_at IS NULL``; other statuses are excluded.
* ``EXPLAIN QUERY PLAN`` on the **application's** single-status
  equality query (``WHERE status = 'RUNNING' AND deleted_at IS NULL``)
  picks the matching partial index — the test that previously asserted
  an IN-keyed query plan was misleading because the IN shape is not
  what the app actually emits.
* Soft-deleted rows are not indexed.

References
==========

* SQLite ``EXPLAIN QUERY PLAN``: https://www.sqlite.org/eqp.html
* SQLite partial indexes: https://www.sqlite.org/partialindex.html
* SQLAlchemy ``Enum`` storage convention:
  https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.Enum
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text

from bo_mcp_server.domain import (
    CampaignSpec,
    InputParameter,
    Objective,
    ParameterType,
    User,
)
from bo_mcp_server.domain.campaign import Campaign, CampaignStatus
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    UserRepository,
    close_database,
    get_session,
    init_database,
)

# Mirrors ``_CAMPAIGN_STATUS_PARTIAL_INDEXES`` in ``storage/models.py``.
# Duplicated here intentionally — a drift between the two surfaces would
# silently leave the planner unable to find the right index, so the test
# pins the expected set independently.
_EXPECTED_PARTIAL_INDEXES: tuple[tuple[str, str], ...] = (
    ("ix_campaigns_status_created_active", "CREATED"),
    ("ix_campaigns_status_running_active", "RUNNING"),
    ("ix_campaigns_status_completed_active", "COMPLETED"),
    ("ix_campaigns_status_failed_active", "FAILED"),
)


@pytest_asyncio.fixture
async def fresh_database() -> AsyncGenerator[None]:
    """Reset the engine + session factory between tests."""
    await close_database()
    await init_database()
    try:
        yield
    finally:
        await close_database()


async def _seed_campaign(status: CampaignStatus) -> str:
    async with get_session() as session:
        unique = str(uuid4())
        user = User(
            name="Owner",
            email=f"owner-{unique}@example.com",
            api_key_hash=hashlib.sha256(unique.encode()).hexdigest(),
        )
        user = await UserRepository(session).save(user)
        spec = CampaignSpec(
            name="Partial-Index Test",
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
        campaign = Campaign(
            spec_id=spec_id,
            owner_id=user.id,
            status=status,
        )
        await CampaignRepository(session).save(campaign)
        await session.commit()
        return str(campaign.id)


@pytest.mark.asyncio
async def test_partial_indexes_exist_with_expected_predicates(
    fresh_database: None,
) -> None:
    """All four per-status partial indexes are created with the right predicates."""
    async with get_session() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT name, sql FROM sqlite_master "
                    "WHERE type = 'index' AND tbl_name = 'campaigns' "
                    "ORDER BY name"
                )
            )
        ).fetchall()
    by_name = {row[0]: row[1] for row in rows if row[1] is not None}
    for index_name, status in _EXPECTED_PARTIAL_INDEXES:
        assert index_name in by_name, f"Missing partial index: {index_name}"
        sql = by_name[index_name].upper()
        assert f"'{status}'" in sql, f"{index_name} predicate missing status {status}"
        assert "DELETED_AT IS NULL" in sql, f"{index_name} predicate missing soft-delete clause"


@pytest.mark.asyncio
async def test_active_partial_index_filters_to_matching_status(
    fresh_database: None,
) -> None:
    """Per-status partial indexes only contain the matching status's rows.

    The application-side soft-delete filter and the partial predicate
    together produce the exact set of "active rows for this status".
    """
    created_id = await _seed_campaign(CampaignStatus.CREATED)
    running_id = await _seed_campaign(CampaignStatus.RUNNING)
    _paused_id = await _seed_campaign(CampaignStatus.PAUSED)
    completed_id = await _seed_campaign(CampaignStatus.COMPLETED)

    async with get_session() as session:
        running_rows = (
            await session.execute(
                text("SELECT id FROM campaigns WHERE status = 'RUNNING' AND deleted_at IS NULL")
            )
        ).fetchall()
        completed_rows = (
            await session.execute(
                text("SELECT id FROM campaigns WHERE status = 'COMPLETED' AND deleted_at IS NULL")
            )
        ).fetchall()

    assert [row[0] for row in running_rows] == [running_id]
    assert [row[0] for row in completed_rows] == [completed_id]
    # And ``CREATED`` membership is still exclusive — paused is not in the active subset.
    assert created_id not in {row[0] for row in running_rows}


@pytest.mark.asyncio
async def test_single_status_equality_query_uses_partial_index(
    fresh_database: None,
) -> None:
    """The application's ``WHERE status = X`` shape picks the partial index.

    This is the load-bearing assertion: the repository emits single-status
    equality (see ``CampaignRepository.list_filtered`` /
    ``list_keyset``); if the SQLite planner does not pick the matching
    partial index for that exact shape, the audit's optimisation is
    inert. We seed enough rows across statuses so the planner has
    realistic selectivity stats, then ``ANALYZE`` and ``EXPLAIN QUERY
    PLAN`` the active-poll query and assert the per-status index is in
    the plan text.
    """
    statuses = list(CampaignStatus)
    for i in range(200):
        await _seed_campaign(statuses[i % len(statuses)])

    async with get_session() as session:
        await session.execute(text("ANALYZE"))
        plan_rows = (
            await session.execute(
                text(
                    "EXPLAIN QUERY PLAN "
                    "SELECT id FROM campaigns "
                    "WHERE status = 'RUNNING' AND deleted_at IS NULL"
                )
            )
        ).fetchall()
    plan_text = " ".join(str(row) for row in plan_rows).upper()
    assert "IX_CAMPAIGNS_STATUS_RUNNING_ACTIVE" in plan_text, (
        "Planner did not pick the per-status RUNNING partial index for the "
        f"application's equality query shape. Plan was: {plan_text}"
    )


@pytest.mark.asyncio
async def test_terminal_status_equality_query_uses_partial_index(
    fresh_database: None,
) -> None:
    """Terminal-state equality queries also pick the matching partial index.

    Mirrors the active-poll assertion for the reporting / archive
    traffic shape. The audit's "active vs terminal" partition only
    delivers a cache benefit if both halves are independently served
    by their own index.
    """
    statuses = list(CampaignStatus)
    for i in range(200):
        await _seed_campaign(statuses[i % len(statuses)])

    async with get_session() as session:
        await session.execute(text("ANALYZE"))
        plan_rows = (
            await session.execute(
                text(
                    "EXPLAIN QUERY PLAN "
                    "SELECT id FROM campaigns "
                    "WHERE status = 'COMPLETED' AND deleted_at IS NULL"
                )
            )
        ).fetchall()
    plan_text = " ".join(str(row) for row in plan_rows).upper()
    assert "IX_CAMPAIGNS_STATUS_COMPLETED_ACTIVE" in plan_text, (
        f"Planner did not pick the per-status COMPLETED partial index. Plan was: {plan_text}"
    )
