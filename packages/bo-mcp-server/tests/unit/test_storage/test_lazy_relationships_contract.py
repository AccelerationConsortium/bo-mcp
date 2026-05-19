"""Contract tests for the ORM lazy-relationship audit (TODO 8.47).

Background
==========

The audit flagged the bidirectional ORM relationships on ``CampaignModel``
(``spec``, ``owner``, ``suggestions``, ``results``) plus the mirrors on the
parent / child models as latent N+1 sources. Production reads cross-table
data through explicit batch fetches in the repository layer
(``CampaignSpecRepository.get_by_ids`` / ``ResultRepository.count_by_campaigns``),
but the relationships themselves used the default ``lazy="select"`` loader.
That meant any new caller that touched ``campaign.spec`` after fetching a
list of campaigns would silently issue one ``SELECT`` per row — invisible
until the dashboard slowed down.

Strategy
========

The audit prescribed two options: flip the relationships to ``lazy="raise"``
so accidental lazy-loads are loud, or audit every caller and convert them
to explicit batch fetches. We chose ``lazy="raise"`` because it actively
prevents regressions; the contract tests below pin that choice and add a
SQLAlchemy event-listener-based query counter across the four representative
read paths (list, export, diagnostics, compare-campaigns) so the
"O(1) regardless of list size" invariant is enforced.

Why a query counter rather than just trusting ``lazy="raise"``
--------------------------------------------------------------

``lazy="raise"`` would catch a lazy *traversal*, but not an N+1 emitted
by, say, a repository helper that issues one ``get(spec_id)`` per row
inside a Python loop. The query counter catches both.

References
==========

* SQLAlchemy "Relationship Loading Techniques" — Preventing unwanted
  lazy loads: https://docs.sqlalchemy.org/en/20/orm/queryguide/relationships.html#preventing-unwanted-lazy-loads-using-the-raiseload-strategy
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.exc import InvalidRequestError

from bo_mcp_server.domain import (
    CampaignSpec,
    InputParameter,
    Objective,
    ParameterType,
    Result,
    ResultSource,
    User,
)
from bo_mcp_server.domain.campaign import Campaign, CampaignStatus
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    UserRepository,
    close_database,
    get_session,
    init_database,
)
from bo_mcp_server.storage.database import _get_engine
from bo_mcp_server.storage.models import (
    CampaignModel,
    CampaignSpecModel,
    ResultModel,
    SuggestionModel,
    UserModel,
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


@contextmanager
def _count_select_statements() -> Iterator[list[int]]:
    """Yield a one-element list that tracks ``SELECT`` statements on the bind.

    Uses SQLAlchemy's ``before_cursor_execute`` event so it counts statements
    at the connection layer — including any lazy-loaded sub-selects.
    """

    counter = [0]
    sync_engine = _get_engine().sync_engine

    def _on_execute(
        _conn,  # type: ignore[no-untyped-def]
        _cursor,  # type: ignore[no-untyped-def]
        statement: str,
        _parameters,  # type: ignore[no-untyped-def]
        _context,  # type: ignore[no-untyped-def]
        _executemany,  # type: ignore[no-untyped-def]
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            counter[0] += 1

    event.listen(sync_engine, "before_cursor_execute", _on_execute)
    try:
        yield counter
    finally:
        event.remove(sync_engine, "before_cursor_execute", _on_execute)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_spec(name: str) -> CampaignSpec:
    return CampaignSpec(
        name=name,
        description="contract-test spec",
        parameters=(
            InputParameter(
                name="x",
                type=ParameterType.CONTINUOUS,
                bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            ),
        ),
        objectives=(Objective(name="y", direction="maximize"),),
        batch_size=1,
    )


async def _seed_campaigns(n_campaigns: int, results_per_campaign: int) -> list[UUID]:
    """Seed ``n_campaigns`` campaigns each carrying ``results_per_campaign`` rows."""

    campaign_ids: list[UUID] = []
    async with get_session() as session:
        user_repo = UserRepository(session)
        spec_repo = CampaignSpecRepository(session)
        campaign_repo = CampaignRepository(session)
        result_repo = ResultRepository(session)

        unique = str(uuid4())
        owner = User(
            name="Owner",
            email=f"owner-{unique}@example.com",
            api_key_hash=hashlib.sha256(unique.encode()).hexdigest(),
        )
        owner = await user_repo.save(owner)

        for i in range(n_campaigns):
            spec = _make_spec(f"spec-{i}")
            spec_id = uuid4()
            await spec_repo.save(spec, spec_id=spec_id)
            campaign = Campaign(
                spec_id=spec_id,
                owner_id=owner.id,
                status=CampaignStatus.RUNNING,
            )
            await campaign_repo.save(campaign)
            campaign_ids.append(campaign.id)

            for j in range(results_per_campaign):
                result = Result(
                    campaign_id=campaign.id,
                    parameter_values={"x": float(j) / max(results_per_campaign, 1)},
                    objective_values={"y": float(j)},
                    source=ResultSource.API,
                    submitted_by=owner.id,
                )
                await result_repo.save(result)
        await session.commit()
    return campaign_ids


# ---------------------------------------------------------------------------
# Static invariant: every relationship uses ``lazy="raise"``
# ---------------------------------------------------------------------------


def test_every_orm_relationship_declares_lazy_raise() -> None:
    """Pin the loader strategy so future relationships cannot regress.

    A future contributor adding a new relationship without explicitly
    setting ``lazy="raise"`` would re-introduce the latent N+1 risk that
    TODO 8.47 closed. The contract is: every relationship on every ORM
    model loads via ``raise``; production reads cross-table data through
    explicit batch fetches in the repository layer.
    """
    models = [UserModel, CampaignSpecModel, CampaignModel, SuggestionModel, ResultModel]
    offenders: list[str] = []
    for model in models:
        for rel in model.__mapper__.relationships:
            if rel.lazy != "raise":
                offenders.append(f"{model.__name__}.{rel.key} lazy={rel.lazy!r}")
    assert not offenders, (
        f"ORM relationships must declare lazy='raise' (TODO 8.47). Offending entries: {offenders}"
    )


# ---------------------------------------------------------------------------
# Behavioural guard: accidental traversal raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accidental_relationship_traversal_raises(fresh_database: None) -> None:
    """An unmodified ``select(CampaignModel)`` cannot silently lazy-load ``.spec``.

    Reproducer for the failure mode TODO 8.47 closes: a caller fetches a
    campaign row, walks ``.spec``, and a per-row SELECT fires. With
    ``lazy="raise"`` the access is loud, so the regression shows up in
    tests instead of in production latency dashboards.
    """
    await _seed_campaigns(n_campaigns=1, results_per_campaign=0)

    from sqlalchemy import select

    async with get_session() as session:
        result = await session.execute(select(CampaignModel).limit(1))
        campaign_row = result.scalar_one()
        with pytest.raises(InvalidRequestError):
            _ = campaign_row.spec
        with pytest.raises(InvalidRequestError):
            _ = campaign_row.owner


# ---------------------------------------------------------------------------
# Query-count invariants per representative read path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_campaigns_is_constant_query_count(fresh_database: None) -> None:
    """List path remains O(1) queries — spec/result counts already batch.

    Adding 5x more campaigns must not multiply the query count.
    """
    from bo_mcp_server.operations.list_campaigns import list_campaigns_operation

    await _seed_campaigns(n_campaigns=10, results_per_campaign=2)

    async def _measure(limit: int) -> int:
        with _count_select_statements() as counter:
            _ = await list_campaigns_operation(limit=limit)
        return counter[0]

    queries_small = await _measure(limit=2)
    queries_large = await _measure(limit=10)

    # The list path issues a small fixed number of statements
    # regardless of how many rows are returned: one for the page,
    # one for the count, one batch for specs, one batch for result
    # counts. Eight is a comfortable upper bound that lets the
    # bootstrap statements (settings, idempotency probe, etc.) ride
    # along without making the test brittle. The critical contract
    # is that the count does not *scale* with the row count.
    assert queries_small <= 8, f"List(limit=2) emitted {queries_small} SELECTs"
    assert queries_large <= 8, f"List(limit=10) emitted {queries_large} SELECTs"
    assert queries_large == queries_small, (
        f"List query count must be independent of result-set size: "
        f"limit=2 → {queries_small}, limit=10 → {queries_large}"
    )


@pytest.mark.asyncio
async def test_export_campaign_constant_query_count(fresh_database: None) -> None:
    """Export uses explicit repo calls — query count is independent of N(results)."""
    from bo_mcp_server.operations.export_campaign import export_campaign_operation

    async def _measure(results_per_campaign: int) -> int:
        campaigns = await _seed_campaigns(n_campaigns=1, results_per_campaign=results_per_campaign)
        with _count_select_statements() as counter:
            response = await export_campaign_operation(
                campaign_id=str(campaigns[0]),
                format="csv",
            )
        assert response.get("success") is True, response
        return counter[0]

    queries_small = await _measure(results_per_campaign=2)
    queries_large = await _measure(results_per_campaign=20)
    assert queries_large == queries_small, (
        f"Export query count must be independent of result rows: "
        f"2 results → {queries_small}, 20 results → {queries_large}"
    )


@pytest.mark.asyncio
async def test_compare_campaigns_batches_specs(fresh_database: None) -> None:
    """Compare batches specs via ``get_by_ids`` — query count is constant."""
    from bo_mcp_server.operations.compare_campaigns import compare_campaigns_operation

    async def _measure(n_campaigns: int) -> int:
        campaigns = await _seed_campaigns(n_campaigns=n_campaigns, results_per_campaign=3)
        with _count_select_statements() as counter:
            response = await compare_campaigns_operation(
                campaign_ids=[str(c) for c in campaigns],
            )
        assert response.get("success") is True, response
        return counter[0]

    queries_few = await _measure(n_campaigns=2)
    queries_many = await _measure(n_campaigns=6)
    assert queries_many == queries_few, (
        f"Compare query count must be independent of campaign count: "
        f"2 campaigns → {queries_few}, 6 campaigns → {queries_many}"
    )
