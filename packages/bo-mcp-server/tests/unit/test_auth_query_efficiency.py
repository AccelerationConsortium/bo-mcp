"""Query-efficiency contracts for the batch authorization helpers.

``ensure_owned_campaigns`` authorizes up to ``MAX_BATCH_CAMPAIGN_IDS``
ids per request and ``get_spec_for_user`` authorizes one spec per GET;
both previously scaled their SELECT count with input size (one
``repo.get`` per id, respectively a full hydration of the owner's
campaign list). These tests pin the O(1)-queries contract with a
connection-level statement counter (the pattern from
``test_storage/test_lazy_relationships_contract.py``) alongside the
unchanged authorization semantics.

Reference: OWASP API Security Top 10, API4 "Unrestricted Resource
Consumption" — request cost must not scale linearly with an
attacker-controlled input list when a set-based query suffices.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event

from bo_mcp_server.client.auth import (
    NotAuthorizedError,
    NotFoundError,
    ensure_owned_campaigns,
    get_spec_for_user,
    list_owner_campaigns_with_specs,
)
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
    get_session,
)
from bo_mcp_server.storage.database import _get_engine

pytestmark = pytest.mark.usefixtures("setup_database")


@contextmanager
def _count_select_statements() -> Iterator[list[int]]:
    """Yield a one-element list counting SELECTs on the current engine."""
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


def _make_spec(name: str) -> CampaignSpec:
    return CampaignSpec(
        name=name,
        description="query-efficiency test spec",
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


async def _seed_user() -> User:
    unique = str(uuid4())
    async with get_session() as session:
        return await UserRepository(session).save(
            User(
                name="Owner",
                email=f"owner-{unique}@example.com",
                api_key_hash=hashlib.sha256(unique.encode()).hexdigest(),
            )
        )


async def _seed_campaigns(owner_id: UUID, n_campaigns: int) -> tuple[list[UUID], list[UUID]]:
    """Seed campaigns for ``owner_id``; returns (campaign_ids, spec_ids)."""
    campaign_ids: list[UUID] = []
    spec_ids: list[UUID] = []
    async with get_session() as session:
        spec_repo = CampaignSpecRepository(session)
        campaign_repo = CampaignRepository(session)
        for i in range(n_campaigns):
            spec_id = uuid4()
            await spec_repo.save(_make_spec(f"spec-{i}"), spec_id=spec_id)
            campaign = Campaign(
                spec_id=spec_id,
                owner_id=owner_id,
                status=CampaignStatus.RUNNING,
            )
            await campaign_repo.save(campaign)
            campaign_ids.append(campaign.id)
            spec_ids.append(spec_id)
    return campaign_ids, spec_ids


class TestEnsureOwnedCampaignsBatched:
    """The batch ownership check issues one set-based SELECT, not N."""

    @pytest.mark.asyncio
    async def test_query_count_is_constant_in_batch_size(self) -> None:
        owner = await _seed_user()
        campaign_ids, _ = await _seed_campaigns(owner.id, 10)

        with _count_select_statements() as counter:
            await ensure_owned_campaigns([str(cid) for cid in campaign_ids], owner.id)

        assert counter[0] == 1, (
            f"expected one set-based SELECT for 10 campaign ids, saw {counter[0]}"
        )

    @pytest.mark.asyncio
    async def test_foreign_owned_campaign_still_raises(self) -> None:
        owner = await _seed_user()
        stranger = await _seed_user()
        campaign_ids, _ = await _seed_campaigns(owner.id, 2)

        with pytest.raises(NotAuthorizedError):
            await ensure_owned_campaigns([str(cid) for cid in campaign_ids], stranger.id)

    @pytest.mark.asyncio
    async def test_invalid_and_missing_ids_are_ignored(self) -> None:
        owner = await _seed_user()
        campaign_ids, _ = await _seed_campaigns(owner.id, 1)

        # Invalid format and unknown UUID are surfaced by the operation
        # layer, never by the authorizer.
        await ensure_owned_campaigns(["not-a-uuid", str(uuid4()), str(campaign_ids[0])], owner.id)

    @pytest.mark.asyncio
    async def test_all_invalid_ids_short_circuit_without_queries(self) -> None:
        owner = await _seed_user()
        with _count_select_statements() as counter:
            await ensure_owned_campaigns(["nope", "also-nope"], owner.id)
        assert counter[0] == 0


class TestListOwnerCampaignsWithSpecsCapped:
    """The bare ``GET /campaigns`` view must not return an unbounded array.

    Mirrors the suggestion/result list caps: ``limit`` truncates at the
    database to a stable oldest-first prefix instead of a large owner's
    full campaign history (OWASP API4, unrestricted resource consumption).
    """

    @pytest.mark.asyncio
    async def test_limit_truncates_to_oldest_first_prefix(self) -> None:
        owner = await _seed_user()
        await _seed_campaigns(owner.id, 5)

        pairs = await list_owner_campaigns_with_specs(owner.id, limit=3)

        assert [spec.name for _, spec in pairs] == ["spec-0", "spec-1", "spec-2"]

    @pytest.mark.asyncio
    async def test_no_limit_returns_everything(self) -> None:
        owner = await _seed_user()
        await _seed_campaigns(owner.id, 5)

        pairs = await list_owner_campaigns_with_specs(owner.id)

        assert len(pairs) == 5


class TestGetSpecForUserSingleQuery:
    """Spec authorization must not hydrate the owner's campaign list."""

    @pytest.mark.asyncio
    async def test_query_count_is_constant_in_owned_campaigns(self) -> None:
        owner = await _seed_user()
        _, spec_ids = await _seed_campaigns(owner.id, 10)

        with _count_select_statements() as counter:
            spec, _created_at = await get_spec_for_user(str(spec_ids[0]), owner.id)

        assert spec.name == "spec-0"
        # One SELECT for the spec row, one LIMIT-1 ownership probe —
        # independent of how many campaigns the owner has.
        assert counter[0] == 2, (
            f"expected 2 SELECTs regardless of owned-campaign count, saw {counter[0]}"
        )

    @pytest.mark.asyncio
    async def test_foreign_tenant_gets_uniform_not_found(self) -> None:
        owner = await _seed_user()
        stranger = await _seed_user()
        _, spec_ids = await _seed_campaigns(owner.id, 1)

        with pytest.raises(NotFoundError):
            await get_spec_for_user(str(spec_ids[0]), stranger.id)

    @pytest.mark.asyncio
    async def test_unknown_spec_raises_not_found(self) -> None:
        owner = await _seed_user()
        with pytest.raises(NotFoundError):
            await get_spec_for_user(str(uuid4()), owner.id)
