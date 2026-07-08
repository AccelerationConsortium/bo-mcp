"""Bounded rendering for the ``suggestions://{campaign_id}`` resource.

Resources have no parameters, so unlike the ``bo_list_suggestions``
tool they cannot paginate — a long campaign's pending pool previously
rendered as an unbounded Markdown blob. The resource now caps the
listing at :data:`MAX_PENDING_SUGGESTIONS_RENDERED` (oldest first) and
appends a trailer pointing at the paginated tool, mirroring the fixed
``limit`` the events resource already applies.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from bo_mcp_server.domain import (
    CampaignSpec,
    InputParameter,
    Objective,
    ParameterType,
    Suggestion,
    SuggestionProvenance,
    SuggestionStatus,
    User,
)
from bo_mcp_server.domain.campaign import Campaign, CampaignStatus
from bo_mcp_server.resources.suggestion_resource import (
    MAX_PENDING_SUGGESTIONS_RENDERED,
    get_suggestions,
)
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    SuggestionRepository,
    UserRepository,
    get_session,
)

pytestmark = pytest.mark.usefixtures("setup_database")

OVERFLOW = 5


async def _seed_campaign_with_pending(n_pending: int) -> UUID:
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    async with get_session() as session:
        unique = str(uuid4())
        owner = await UserRepository(session).save(
            User(
                name="Owner",
                email=f"owner-{unique}@example.com",
                api_key_hash=hashlib.sha256(unique.encode()).hexdigest(),
            )
        )
        spec_id = uuid4()
        await CampaignSpecRepository(session).save(
            CampaignSpec(
                name="resource-cap",
                description="",
                parameters=(
                    InputParameter(
                        name="x",
                        type=ParameterType.CONTINUOUS,
                        bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                    ),
                ),
                objectives=(Objective(name="y", direction="maximize"),),
                batch_size=1,
            ),
            spec_id=spec_id,
        )
        campaign = Campaign(spec_id=spec_id, owner_id=owner.id, status=CampaignStatus.RUNNING)
        await CampaignRepository(session).save(campaign)
        suggestion_repo = SuggestionRepository(session)
        for i in range(n_pending):
            await suggestion_repo.save(
                Suggestion(
                    campaign_id=campaign.id,
                    parameter_values={"x": i / max(n_pending, 1)},
                    provenance=SuggestionProvenance(iteration=1, batch_index=i),
                    created_at=base_time + timedelta(seconds=i),
                )
            )
    return campaign.id


@pytest.mark.asyncio
async def test_overflowing_pool_renders_cap_blocks_plus_trailer() -> None:
    campaign_id = await _seed_campaign_with_pending(MAX_PENDING_SUGGESTIONS_RENDERED + OVERFLOW)

    body = await get_suggestions(str(campaign_id))

    assert body.count("## Suggestion ") == MAX_PENDING_SUGGESTIONS_RENDERED
    assert f"{OVERFLOW} more pending suggestion(s) not shown" in body
    assert "bo_list_suggestions" in body


@pytest.mark.asyncio
async def test_small_pool_renders_fully_without_trailer() -> None:
    campaign_id = await _seed_campaign_with_pending(3)

    body = await get_suggestions(str(campaign_id))

    assert body.count("## Suggestion ") == 3
    assert "more pending suggestion(s) not shown" not in body


@pytest.mark.asyncio
async def test_resource_limits_the_database_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap must be a query-level LIMIT, not a Python slice.

    Slicing after a full ``list_by_campaign`` would keep the Markdown
    bounded while still hydrating (and sorting) every pending row — the
    resource-consumption half of the finding.
    """
    campaign_id = await _seed_campaign_with_pending(MAX_PENDING_SUGGESTIONS_RENDERED + OVERFLOW)

    seen_limits: list[int | None] = []
    original = SuggestionRepository.list_by_campaign

    async def _recording_list(
        self: SuggestionRepository,
        campaign_id: UUID,
        status: SuggestionStatus | None = None,
        *,
        limit: int | None = None,
        include_deleted: bool = False,
    ) -> list[Suggestion]:
        seen_limits.append(limit)
        return await original(
            self, campaign_id, status, limit=limit, include_deleted=include_deleted
        )

    monkeypatch.setattr(SuggestionRepository, "list_by_campaign", _recording_list)

    body = await get_suggestions(str(campaign_id))

    assert body.count("## Suggestion ") == MAX_PENDING_SUGGESTIONS_RENDERED
    assert seen_limits == [MAX_PENDING_SUGGESTIONS_RENDERED]
