"""Inline-content cap for ``bo_export_campaign``.

The MCP transport injects the export ``content`` string straight into
the agent's context, so it must be bounded: the operation truncates at
the last full CSV row within ``max_content_bytes`` and reports
``truncated`` / ``n_results_included`` plus a pagination hint. The REST
download route opts out with ``max_content_bytes=None`` and keeps the
full payload.

Reference: MCP tool results are context-bound (token-budgeted) unlike
HTTP downloads — see the transport guidance in
https://modelcontextprotocol.io/specification/2025-06-18/server/tools.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Coroutine
from typing import Any
from uuid import UUID, uuid4

import pytest

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
from bo_mcp_server.operations.export_campaign import (
    MAX_EXPORT_CONTENT_BYTES,
    MIN_EXPORT_ROW_BYTES,
    export_campaign_operation,
)
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    UserRepository,
    get_session,
)

pytestmark = pytest.mark.usefixtures("setup_database")

N_RESULTS = 20

_ListByCampaign = Callable[..., Coroutine[Any, Any, list[Result]]]


def _recording_list_by_campaign(seen_limits: list[int | None]) -> _ListByCampaign:
    """Wrap ``ResultRepository.list_by_campaign`` to record the LIMIT passed."""
    original = ResultRepository.list_by_campaign

    async def _wrapper(
        self: ResultRepository,
        campaign_id: UUID,
        *,
        limit: int | None = None,
        include_deleted: bool = False,
    ) -> list[Result]:
        seen_limits.append(limit)
        return await original(self, campaign_id, limit=limit, include_deleted=include_deleted)

    return _wrapper


async def _seed_campaign_with_results(n_results: int) -> UUID:
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
                name="export-cap",
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
        result_repo = ResultRepository(session)
        for i in range(n_results):
            await result_repo.save(
                Result(
                    campaign_id=campaign.id,
                    parameter_values={"x": i / n_results},
                    objective_values={"y": float(i)},
                    source=ResultSource.API,
                    submitted_by=owner.id,
                )
            )
    return campaign.id


@pytest.mark.asyncio
async def test_small_cap_truncates_at_row_boundary_with_hint() -> None:
    campaign_id = await _seed_campaign_with_results(N_RESULTS)

    # Cap sized to hold the header plus only a few rows.
    result = await export_campaign_operation(str(campaign_id), max_content_bytes=600)

    assert result["success"] is True
    assert result["truncated"] is True
    assert result["n_results"] == N_RESULTS
    assert 0 < result["n_results_included"] < N_RESULTS
    # Content ends on a full row: header + n_results_included data lines.
    lines = result["content"].rstrip("\r\n").splitlines()
    assert len(lines) == 1 + result["n_results_included"]
    assert any("bo_list_results" in w for w in result["warnings"])


@pytest.mark.asyncio
async def test_uncapped_export_returns_all_rows() -> None:
    campaign_id = await _seed_campaign_with_results(N_RESULTS)

    result = await export_campaign_operation(str(campaign_id), max_content_bytes=None)

    assert result["truncated"] is False
    assert result["n_results_included"] == N_RESULTS
    lines = result["content"].rstrip("\r\n").splitlines()
    assert len(lines) == 1 + N_RESULTS


@pytest.mark.asyncio
async def test_default_cap_leaves_small_campaigns_untouched() -> None:
    campaign_id = await _seed_campaign_with_results(5)

    result = await export_campaign_operation(str(campaign_id))

    assert result["truncated"] is False
    assert result["n_results_included"] == 5
    assert len(result["content"]) < MAX_EXPORT_CONTENT_BYTES


@pytest.mark.asyncio
async def test_header_larger_than_cap_returns_empty_content_within_budget() -> None:
    """The byte contract holds even when the header alone exceeds the cap.

    Rows are truncated at the budget, but the header was previously
    written unconditionally — a cap smaller than the header (or a
    pathologically wide spec) silently violated ``max_content_bytes``.
    """
    campaign_id = await _seed_campaign_with_results(5)

    # Smaller than the ~60-byte header for this spec.
    result = await export_campaign_operation(str(campaign_id), max_content_bytes=10)

    assert result["success"] is True
    assert result["content"] == ""
    assert len(result["content"]) <= 10
    assert result["n_results"] == 5
    assert result["n_results_included"] == 0
    assert result["truncated"] is True
    assert any("header alone exceeds" in w for w in result["warnings"])


@pytest.mark.asyncio
async def test_capped_export_bounds_the_database_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capped export must LIMIT the repository read, not slice after.

    Truncating only the rendered CSV would leave the DB hydration (and
    ORM/entity work) proportional to campaign size — the resource-
    consumption half of the finding. The byte budget converts to a row
    LIMIT via ``MIN_EXPORT_ROW_BYTES``.
    """
    max_content_bytes = 600
    campaign_id = await _seed_campaign_with_results(N_RESULTS)

    seen_limits: list[int | None] = []
    monkeypatch.setattr(
        ResultRepository, "list_by_campaign", _recording_list_by_campaign(seen_limits)
    )

    result = await export_campaign_operation(str(campaign_id), max_content_bytes=max_content_bytes)

    assert result["truncated"] is True
    assert seen_limits == [max_content_bytes // MIN_EXPORT_ROW_BYTES + 1]


@pytest.mark.asyncio
async def test_uncapped_export_reads_without_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    campaign_id = await _seed_campaign_with_results(5)

    seen_limits: list[int | None] = []
    monkeypatch.setattr(
        ResultRepository, "list_by_campaign", _recording_list_by_campaign(seen_limits)
    )

    result = await export_campaign_operation(str(campaign_id), max_content_bytes=None)

    assert result["n_results_included"] == 5
    assert seen_limits == [None]
