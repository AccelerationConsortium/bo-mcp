"""List campaigns operation - protocol-neutral business logic."""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.backend_context import with_campaign_backend_scope
from bo_mcp_server.domain import Campaign, CampaignSpec, CampaignStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.pagination import build_page_cursor, parse_optional_cursor
from bo_mcp_server.response_formatter import VerbosityLevel, with_response_metadata
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)

MAX_LIMIT = 100


def _build_minimal_summary(campaign: Campaign, name: str) -> dict[str, Any]:
    return {
        "campaign_id": str(campaign.id),
        "name": name,
        "status": campaign.status.value,
    }


def _build_standard_summary(
    campaign: Campaign, name: str, n_results: int, backend: str | None
) -> dict[str, Any]:
    return {
        "campaign_id": str(campaign.id),
        "name": name,
        "status": campaign.status.value,
        "backend": backend,
        "iteration": campaign.iteration,
        "n_results": n_results,
        "created_at": campaign.created_at.isoformat(),
    }


def _build_summary(
    campaign: Campaign,
    name: str,
    n_results: int,
    spec: CampaignSpec | None,
    verbosity_level: VerbosityLevel,
) -> dict[str, Any]:
    if verbosity_level == VerbosityLevel.MINIMAL:
        return _build_minimal_summary(campaign, name)
    if verbosity_level == VerbosityLevel.STANDARD:
        return _build_standard_summary(campaign, name, n_results, spec.backend if spec else None)
    return _build_detailed_summary(campaign, name, n_results, spec)


def _build_detailed_summary(
    campaign: Campaign,
    name: str,
    n_results: int,
    spec: CampaignSpec | None,
) -> dict[str, Any]:
    spec_summary = None
    if spec:
        spec_summary = {
            "n_parameters": len(spec.parameters),
            "n_objectives": len(spec.objectives),
            "n_constraints": len(spec.constraints) if spec.constraints else 0,
            "batch_size": spec.batch_size,
            "parameter_names": [p.name for p in spec.parameters],
            "objective_names": [o.name for o in spec.objectives],
        }
    return {
        "campaign_id": str(campaign.id),
        "name": name,
        "status": campaign.status.value,
        "backend": spec.backend if spec else None,
        "iteration": campaign.iteration,
        "n_results": n_results,
        "created_at": campaign.created_at.isoformat(),
        "owner_id": str(campaign.owner_id),
        "spec_id": str(campaign.spec_id),
        "spec_summary": spec_summary,
        "has_hypervolume_history": len(campaign.hypervolume_history) > 0,
        "has_backend_state": campaign.backend_state is not None,
    }


def _validate_list_campaigns_inputs(
    status: str | None,
    verbosity: str,
    cursor: str | None,
    offset: int,
) -> dict[str, Any] | tuple[VerbosityLevel, CampaignStatus | None, Any, Any]:
    """Validate filter and pagination inputs.

    Splits the multi-step validation out of the main operation so
    cognitive complexity stays manageable. Returns an error response
    dict on failure, or the parsed tuple on success.

    ``cursor`` and ``offset`` are mutually exclusive: keyset-based
    cursor pagination is stable under concurrent inserts (no
    duplicates, no skips) whereas an offset window silently shifts
    when new rows land between page reads. Accepting both at once is
    almost always a caller bug — usually a half-migrated polling
    loop — so the operation surfaces a structured validation error
    instead of silently honouring one and ignoring the other.
    """
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    status_filter: CampaignStatus | None = None
    if status is not None:
        try:
            status_filter = CampaignStatus(status)
        except ValueError:
            valid_statuses = [s.value for s in CampaignStatus]
            return make_error_response(
                ErrorCode.VALIDATION_FAILED,
                message=f"Invalid status '{status}'. Must be one of: {valid_statuses}",
                details={"status": status, "valid_statuses": valid_statuses},
            )

    if cursor is not None and offset > 0:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(
                "cursor and offset are mutually exclusive. Use cursor for stable "
                "pagination under concurrent inserts; offset is deprecated and "
                "preserved only for callers that have not migrated yet."
            ),
            details={"cursor": cursor, "offset": offset},
        )

    cursor_parsed = parse_optional_cursor(cursor)
    if isinstance(cursor_parsed, str):
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid cursor: {cursor_parsed}",
            details={"cursor": cursor},
        )

    return verbosity_level, status_filter, cursor_parsed[0], cursor_parsed[1]


@with_campaign_backend_scope
@with_response_metadata
async def list_campaigns_operation(
    owner_id: UUID | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
    verbosity: str = "standard",
    cursor: str | None = None,
) -> dict[str, Any]:
    """List optimization campaigns with optional filtering and pagination.

    Args:
        owner_id: Optional owner UUID to filter by.
        status: Optional campaign status string to filter by.
        limit: Maximum number of campaigns to return (capped at MAX_LIMIT).
        offset: **Deprecated** — kept for backward compatibility. Use
            ``cursor`` for stable pagination under concurrent inserts.
        verbosity: Response verbosity level (minimal, standard, detailed).
        cursor: Opaque cursor from a previous response's ``next_cursor``
            field. When supplied, ``offset`` is ignored and pagination
            walks the keyset on ``(created_at, id)``.

    Returns:
        Dictionary with success, campaigns, total_count, limit, offset,
        next_cursor, errors. ``next_cursor`` is ``null`` when there are
        no more pages.
    """
    logger.info(
        "Listing campaigns: owner_id=%s, status=%s, limit=%d, offset=%d, cursor=%s, verbosity=%s",
        owner_id,
        status,
        limit,
        offset,
        "set" if cursor else None,
        verbosity,
    )

    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)

    validated = _validate_list_campaigns_inputs(status, verbosity, cursor, offset)
    if isinstance(validated, dict):
        return validated
    verbosity_level, status_filter, cursor_created_at, cursor_id = validated

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)

        if cursor is not None:
            # Over-fetch by 1 so we can distinguish "exactly limit rows
            # left" from "more rows remain". Otherwise an exact final
            # page emits a misleading ``next_cursor`` that points past
            # the last row.
            campaigns, total_count = await campaign_repo.list_keyset(
                owner_id=owner_id,
                status=status_filter,
                cursor_created_at=cursor_created_at,
                cursor_id=cursor_id,
                limit=limit + 1,
            )
        else:
            campaigns, total_count = await campaign_repo.list_filtered(
                owner_id=owner_id,
                status=status_filter,
                limit=limit + 1,
                offset=offset,
            )

        has_more_page = len(campaigns) > limit
        if has_more_page:
            campaigns = campaigns[:limit]

        # Batch-fetch specs for all campaigns in one query
        spec_ids = list({c.spec_id for c in campaigns})
        specs = await spec_repo.get_by_ids(spec_ids)

        # Batch-fetch result counts if needed
        result_counts: dict[UUID, int] = {}
        if verbosity_level != VerbosityLevel.MINIMAL:
            campaign_ids = [c.id for c in campaigns]
            result_counts = await result_repo.count_by_campaigns(campaign_ids)

        # Build response
        campaign_summaries: list[dict[str, Any]] = []
        for campaign in campaigns:
            spec = specs.get(campaign.spec_id)
            name = spec.name if spec else "Unknown"
            n_results = result_counts.get(campaign.id, 0)
            campaign_summaries.append(
                _build_summary(campaign, name, n_results, spec, verbosity_level)
            )

    logger.info(
        "Listed %d campaigns (total matching: %d)",
        len(campaign_summaries),
        total_count,
    )

    next_cursor = None
    if has_more_page:
        next_cursor = build_page_cursor(campaigns, "created_at", "id")

    return {
        "success": True,
        "campaigns": campaign_summaries,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "next_cursor": next_cursor,
        "errors": [],
    }
