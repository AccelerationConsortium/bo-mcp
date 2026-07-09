"""List results operation - protocol-neutral business logic."""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import Result
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.pagination import build_page_cursor, parse_optional_cursor
from bo_mcp_server.response_formatter import VerbosityLevel, with_response_metadata
from bo_mcp_server.storage import (
    CampaignRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)

MAX_RESULTS_LIMIT = 500


def _serialize_result(r: Result, verbosity_level: VerbosityLevel) -> dict[str, Any]:
    """Project a Result into the verbosity-specific response shape."""
    if verbosity_level == VerbosityLevel.MINIMAL:
        return {
            "result_id": str(r.id),
            "objective_values": r.objective_values,
        }
    if verbosity_level == VerbosityLevel.STANDARD:
        return {
            "result_id": str(r.id),
            "parameter_values": r.parameter_values,
            "objective_values": r.objective_values,
            "suggestion_id": str(r.suggestion_id) if r.suggestion_id else None,
            "created_at": r.created_at.isoformat(),
        }
    return {
        "result_id": str(r.id),
        "parameter_values": r.parameter_values,
        "objective_values": r.objective_values,
        "suggestion_id": str(r.suggestion_id) if r.suggestion_id else None,
        "source": r.source.value,
        "submitted_by": str(r.submitted_by),
        "measurement_uncertainty": r.measurement_uncertainty,
        "metadata": r.metadata,
        "created_at": r.created_at.isoformat(),
    }


def _validate_list_results_inputs(
    campaign_id: str,
    verbosity: str,
    cursor: str | None,
) -> dict[str, Any] | tuple[VerbosityLevel, UUID, Any, Any]:
    """Validate verbosity / campaign id / cursor. Returns parsed tuple or error dict."""
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        return make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    cursor_parsed = parse_optional_cursor(cursor)
    if isinstance(cursor_parsed, str):
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid cursor: {cursor_parsed}",
            details={"cursor": cursor},
        )

    return verbosity_level, campaign_uuid, cursor_parsed[0], cursor_parsed[1]


@with_response_metadata
async def list_results_operation(
    campaign_id: str,
    limit: int = 50,
    offset: int = 0,
    verbosity: str = "standard",
    cursor: str | None = None,
) -> dict[str, Any]:
    """List experimental results for a campaign with pagination.

    Args:
        campaign_id: UUID string of the campaign.
        limit: Maximum number of results to return (capped at MAX_RESULTS_LIMIT).
        offset: **Deprecated** — kept for backward compatibility. Use
            ``cursor`` for stable pagination under concurrent inserts.
        verbosity: Response verbosity level (minimal, standard, detailed).
        cursor: Opaque cursor from a previous response's ``next_cursor``.

    Returns:
        Dictionary with success, results, total_count, limit, offset,
        next_cursor, errors.
    """
    logger.info(
        "Listing results: campaign_id=%s, limit=%d, offset=%d, cursor=%s",
        campaign_id,
        limit,
        offset,
        "set" if cursor else None,
    )

    validated = _validate_list_results_inputs(campaign_id, verbosity, cursor)
    if isinstance(validated, dict):
        return validated
    verbosity_level, campaign_uuid, cursor_created_at, cursor_id = validated

    limit = max(1, min(limit, MAX_RESULTS_LIMIT))
    offset = max(0, offset)

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            return make_error_response(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )

        result_repo = ResultRepository(session)
        # Over-fetch by 1 so we can detect "more pages remain" without
        # ambiguity when the total happens to be a multiple of ``limit``.
        if cursor is not None:
            page, total_count = await result_repo.list_by_campaign_keyset(
                campaign_uuid,
                cursor_created_at=cursor_created_at,
                cursor_id=cursor_id,
                limit=limit + 1,
            )
        else:
            page, total_count = await result_repo.list_by_campaign_paginated(
                campaign_uuid, limit=limit + 1, offset=offset
            )

    has_more_page = len(page) > limit
    if has_more_page:
        page = page[:limit]

    results_out = [_serialize_result(r, verbosity_level) for r in page]

    next_cursor = None
    if has_more_page:
        next_cursor = build_page_cursor(page, "created_at", "id")

    return {
        "success": True,
        "results": results_out,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "next_cursor": next_cursor,
        "errors": [],
    }
