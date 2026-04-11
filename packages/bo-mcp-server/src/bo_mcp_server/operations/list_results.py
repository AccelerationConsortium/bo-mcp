"""List results operation - protocol-neutral business logic."""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel
from bo_mcp_server.storage import (
    CampaignRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)

MAX_RESULTS_LIMIT = 500


async def list_results_operation(
    campaign_id: str,
    limit: int = 50,
    offset: int = 0,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """List experimental results for a campaign with pagination.

    Args:
        campaign_id: UUID string of the campaign.
        limit: Maximum number of results to return (capped at MAX_RESULTS_LIMIT).
        offset: Number of results to skip for pagination.
        verbosity: Response verbosity level (minimal, standard, detailed).

    Returns:
        Dictionary with success, results, total_count, limit, offset, errors.
    """
    logger.info(
        "Listing results: campaign_id=%s, limit=%d, offset=%d",
        campaign_id,
        limit,
        offset,
    )

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
        all_results = await result_repo.list_by_campaign(campaign_uuid)

    # Sort by created_at descending (most recent first)
    all_results.sort(key=lambda r: r.created_at, reverse=True)
    total_count = len(all_results)
    page = all_results[offset : offset + limit]

    results_out: list[dict[str, Any]] = []
    for r in page:
        if verbosity_level == VerbosityLevel.MINIMAL:
            results_out.append(
                {
                    "result_id": str(r.id),
                    "objective_values": r.objective_values,
                }
            )
        elif verbosity_level == VerbosityLevel.STANDARD:
            results_out.append(
                {
                    "result_id": str(r.id),
                    "parameter_values": r.parameter_values,
                    "objective_values": r.objective_values,
                    "suggestion_id": str(r.suggestion_id) if r.suggestion_id else None,
                    "created_at": r.created_at.isoformat(),
                }
            )
        else:
            results_out.append(
                {
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
            )

    return {
        "success": True,
        "results": results_out,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "errors": [],
    }
