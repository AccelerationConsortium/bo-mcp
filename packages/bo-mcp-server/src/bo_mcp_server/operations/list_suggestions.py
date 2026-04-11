"""List suggestions operation - protocol-neutral business logic."""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import SuggestionStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel
from bo_mcp_server.storage import (
    CampaignRepository,
    SuggestionRepository,
    get_session,
)

logger = logging.getLogger(__name__)


MAX_SUGGESTIONS_LIMIT = 500


async def list_suggestions_operation(
    campaign_id: str,
    status_filter: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """List suggestions for a campaign with optional status filtering and pagination.

    Args:
        campaign_id: UUID string of the campaign.
        status_filter: Optional suggestion status string to filter by.
        limit: Maximum number of suggestions to return. None returns all (backward-compatible).
        offset: Number of suggestions to skip for pagination.
        verbosity: Response verbosity level (minimal, standard, detailed).

    Returns:
        Dictionary with success, suggestions, total_count, limit, offset, errors.
    """
    logger.info(
        "Listing suggestions: campaign_id=%s, status_filter=%s",
        campaign_id,
        status_filter,
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

    status: SuggestionStatus | None = None
    if status_filter is not None:
        try:
            status = SuggestionStatus(status_filter)
        except ValueError:
            valid_statuses = [s.value for s in SuggestionStatus]
            return make_error_response(
                ErrorCode.VALIDATION_FAILED,
                message=(
                    f"Invalid status_filter '{status_filter}'. Must be one of: {valid_statuses}"
                ),
                details={"status_filter": status_filter, "valid_statuses": valid_statuses},
            )

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            return make_error_response(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )

        suggestion_repo = SuggestionRepository(session)
        suggestions = await suggestion_repo.list_by_campaign(campaign_uuid, status=status)

    # Sort by created_at descending
    suggestions.sort(key=lambda s: s.created_at, reverse=True)
    total_count = len(suggestions)

    # Apply pagination
    offset = max(0, offset)
    if limit is not None:
        limit = max(1, min(limit, MAX_SUGGESTIONS_LIMIT))
        suggestions = suggestions[offset : offset + limit]
    else:
        suggestions = suggestions[offset:]

    suggestions_out: list[dict[str, Any]] = []
    for s in suggestions:
        if verbosity_level == VerbosityLevel.MINIMAL:
            suggestions_out.append(
                {
                    "suggestion_id": str(s.id),
                    "status": s.status.value,
                }
            )
        elif verbosity_level == VerbosityLevel.STANDARD:
            suggestions_out.append(
                {
                    "suggestion_id": str(s.id),
                    "status": s.status.value,
                    "parameter_values": s.parameter_values,
                    "iteration": s.provenance.iteration,
                    "generation_method": s.provenance.generation_method,
                    "created_at": s.created_at.isoformat(),
                }
            )
        else:
            suggestions_out.append(
                {
                    "suggestion_id": str(s.id),
                    "status": s.status.value,
                    "parameter_values": s.parameter_values,
                    "iteration": s.provenance.iteration,
                    "batch_index": s.provenance.batch_index,
                    "generation_method": s.provenance.generation_method,
                    "acquisition_function": s.provenance.acquisition_function,
                    "acquisition_value": s.provenance.acquisition_value,
                    "model_uncertainty": s.provenance.model_uncertainty,
                    "model_type": s.provenance.model_type,
                    "confidence_level": s.provenance.confidence_level,
                    "predicted_objectives": s.provenance.predicted_objectives,
                    "predicted_std": s.provenance.predicted_std,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                }
            )

    return {
        "success": True,
        "suggestions": suggestions_out,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "errors": [],
    }
