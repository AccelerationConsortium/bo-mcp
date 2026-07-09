"""List suggestions operation - protocol-neutral business logic."""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import Suggestion, SuggestionStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.pagination import build_page_cursor, parse_optional_cursor
from bo_mcp_server.response_formatter import VerbosityLevel, with_response_metadata
from bo_mcp_server.storage import (
    CampaignRepository,
    SuggestionRepository,
    get_session,
)

logger = logging.getLogger(__name__)


MAX_SUGGESTIONS_LIMIT = 500


def _serialize_suggestion(s: Suggestion, verbosity_level: VerbosityLevel) -> dict[str, Any]:
    """Project a Suggestion into the verbosity-specific response shape."""
    if verbosity_level == VerbosityLevel.MINIMAL:
        return {
            "suggestion_id": str(s.id),
            "status": s.status.value,
        }
    if verbosity_level == VerbosityLevel.STANDARD:
        return {
            "suggestion_id": str(s.id),
            "status": s.status.value,
            "parameter_values": s.parameter_values,
            "iteration": s.provenance.iteration,
            "generation_method": s.provenance.generation_method,
            "created_at": s.created_at.isoformat(),
        }
    return {
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


def _validate_list_suggestions_inputs(
    campaign_id: str,
    status_filter: str | None,
    verbosity: str,
    cursor: str | None,
) -> dict[str, Any] | tuple[VerbosityLevel, UUID, SuggestionStatus | None, Any, Any]:
    """Validate verbosity / status / campaign id / cursor. Returns parsed tuple or error dict."""
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

    cursor_parsed = parse_optional_cursor(cursor)
    if isinstance(cursor_parsed, str):
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid cursor: {cursor_parsed}",
            details={"cursor": cursor},
        )

    return verbosity_level, campaign_uuid, status, cursor_parsed[0], cursor_parsed[1]


@with_response_metadata
async def list_suggestions_operation(
    campaign_id: str,
    status_filter: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    verbosity: str = "standard",
    cursor: str | None = None,
) -> dict[str, Any]:
    """List suggestions for a campaign with optional status filtering and pagination.

    Args:
        campaign_id: UUID string of the campaign.
        status_filter: Optional suggestion status string to filter by.
        limit: Maximum number of suggestions to return. None returns all (backward-compatible).
        offset: **Deprecated** — kept for backward compatibility. Use
            ``cursor`` for stable pagination under concurrent inserts.
        verbosity: Response verbosity level (minimal, standard, detailed).
        cursor: Opaque cursor from a previous response's ``next_cursor``.

    Returns:
        Dictionary with success, suggestions, total_count, limit, offset,
        next_cursor, errors.
    """
    logger.info(
        "Listing suggestions: campaign_id=%s, status_filter=%s",
        campaign_id,
        status_filter,
    )

    validated = _validate_list_suggestions_inputs(campaign_id, status_filter, verbosity, cursor)
    if isinstance(validated, dict):
        return validated
    verbosity_level, campaign_uuid, status, cursor_created_at, cursor_id = validated

    offset = max(0, offset)
    if limit is not None:
        effective_limit = max(1, min(limit, MAX_SUGGESTIONS_LIMIT))
    else:
        effective_limit = MAX_SUGGESTIONS_LIMIT

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
        # Over-fetch by 1 so the final exact-multiple-of-limit page does
        # not emit a misleading ``next_cursor``.
        if cursor is not None:
            suggestions, total_count = await suggestion_repo.list_by_campaign_keyset(
                campaign_uuid,
                status=status,
                cursor_created_at=cursor_created_at,
                cursor_id=cursor_id,
                limit=effective_limit + 1,
            )
        else:
            suggestions, total_count = await suggestion_repo.list_by_campaign_paginated(
                campaign_uuid,
                status=status,
                limit=effective_limit + 1,
                offset=offset,
            )

    limit = effective_limit
    has_more_page = len(suggestions) > limit
    if has_more_page:
        suggestions = suggestions[:limit]

    suggestions_out = [_serialize_suggestion(s, verbosity_level) for s in suggestions]

    next_cursor = None
    if has_more_page:
        next_cursor = build_page_cursor(suggestions, "created_at", "id")

    return {
        "success": True,
        "suggestions": suggestions_out,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "next_cursor": next_cursor,
        "errors": [],
    }
