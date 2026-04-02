"""Update suggestion status tool for MCP."""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import SuggestionStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import SuggestionRepository, get_session

logger = logging.getLogger(__name__)

# Allowed manual status transitions. "completed" is set by submit_results, not directly.
_ALLOWED_TARGET_STATUSES = {
    SuggestionStatus.ACCEPTED,
    SuggestionStatus.REJECTED,
    SuggestionStatus.EXPIRED,
}

_VALID_SOURCE_STATUSES: dict[SuggestionStatus, set[SuggestionStatus]] = {
    SuggestionStatus.ACCEPTED: {SuggestionStatus.PENDING},
    SuggestionStatus.REJECTED: {SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED},
    SuggestionStatus.EXPIRED: {SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED},
}


@mcp.tool(name="bo_update_suggestion_status")
async def update_suggestion_status(
    suggestion_id: str,
    status: str,
) -> dict[str, Any]:
    """Update the status of a suggestion.

    Workflow: Call after reviewing suggestions from bo_list_suggestions to
    accept, reject, or expire them.

    Use this to accept, reject, or expire a suggestion. The "completed"
    status is set automatically when results are submitted via bo_submit_results.

    Valid transitions:
        - pending -> accepted (mark for execution)
        - pending -> rejected (skip this suggestion)
        - pending -> expired  (suggestion no longer relevant)
        - accepted -> rejected (changed mind before executing)
        - accepted -> expired  (suggestion no longer relevant)

    Args:
        suggestion_id: UUID of the suggestion to update.
        status: New status. One of: "accepted", "rejected", "expired".

    Returns:
        Dictionary with:
            - success: Boolean
            - suggestion_id: UUID of the updated suggestion
            - status: New status
            - previous_status: Status before the update
            - errors: List of error messages
    """
    logger.info("Updating suggestion status: suggestion_id=%s, status=%s", suggestion_id, status)

    try:
        suggestion_uuid = UUID(suggestion_id)
    except ValueError:
        return make_error_response(
            ErrorCode.SUGGESTION_NOT_FOUND,
            message="Invalid suggestion_id format",
            details={"suggestion_id": suggestion_id},
        )

    try:
        target_status = SuggestionStatus(status)
    except ValueError:
        valid = sorted(s.value for s in _ALLOWED_TARGET_STATUSES)
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid status '{status}'. Must be one of: {valid}",
            details={"status": status, "valid_statuses": valid},
        )

    if target_status not in _ALLOWED_TARGET_STATUSES:
        valid = sorted(s.value for s in _ALLOWED_TARGET_STATUSES)
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(
                f"Cannot manually set status to '{status}'. "
                f"Allowed statuses: {valid}. "
                "'completed' is set automatically by bo_submit_results."
            ),
        )

    async with get_session() as session:
        suggestion_repo = SuggestionRepository(session)
        suggestion = await suggestion_repo.get(suggestion_uuid)

        if suggestion is None:
            return make_error_response(
                ErrorCode.SUGGESTION_NOT_FOUND,
                message=f"Suggestion {suggestion_id} not found",
                details={"suggestion_id": suggestion_id},
            )

        previous_status = suggestion.status
        valid_sources = _VALID_SOURCE_STATUSES.get(target_status, set())

        if previous_status not in valid_sources:
            return make_error_response(
                ErrorCode.INVALID_STATE_TRANSITION,
                message=(
                    f"Cannot transition from '{previous_status.value}' to '{target_status.value}'. "
                    f"Valid source statuses: {sorted(s.value for s in valid_sources)}"
                ),
                details={
                    "current_status": previous_status.value,
                    "target_status": target_status.value,
                    "valid_source_statuses": sorted(s.value for s in valid_sources),
                },
            )

        updated = suggestion.with_status(target_status)
        await suggestion_repo.save(updated)

    logger.info(
        "Suggestion %s status updated: %s -> %s",
        suggestion_id,
        previous_status.value,
        target_status.value,
    )

    return {
        "success": True,
        "suggestion_id": suggestion_id,
        "status": target_status.value,
        "previous_status": previous_status.value,
        "errors": [],
    }
