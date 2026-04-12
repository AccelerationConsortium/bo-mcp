"""Update suggestion status operation - protocol-neutral business logic."""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import SuggestionStatus
from bo_mcp_server.domain.event import Event, EventType
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.storage import EventRepository, SuggestionRepository, get_session

logger = logging.getLogger(__name__)

# Allowed manual status transitions. "completed" is set by submit_results, not directly.
ALLOWED_TARGET_STATUSES = {
    SuggestionStatus.ACCEPTED,
    SuggestionStatus.REJECTED,
    SuggestionStatus.EXPIRED,
}

VALID_SOURCE_STATUSES: dict[SuggestionStatus, set[SuggestionStatus]] = {
    SuggestionStatus.ACCEPTED: {SuggestionStatus.PENDING},
    SuggestionStatus.REJECTED: {SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED},
    SuggestionStatus.EXPIRED: {SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED},
}


async def update_suggestion_status_operation(
    suggestion_id: str,
    status: str,
) -> dict[str, Any]:
    """Update the status of a suggestion.

    Validates the transition, persists the new status, and returns the result.

    Valid transitions:
        - pending -> accepted (mark for execution)
        - pending -> rejected (skip this suggestion)
        - pending -> expired  (suggestion no longer relevant)
        - accepted -> rejected (changed mind before executing)
        - accepted -> expired  (suggestion no longer relevant)

    Args:
        suggestion_id: UUID string of the suggestion to update.
        status: New status string. One of: "accepted", "rejected", "expired".

    Returns:
        Dictionary with success, suggestion_id, status, previous_status, errors.
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
        valid = sorted(s.value for s in ALLOWED_TARGET_STATUSES)
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid status '{status}'. Must be one of: {valid}",
            details={"status": status, "valid_statuses": valid},
        )

    if target_status not in ALLOWED_TARGET_STATUSES:
        valid = sorted(s.value for s in ALLOWED_TARGET_STATUSES)
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
        valid_sources = VALID_SOURCE_STATUSES.get(target_status, set())

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

        # Record audit event for traceability
        event_repo = EventRepository(session)
        await event_repo.save(
            Event(
                campaign_id=suggestion.campaign_id,
                event_type=EventType.LIFECYCLE,
                tool_name="bo_update_suggestion_status",
                input_summary={
                    "suggestion_id": suggestion_id,
                    "target_status": target_status.value,
                },
                output_summary={
                    "previous_status": previous_status.value,
                    "new_status": target_status.value,
                },
            )
        )

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
