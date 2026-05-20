"""Update suggestion status operation - protocol-neutral business logic."""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.domain import SuggestionStatus
from bo_mcp_server.domain.event import Event, EventType
from bo_mcp_server.errors import (
    ErrorCode,
    make_concurrent_modification_response,
    make_error_response,
)
from bo_mcp_server.idempotency import session_scope
from bo_mcp_server.response_formatter import with_response_metadata
from bo_mcp_server.storage import (
    ConcurrentModificationError,
    EventRepository,
    SuggestionRepository,
)

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


def _build_status_preview(
    suggestion_id: str,
    previous_status: SuggestionStatus,
    target_status: SuggestionStatus,
) -> dict[str, Any]:
    """Compose the dry-run preview for a suggestion-status update."""
    return {
        "success": True,
        "dry_run": True,
        "suggestion_id": suggestion_id,
        "status": previous_status.value,
        "previous_status": previous_status.value,
        "preview": {
            "from_status": previous_status.value,
            "to_status": target_status.value,
        },
        "errors": [],
    }


def _validate_update_request(
    suggestion_id: str,
    status: str,
) -> tuple[UUID, SuggestionStatus] | dict[str, Any]:
    """Parse the suggestion id + target status before opening a DB session.

    Pulling the early validation paths into a helper keeps
    ``update_suggestion_status_operation`` below ruff's 6-return ceiling.
    Returns ``(suggestion_uuid, target_status)`` on success or a
    structured error response on failure.
    """
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

    return suggestion_uuid, target_status


@with_response_metadata
async def update_suggestion_status_operation(
    suggestion_id: str,
    status: str,
    *,
    session: AsyncSession | None = None,
    dry_run: bool = False,
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
        session: Optional session to reuse for atomic commit with an
            outer transaction (e.g. ``apply_idempotency``'s
            session-aware path).
        dry_run: If True, validate the transition and return a preview
            (``dry_run: True`` plus a ``preview`` block) without
            mutating storage or emitting audit events.

    Returns:
        Dictionary with success, suggestion_id, status, previous_status, errors.
    """
    logger.info(
        "Updating suggestion status: suggestion_id=%s, status=%s, dry_run=%s",
        suggestion_id,
        status,
        dry_run,
    )

    validated = _validate_update_request(suggestion_id, status)
    if isinstance(validated, dict):
        return validated
    suggestion_uuid, target_status = validated

    async with session_scope(session) as db:
        suggestion_repo = SuggestionRepository(db)
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

        if dry_run:
            logger.info(
                "Suggestion %s status update (dry-run): %s -> %s",
                suggestion_id,
                previous_status.value,
                target_status.value,
            )
            return _build_status_preview(suggestion_id, previous_status, target_status)

        # Atomic ``UPDATE … WHERE status = previous_status AND
        # deleted_at IS NULL`` so two concurrent transitions from
        # the same ``previous_status`` (e.g. PENDING→ACCEPTED vs
        # PENDING→REJECTED) cannot both succeed — last-writer-wins
        # at the in-Python validation layer would have silently
        # clobbered one of them. ``rowcount == 0`` means the row
        # was either soft-deleted or transitioned by a competing
        # caller between this operation's ``get`` and the UPDATE;
        # either way the caller should retry, so we route through
        # the existing CMR envelope.
        if not await suggestion_repo.transition_status(
            suggestion_uuid, previous_status, target_status
        ):
            logger.warning(
                "Suggestion %s status transition %s -> %s lost to a "
                "concurrent update (soft-delete or status change)",
                suggestion_id,
                previous_status.value,
                target_status.value,
            )
            err = ConcurrentModificationError("Suggestion", suggestion_uuid, -1)
            response = make_concurrent_modification_response(
                err,
                extra_details={
                    "suggestion_id": suggestion_id,
                    "expected_source_status": previous_status.value,
                    "target_status": target_status.value,
                },
            )
            response.update(
                {
                    "suggestion_id": suggestion_id,
                    "status": previous_status.value,
                    "previous_status": previous_status.value,
                }
            )
            return response

        # Record audit event for traceability
        event_repo = EventRepository(db)
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
