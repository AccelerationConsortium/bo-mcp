"""Shared campaign lifecycle operations."""

import logging
from typing import Any, Literal
from uuid import UUID

from bo_mcp_server.domain import CampaignStatus
from bo_mcp_server.errors import (
    ErrorCode,
    make_concurrent_modification_response,
    make_error_response,
)
from bo_mcp_server.response_formatter import with_response_metadata
from bo_mcp_server.storage import (
    CampaignRepository,
    ConcurrentModificationError,
    get_session,
)
from bo_mcp_server.subscriptions import notify_campaign_updated_after_commit

logger = logging.getLogger(__name__)

LifecycleAction = Literal["pause", "resume", "terminate", "reopen"]

_ACTION_MAPPING: dict[LifecycleAction, tuple[CampaignStatus, list[CampaignStatus]]] = {
    "pause": (CampaignStatus.PAUSED, [CampaignStatus.RUNNING]),
    "resume": (CampaignStatus.RUNNING, [CampaignStatus.PAUSED]),
    "terminate": (
        CampaignStatus.COMPLETED,
        [CampaignStatus.RUNNING, CampaignStatus.PAUSED, CampaignStatus.CREATED],
    ),
    # A completed campaign keeps its full spec, model history, and results;
    # reopening is the continuation path ("run another N experiments") that
    # otherwise forces clients to rebuild the campaign and replay every prior
    # result as seeds.
    "reopen": (CampaignStatus.RUNNING, [CampaignStatus.COMPLETED]),
}


def _build_dry_run_preview(
    campaign_id: str,
    previous_status: str,
    target_status: CampaignStatus,
    action: LifecycleAction,
) -> dict[str, Any]:
    """Compose the dry-run response envelope shared across actions."""
    return {
        "success": True,
        "dry_run": True,
        "campaign_id": campaign_id,
        "status": previous_status,
        "previous_status": previous_status,
        "preview": {
            "action": action,
            "from_status": previous_status,
            "to_status": target_status.value,
        },
        "errors": [],
    }


def _validate_request(
    campaign_id: str,
    action: LifecycleAction,
) -> tuple[UUID, CampaignStatus, list[CampaignStatus]] | dict[str, Any]:
    """Parse the action + campaign id pair before touching the database.

    Returns the tuple of resolved values on success, or a structured
    error response that already carries the lifecycle-specific keys
    (``campaign_id``, ``status``, ``previous_status``) on failure.
    Pulling these two early-return paths out of the operation body
    keeps the function below ruff's 6-return ceiling.
    """
    if action not in _ACTION_MAPPING:
        valid_actions = list(_ACTION_MAPPING.keys())
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid action '{action}'. Must be one of: {valid_actions}",
            details={"action": action, "valid_actions": valid_actions},
        )

    target_status, valid_from_statuses = _ACTION_MAPPING[action]

    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        logger.warning("Invalid campaign_id format: %s", campaign_id)
        response = make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )
        response.update({"campaign_id": campaign_id, "status": None, "previous_status": None})
        return response

    return campaign_uuid, target_status, valid_from_statuses


@with_response_metadata
async def manage_campaign_lifecycle_operation(
    campaign_id: str,
    action: LifecycleAction,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Manage campaign lifecycle transitions.

    When ``dry_run`` is true the operation runs every validation
    step but skips the commit and notification — the response
    carries a ``dry_run`` flag plus a ``preview`` describing what
    *would* change (current vs. proposed status, action label).
    """
    logger.info(
        "Managing campaign lifecycle: campaign_id=%s, action=%s, dry_run=%s",
        campaign_id,
        action,
        dry_run,
    )

    validated = _validate_request(campaign_id, action)
    if isinstance(validated, dict):
        return validated
    campaign_uuid, target_status, valid_from_statuses = validated

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        campaign = await campaign_repo.get(campaign_uuid)

        if campaign is None:
            response = make_error_response(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )
            response.update({"campaign_id": campaign_id, "status": None, "previous_status": None})
            return response

        previous_status = campaign.status.value

        if campaign.status not in valid_from_statuses:
            response = make_error_response(
                ErrorCode.INVALID_STATE_TRANSITION,
                message=(
                    f"Cannot {action} campaign with status '{campaign.status.value}'. "
                    f"Valid statuses for {action}: {[s.value for s in valid_from_statuses]}"
                ),
                details={
                    "current_status": campaign.status.value,
                    "valid_statuses": [s.value for s in valid_from_statuses],
                    "action": action,
                },
            )
            response.update(
                {
                    "campaign_id": campaign_id,
                    "status": campaign.status.value,
                    "previous_status": previous_status,
                }
            )
            return response

        if dry_run:
            logger.info(
                "Campaign %s lifecycle action %s (dry-run): %s -> %s",
                campaign_id,
                action,
                previous_status,
                target_status.value,
            )
            return _build_dry_run_preview(campaign_id, previous_status, target_status, action)

        updated = campaign.with_status(target_status)
        try:
            await campaign_repo.save(updated, expected_version=campaign.version)
        except ConcurrentModificationError as err:
            logger.warning(
                "Concurrent modification while %sing campaign %s: %s",
                action,
                campaign_id,
                err,
            )
            response = make_concurrent_modification_response(
                err, extra_details={"campaign_id": campaign_id, "action": action}
            )
            response.update(
                {
                    "campaign_id": campaign_id,
                    "status": campaign.status.value,
                    "previous_status": previous_status,
                }
            )
            return response

        # Arm the notification as a post-commit hook so subscribers
        # never see a transition that the surrounding transaction
        # later rolls back (and so a future caller that wires an
        # external session through ``apply_idempotency`` cannot
        # accidentally publish a pre-commit notification).
        notify_campaign_updated_after_commit(session, campaign_uuid)

        logger.info(
            "Campaign %s lifecycle action %s: %s -> %s",
            campaign_id,
            action,
            previous_status,
            target_status.value,
        )

    return {
        "success": True,
        "campaign_id": campaign_id,
        "status": target_status.value,
        "previous_status": previous_status,
        "errors": [],
    }
