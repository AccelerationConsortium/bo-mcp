"""Shared campaign lifecycle operations."""

import logging
from typing import Any, Literal
from uuid import UUID

from bo_mcp_server.domain import CampaignStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.storage import CampaignRepository, get_session

logger = logging.getLogger(__name__)

LifecycleAction = Literal["pause", "resume", "terminate"]

_ACTION_MAPPING: dict[LifecycleAction, tuple[CampaignStatus, list[CampaignStatus]]] = {
    "pause": (CampaignStatus.PAUSED, [CampaignStatus.RUNNING]),
    "resume": (CampaignStatus.RUNNING, [CampaignStatus.PAUSED]),
    "terminate": (
        CampaignStatus.COMPLETED,
        [CampaignStatus.RUNNING, CampaignStatus.PAUSED, CampaignStatus.CREATED],
    ),
}


async def manage_campaign_lifecycle_operation(
    campaign_id: str,
    action: LifecycleAction,
) -> dict[str, Any]:
    """Manage campaign lifecycle transitions."""
    logger.info("Managing campaign lifecycle: campaign_id=%s, action=%s", campaign_id, action)

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

        updated = campaign.with_status(target_status)
        await campaign_repo.save(updated, expected_version=campaign.version)

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
