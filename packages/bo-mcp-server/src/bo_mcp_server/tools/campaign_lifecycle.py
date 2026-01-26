"""Campaign lifecycle tools for MCP.

This module provides tools for managing campaign lifecycle state transitions.

Per MCP best practices, we provide both:
1. Individual tools (pause_campaign, resume_campaign, terminate_campaign) for backward compatibility
2. A consolidated tool (manage_campaign_lifecycle) for reduced cognitive load

Reference: MCP Best Practices - "Avoid mapping every API endpoint to a new MCP tool.
Instead, group related tasks."
https://modelcontextprotocol.io/docs/best-practices
"""

import logging
from typing import Any, Literal
from uuid import UUID

from bo_mcp_server.domain import CampaignStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import CampaignRepository, get_session

logger = logging.getLogger(__name__)


@mcp.tool()
async def pause_campaign(campaign_id: str) -> dict[str, Any]:
    """Pause an active campaign.

    Args:
        campaign_id: UUID of the campaign to pause

    Returns:
        Dictionary with success status and campaign info
    """
    return await _change_campaign_status(
        campaign_id,
        target_status=CampaignStatus.PAUSED,
        valid_from_statuses=[CampaignStatus.RUNNING],
        action="pause",
    )


@mcp.tool()
async def resume_campaign(campaign_id: str) -> dict[str, Any]:
    """Resume a paused campaign.

    Args:
        campaign_id: UUID of the campaign to resume

    Returns:
        Dictionary with success status and campaign info
    """
    return await _change_campaign_status(
        campaign_id,
        target_status=CampaignStatus.RUNNING,
        valid_from_statuses=[CampaignStatus.PAUSED],
        action="resume",
    )


@mcp.tool()
async def terminate_campaign(campaign_id: str) -> dict[str, Any]:
    """Terminate a campaign (cannot be undone).

    Args:
        campaign_id: UUID of the campaign to terminate

    Returns:
        Dictionary with success status and campaign info
    """
    return await _change_campaign_status(
        campaign_id,
        target_status=CampaignStatus.COMPLETED,
        valid_from_statuses=[CampaignStatus.RUNNING, CampaignStatus.PAUSED, CampaignStatus.CREATED],
        action="terminate",
    )


async def _change_campaign_status(
    campaign_id: str,
    target_status: CampaignStatus,
    valid_from_statuses: list[CampaignStatus],
    action: str,
) -> dict[str, Any]:
    """Helper to change campaign status with validation."""
    logger.info("Attempting to %s campaign %s", action, campaign_id)

    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        logger.warning("Invalid campaign_id format: %s", campaign_id)
        response = make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )
        response.update({"campaign_id": campaign_id, "status": None})
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
            response.update({"campaign_id": campaign_id, "status": None})
            return response

        if campaign.status not in valid_from_statuses:
            response = make_error_response(
                ErrorCode.INVALID_STATE_TRANSITION,
                message=(
                    f"Cannot {action} campaign with status '{campaign.status.value}'. "
                    f"Valid statuses: {[s.value for s in valid_from_statuses]}"
                ),
                details={
                    "current_status": campaign.status.value,
                    "valid_statuses": [s.value for s in valid_from_statuses],
                },
            )
            response.update({"campaign_id": campaign_id, "status": campaign.status.value})
            return response

        updated = campaign.with_status(target_status)
        await campaign_repo.save(updated, expected_version=campaign.version)

        logger.info(
            "Campaign %s status changed: %s -> %s",
            campaign_id,
            campaign.status.value,
            target_status.value,
        )
        return {
            "success": True,
            "campaign_id": campaign_id,
            "status": target_status.value,
            "errors": [],
        }


# =============================================================================
# Consolidated Lifecycle Tool (v3.3)
# =============================================================================

# Mapping from action to (target_status, valid_from_statuses)
_ACTION_MAPPING: dict[str, tuple[CampaignStatus, list[CampaignStatus]]] = {
    "pause": (CampaignStatus.PAUSED, [CampaignStatus.RUNNING]),
    "resume": (CampaignStatus.RUNNING, [CampaignStatus.PAUSED]),
    "terminate": (
        CampaignStatus.COMPLETED,
        [CampaignStatus.RUNNING, CampaignStatus.PAUSED, CampaignStatus.CREATED],
    ),
}


@mcp.tool()
async def manage_campaign_lifecycle(
    campaign_id: str,
    action: Literal["pause", "resume", "terminate"],
) -> dict[str, Any]:
    """Manage campaign lifecycle with a single consolidated tool.

    This tool consolidates pause_campaign, resume_campaign, and terminate_campaign
    into a single interface for reduced cognitive load. Use this as the preferred
    tool for lifecycle management.

    Args:
        campaign_id: UUID of the campaign to manage.
        action: The lifecycle action to perform:
            - "pause": Pause an active (RUNNING) campaign. Can be resumed later.
            - "resume": Resume a paused campaign back to RUNNING state.
            - "terminate": End a campaign permanently. Cannot be undone.
              Works on CREATED, RUNNING, or PAUSED campaigns.

    Returns:
        Dictionary with:
            - success: Boolean indicating if the action succeeded
            - campaign_id: UUID of the campaign
            - status: New status after the action (or current status if failed)
            - previous_status: Status before the action (if successful)
            - errors: List of error messages (if any)

    Example:
        # Pause a running campaign
        {"campaign_id": "abc-123", "action": "pause"}

        # Resume a paused campaign
        {"campaign_id": "abc-123", "action": "resume"}

        # Terminate a campaign
        {"campaign_id": "abc-123", "action": "terminate"}

    State Transitions:
        pause:     RUNNING → PAUSED
        resume:    PAUSED → RUNNING
        terminate: CREATED/RUNNING/PAUSED → COMPLETED
    """
    logger.info("Managing campaign lifecycle: campaign_id=%s, action=%s", campaign_id, action)

    # Validate action
    if action not in _ACTION_MAPPING:
        valid_actions = list(_ACTION_MAPPING.keys())
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid action '{action}'. Must be one of: {valid_actions}",
            details={"action": action, "valid_actions": valid_actions},
        )

    target_status, valid_from_statuses = _ACTION_MAPPING[action]

    # Validate campaign_id
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
