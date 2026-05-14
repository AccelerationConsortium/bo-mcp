"""Campaign lifecycle tools for MCP.

Three individual tools wrap the shared lifecycle operation for better
discoverability, audit trail clarity, and error specificity.
"""

from typing import Any

from bo_mcp_server.operations.campaign_lifecycle import manage_campaign_lifecycle_operation
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import DESTRUCTIVE_MUTATION, IDEMPOTENT_MUTATION


@mcp.tool(name="bo_pause_campaign", annotations=IDEMPOTENT_MUTATION)
async def pause_campaign(campaign_id: str) -> dict[str, Any]:
    """Pause a running campaign.

    Workflow: Call when you need to temporarily halt optimization.

    Pausing preserves all state (model, suggestions, results). The campaign
    can be resumed later with bo_resume_campaign.

    Args:
        campaign_id: UUID of the campaign to pause. Must be in RUNNING status.

    Returns:
        Dictionary with success, campaign_id, status, previous_status, errors.
    """
    return await manage_campaign_lifecycle_operation(campaign_id=campaign_id, action="pause")


@mcp.tool(name="bo_resume_campaign", annotations=IDEMPOTENT_MUTATION)
async def resume_campaign(campaign_id: str) -> dict[str, Any]:
    """Resume a paused campaign.

    Workflow: Call to continue a previously paused campaign.

    Resumes suggestion generation and result submission for the campaign.

    Args:
        campaign_id: UUID of the campaign to resume. Must be in PAUSED status.

    Returns:
        Dictionary with success, campaign_id, status, previous_status, errors.
    """
    return await manage_campaign_lifecycle_operation(campaign_id=campaign_id, action="resume")


@mcp.tool(name="bo_terminate_campaign", annotations=DESTRUCTIVE_MUTATION)
async def terminate_campaign(campaign_id: str) -> dict[str, Any]:
    """Terminate a campaign, marking it as completed.

    Workflow: Call when the optimization goal has been reached or further
    experiments are not worthwhile.

    This is irreversible.

    Args:
        campaign_id: UUID of the campaign to terminate.
            Must be in CREATED, RUNNING, or PAUSED status.

    Returns:
        Dictionary with success, campaign_id, status, previous_status, errors.
    """
    return await manage_campaign_lifecycle_operation(campaign_id=campaign_id, action="terminate")
