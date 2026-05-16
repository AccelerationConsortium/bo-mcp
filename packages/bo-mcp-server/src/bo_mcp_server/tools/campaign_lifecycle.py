"""Campaign lifecycle tools for MCP.

Three individual tools wrap the shared lifecycle operation for better
discoverability, audit trail clarity, and error specificity.
"""

from typing import Any

from bo_mcp_server.operations.campaign_lifecycle import manage_campaign_lifecycle_operation
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import DESTRUCTIVE_MUTATION, IDEMPOTENT_MUTATION
from bo_mcp_server.trace_context import bind_trace_id


@mcp.tool(name="bo_pause_campaign", annotations=IDEMPOTENT_MUTATION)
async def pause_campaign(
    campaign_id: str,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Pause a running campaign.

    Workflow: Call when you need to temporarily halt optimization.

    Pausing preserves all state (model, suggestions, results). The campaign
    can be resumed later with bo_resume_campaign.

    Args:
        campaign_id: UUID of the campaign to pause. Must be in RUNNING status.
        dry_run: If True, validate the transition and return a preview without
            committing. The response carries ``dry_run: True`` and a
            ``preview`` describing the planned status change.
        trace_id: Optional workflow trace id. When supplied it is bound
            for the duration of the call so the audit event and response
            ``_metadata.trace_id`` both echo it. Agents stringing
            multiple MCP calls together should re-use the same id.

    Returns:
        Dictionary with success, campaign_id, status, previous_status, errors.
    """
    with bind_trace_id(trace_id):
        return await manage_campaign_lifecycle_operation(
            campaign_id=campaign_id, action="pause", dry_run=dry_run
        )


@mcp.tool(name="bo_resume_campaign", annotations=IDEMPOTENT_MUTATION)
async def resume_campaign(
    campaign_id: str,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Resume a paused campaign.

    Workflow: Call to continue a previously paused campaign.

    Resumes suggestion generation and result submission for the campaign.

    Args:
        campaign_id: UUID of the campaign to resume. Must be in PAUSED status.
        dry_run: If True, validate the transition and return a preview without
            committing. The response carries ``dry_run: True`` and a
            ``preview`` describing the planned status change.
        trace_id: Optional workflow trace id. See ``bo_pause_campaign``.

    Returns:
        Dictionary with success, campaign_id, status, previous_status, errors.
    """
    with bind_trace_id(trace_id):
        return await manage_campaign_lifecycle_operation(
            campaign_id=campaign_id, action="resume", dry_run=dry_run
        )


@mcp.tool(name="bo_terminate_campaign", annotations=DESTRUCTIVE_MUTATION)
async def terminate_campaign(
    campaign_id: str,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Terminate a campaign, marking it as completed.

    Workflow: Call when the optimization goal has been reached or further
    experiments are not worthwhile.

    This is irreversible.

    Args:
        campaign_id: UUID of the campaign to terminate.
            Must be in CREATED, RUNNING, or PAUSED status.
        dry_run: If True, validate the transition and return a preview without
            committing. Recommended for agentic workflows that want to confirm
            this irreversible action before executing.
        trace_id: Optional workflow trace id. See ``bo_pause_campaign``.

    Returns:
        Dictionary with success, campaign_id, status, previous_status, errors.
    """
    with bind_trace_id(trace_id):
        return await manage_campaign_lifecycle_operation(
            campaign_id=campaign_id, action="terminate", dry_run=dry_run
        )
