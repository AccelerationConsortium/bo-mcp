"""Campaign lifecycle tools for MCP.

Individual tools wrap the shared lifecycle operation for better
discoverability, audit trail clarity, and error specificity.
"""

from typing import cast

from bo_mcp_server.operations.campaign_lifecycle import manage_campaign_lifecycle_operation
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import DESTRUCTIVE_MUTATION, IDEMPOTENT_MUTATION
from bo_mcp_server.tools.response_models import CampaignLifecycleToolResponse
from bo_mcp_server.trace_context import bind_trace_id

# The operation layer returns a plain ``dict[str, Any]`` (see
# ``manage_campaign_lifecycle_operation``); FastMCP validates that dict
# against the declared return type at the tool boundary (Pydantic,
# ``extra="allow"``), which is where the real type guarantee comes from.
# ``cast`` here only tells the static checker about that already-enforced
# runtime contract -- it does not change type-checking behavior elsewhere.
#
# The return type is deliberately a single model, never a ``Model |
# ErrorEnvelope`` union: FastMCP can only treat a return annotation as a
# top-level JSON object (and expose success/error fields directly in
# ``structuredContent``) when it resolves to one object-schema type. A
# union of two object schemas isn't representable as one top-level object
# schema, so FastMCP falls back to wrapping the whole result under a
# ``{"result": ...}`` key -- silently breaking every caller that reads
# e.g. ``response["success"]``. ``extra="allow"`` on the single model
# already accepts the differently-shaped error envelope, so the union
# added no validation value while triggering that wrapping.
_LifecycleResult = CampaignLifecycleToolResponse


@mcp.tool(name="bo_pause_campaign", annotations=IDEMPOTENT_MUTATION)
async def pause_campaign(
    campaign_id: str,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> CampaignLifecycleToolResponse:
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
        return cast(
            _LifecycleResult,
            await manage_campaign_lifecycle_operation(
                campaign_id=campaign_id, action="pause", dry_run=dry_run
            ),
        )


@mcp.tool(name="bo_resume_campaign", annotations=IDEMPOTENT_MUTATION)
async def resume_campaign(
    campaign_id: str,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> CampaignLifecycleToolResponse:
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
        return cast(
            _LifecycleResult,
            await manage_campaign_lifecycle_operation(
                campaign_id=campaign_id, action="resume", dry_run=dry_run
            ),
        )


@mcp.tool(name="bo_terminate_campaign", annotations=DESTRUCTIVE_MUTATION)
async def terminate_campaign(
    campaign_id: str,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> CampaignLifecycleToolResponse:
    """Terminate a campaign, marking it as completed.

    Workflow: Call when the optimization goal has been reached or further
    experiments are not worthwhile.

    A terminated campaign refuses suggestions and results until it is
    explicitly reopened with bo_reopen_campaign.

    Args:
        campaign_id: UUID of the campaign to terminate.
            Must be in CREATED, RUNNING, or PAUSED status.
        dry_run: If True, validate the transition and return a preview without
            committing. Recommended for agentic workflows that want to confirm
            this action before executing.
        trace_id: Optional workflow trace id. See ``bo_pause_campaign``.

    Returns:
        Dictionary with success, campaign_id, status, previous_status, errors.
    """
    with bind_trace_id(trace_id):
        return cast(
            _LifecycleResult,
            await manage_campaign_lifecycle_operation(
                campaign_id=campaign_id, action="terminate", dry_run=dry_run
            ),
        )


@mcp.tool(name="bo_reopen_campaign", annotations=IDEMPOTENT_MUTATION)
async def reopen_campaign(
    campaign_id: str,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> CampaignLifecycleToolResponse:
    """Reopen a completed campaign so optimization can continue.

    Workflow: Call to continue a campaign that was terminated or reached its
    stopping criteria, e.g. to run another batch of experiments. The campaign
    keeps its spec, model history, and results — reopening is the alternative
    to recreating the campaign and replaying prior results as seeds.

    Args:
        campaign_id: UUID of the campaign to reopen. Must be in COMPLETED
            status.
        dry_run: If True, validate the transition and return a preview without
            committing. The response carries ``dry_run: True`` and a
            ``preview`` describing the planned status change.
        trace_id: Optional workflow trace id. See ``bo_pause_campaign``.

    Returns:
        Dictionary with success, campaign_id, status, previous_status, errors.
    """
    with bind_trace_id(trace_id):
        return cast(
            _LifecycleResult,
            await manage_campaign_lifecycle_operation(
                campaign_id=campaign_id, action="reopen", dry_run=dry_run
            ),
        )
