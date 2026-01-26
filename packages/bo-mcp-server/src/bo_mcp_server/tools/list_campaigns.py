"""List campaigns tool for MCP.

This tool wraps the campaigns://list resource functionality to provide
a tool-based interface for agents that prefer tools over MCP resources.

Reference: MCP Tool Best Practices - Agents prefer tools for consistent workflow.
https://modelcontextprotocol.io/docs/concepts/tools
"""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import CampaignStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)


@mcp.tool()
async def list_campaigns(
    owner_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """List optimization campaigns with optional filtering.

    This tool provides a tool-based alternative to the campaigns://list resource,
    which is more convenient for agents that prefer working with tools.

    Args:
        owner_id: Optional UUID of the owner to filter campaigns by.
        status: Optional status filter. Valid values:
            - "created": Campaigns that have been created but not yet started
            - "running": Active campaigns
            - "paused": Paused campaigns
            - "completed": Finished campaigns
            - "failed": Failed campaigns
        limit: Maximum number of campaigns to return (default 20, max 100).
        verbosity: Response verbosity level. Options:
            - "minimal": ~50 tokens - campaign_id, name, status only
            - "standard": ~200 tokens - includes iteration, n_results, created_at
            - "detailed": ~500+ tokens - includes full spec summary and metrics

    Returns:
        Dictionary with:
            - success: Boolean indicating if retrieval succeeded
            - campaigns: List of campaign summaries
            - total_count: Total number of campaigns matching filters
            - errors: List of error messages (if any)

    Example minimal response:
        {
            "success": true,
            "campaigns": [
                {"campaign_id": "uuid", "name": "Campaign A", "status": "running"},
                {"campaign_id": "uuid", "name": "Campaign B", "status": "completed"}
            ],
            "total_count": 2,
            "errors": []
        }
    """
    logger.info(
        "Listing campaigns: owner_id=%s, status=%s, limit=%d, verbosity=%s",
        owner_id,
        status,
        limit,
        verbosity,
    )

    # Validate verbosity parameter
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    # Validate owner_id if provided
    owner_uuid: UUID | None = None
    if owner_id is not None:
        try:
            owner_uuid = UUID(owner_id)
        except ValueError:
            logger.warning("Invalid owner_id format: %s", owner_id)
            return make_error_response(
                ErrorCode.VALIDATION_FAILED,
                message="Invalid owner_id format",
                details={"owner_id": owner_id},
            )

    # Validate status if provided
    status_filter: CampaignStatus | None = None
    if status is not None:
        try:
            status_filter = CampaignStatus(status)
        except ValueError:
            valid_statuses = [s.value for s in CampaignStatus]
            return make_error_response(
                ErrorCode.VALIDATION_FAILED,
                message=f"Invalid status '{status}'. Must be one of: {valid_statuses}",
                details={"status": status, "valid_statuses": valid_statuses},
            )

    # Validate limit
    if limit < 1:
        limit = 1
    elif limit > 100:
        limit = 100

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)

        # Get all campaigns (we'll filter in memory for now)
        all_campaigns = await campaign_repo.list_all()

        # Apply filters
        filtered_campaigns = all_campaigns
        if owner_uuid is not None:
            filtered_campaigns = [c for c in filtered_campaigns if c.owner_id == owner_uuid]
        if status_filter is not None:
            filtered_campaigns = [c for c in filtered_campaigns if c.status == status_filter]

        # Track total before limiting
        total_count = len(filtered_campaigns)

        # Sort by created_at descending (most recent first) and apply limit
        filtered_campaigns = sorted(
            filtered_campaigns,
            key=lambda c: c.created_at,
            reverse=True,
        )[:limit]

        # Build response based on verbosity
        campaign_summaries: list[dict[str, Any]] = []

        for campaign in filtered_campaigns:
            # Get spec for campaign name
            spec = await spec_repo.get(campaign.spec_id)
            campaign_name = spec.name if spec else "Unknown"

            if verbosity_level == VerbosityLevel.MINIMAL:
                campaign_summaries.append(
                    {
                        "campaign_id": str(campaign.id),
                        "name": campaign_name,
                        "status": campaign.status.value,
                    }
                )
            elif verbosity_level == VerbosityLevel.STANDARD:
                # Get result count
                results = await result_repo.list_by_campaign(campaign.id)
                campaign_summaries.append(
                    {
                        "campaign_id": str(campaign.id),
                        "name": campaign_name,
                        "status": campaign.status.value,
                        "iteration": campaign.iteration,
                        "n_results": len(results),
                        "created_at": campaign.created_at.isoformat(),
                    }
                )
            else:  # DETAILED
                results = await result_repo.list_by_campaign(campaign.id)
                spec_summary = None
                if spec:
                    spec_summary = {
                        "n_parameters": len(spec.parameters),
                        "n_objectives": len(spec.objectives),
                        "n_constraints": len(spec.constraints) if spec.constraints else 0,
                        "batch_size": spec.batch_size,
                        "parameter_names": [p.name for p in spec.parameters],
                        "objective_names": [o.name for o in spec.objectives],
                    }
                campaign_summaries.append(
                    {
                        "campaign_id": str(campaign.id),
                        "name": campaign_name,
                        "status": campaign.status.value,
                        "iteration": campaign.iteration,
                        "n_results": len(results),
                        "created_at": campaign.created_at.isoformat(),
                        "owner_id": str(campaign.owner_id),
                        "spec_id": str(campaign.spec_id),
                        "spec_summary": spec_summary,
                        "has_hypervolume_history": len(campaign.hypervolume_history) > 0,
                        "has_turbo_state": campaign.turbo_state is not None,
                    }
                )

        logger.info(
            "Listed %d campaigns (total matching: %d)",
            len(campaign_summaries),
            total_count,
        )

        return {
            "success": True,
            "campaigns": campaign_summaries,
            "total_count": total_count,
            "errors": [],
        }
