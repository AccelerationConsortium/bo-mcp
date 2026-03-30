"""List campaigns tool for MCP.

This tool wraps the campaigns://list resource functionality to provide
a tool-based interface for agents that prefer tools over MCP resources.

Reference: MCP Tool Best Practices - Agents prefer tools for consistent workflow.
https://modelcontextprotocol.io/docs/concepts/tools
"""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import Campaign, CampaignSpec, CampaignStatus
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

MAX_LIMIT = 100


def _build_minimal_summary(campaign: Campaign, name: str) -> dict[str, Any]:
    return {
        "campaign_id": str(campaign.id),
        "name": name,
        "status": campaign.status.value,
    }


def _build_standard_summary(campaign: Campaign, name: str, n_results: int) -> dict[str, Any]:
    return {
        "campaign_id": str(campaign.id),
        "name": name,
        "status": campaign.status.value,
        "iteration": campaign.iteration,
        "n_results": n_results,
        "created_at": campaign.created_at.isoformat(),
    }


def _build_detailed_summary(
    campaign: Campaign,
    name: str,
    n_results: int,
    spec: CampaignSpec | None,
) -> dict[str, Any]:
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
    return {
        "campaign_id": str(campaign.id),
        "name": name,
        "status": campaign.status.value,
        "iteration": campaign.iteration,
        "n_results": n_results,
        "created_at": campaign.created_at.isoformat(),
        "owner_id": str(campaign.owner_id),
        "spec_id": str(campaign.spec_id),
        "spec_summary": spec_summary,
        "has_hypervolume_history": len(campaign.hypervolume_history) > 0,
        "has_turbo_state": campaign.turbo_state is not None,
    }


def _validate_params(
    owner_id: str | None, status: str | None, verbosity: str
) -> dict[str, Any] | tuple[UUID | None, CampaignStatus | None, VerbosityLevel]:
    """Validate input parameters. Returns error dict or parsed values."""
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    owner_uuid: UUID | None = None
    if owner_id is not None:
        try:
            owner_uuid = UUID(owner_id)
        except ValueError:
            return make_error_response(
                ErrorCode.VALIDATION_FAILED,
                message="Invalid owner_id format",
                details={"owner_id": owner_id},
            )

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

    return owner_uuid, status_filter, verbosity_level


@mcp.tool(name="bo_list_campaigns")
async def list_campaigns(
    owner_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """List optimization campaigns with optional filtering and pagination.

    Args:
        owner_id: Optional UUID of the owner to filter campaigns by.
        status: Optional status filter. Valid values:
            - "created": Campaigns that have been created but not yet started
            - "running": Active campaigns
            - "paused": Paused campaigns
            - "completed": Finished campaigns
            - "failed": Failed campaigns
        limit: Maximum number of campaigns to return (default 20, max 100).
        offset: Number of campaigns to skip for pagination (default 0).
        verbosity: Response verbosity level. Options:
            - "minimal": ~50 tokens - campaign_id, name, status only
            - "standard": ~200 tokens - includes iteration, n_results, created_at
            - "detailed": ~500+ tokens - includes full spec summary and metrics

    Returns:
        Dictionary with:
            - success: Boolean indicating if retrieval succeeded
            - campaigns: List of campaign summaries
            - total_count: Total number of campaigns matching filters
            - limit: Applied limit
            - offset: Applied offset
            - errors: List of error messages (if any)
    """
    logger.info(
        "Listing campaigns: owner_id=%s, status=%s, limit=%d, offset=%d, verbosity=%s",
        owner_id,
        status,
        limit,
        offset,
        verbosity,
    )

    validated = _validate_params(owner_id, status, verbosity)
    if isinstance(validated, dict):
        return validated
    owner_uuid, status_filter, verbosity_level = validated

    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)

        campaigns, total_count = await campaign_repo.list_filtered(
            owner_id=owner_uuid,
            status=status_filter,
            limit=limit,
            offset=offset,
        )

        # Batch-fetch specs for all campaigns in one query
        spec_ids = list({c.spec_id for c in campaigns})
        specs = await spec_repo.get_by_ids(spec_ids)

        # Batch-fetch result counts if needed
        result_counts: dict[UUID, int] = {}
        if verbosity_level != VerbosityLevel.MINIMAL:
            campaign_ids = [c.id for c in campaigns]
            result_counts = await result_repo.count_by_campaigns(campaign_ids)

        # Build response
        campaign_summaries: list[dict[str, Any]] = []
        for campaign in campaigns:
            spec = specs.get(campaign.spec_id)
            name = spec.name if spec else "Unknown"
            n_results = result_counts.get(campaign.id, 0)

            if verbosity_level == VerbosityLevel.MINIMAL:
                campaign_summaries.append(_build_minimal_summary(campaign, name))
            elif verbosity_level == VerbosityLevel.STANDARD:
                campaign_summaries.append(_build_standard_summary(campaign, name, n_results))
            else:
                campaign_summaries.append(_build_detailed_summary(campaign, name, n_results, spec))

    logger.info(
        "Listed %d campaigns (total matching: %d)",
        len(campaign_summaries),
        total_count,
    )

    return {
        "success": True,
        "campaigns": campaign_summaries,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "errors": [],
    }
