"""Batch operations tools for MCP.

This module provides batch operations for common multi-campaign scenarios,
reducing the number of tool calls needed for dashboards and monitoring.

Reference: MCP Best Practices - Batch operations for efficiency
https://modelcontextprotocol.io/docs/best-practices
"""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import CampaignStatus, SuggestionStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    SuggestionRepository,
    get_session,
)

logger = logging.getLogger(__name__)

# Maximum campaigns that can be queried in a single batch
MAX_BATCH_SIZE = 20


@mcp.tool()
async def batch_get_status(
    campaign_ids: list[str],
    verbosity: str = "minimal",
) -> dict[str, Any]:
    """Get status of multiple campaigns in one call.

    Efficient for dashboards and monitoring scenarios where you need to check
    multiple campaigns. Reduces N separate get_diagnostics calls to 1 batch call.

    Args:
        campaign_ids: List of campaign UUIDs to query (max 20).
        verbosity: Response verbosity level. Options:
            - "minimal": ~30 tokens per campaign - status, iteration, n_results only
            - "standard": ~80 tokens per campaign - includes health, progress, key_metric
            - "detailed": ~150 tokens per campaign - includes convergence info

    Returns:
        Dictionary with:
            - success: Boolean indicating if retrieval succeeded
            - campaigns: Dict mapping campaign_id to status info
            - failed_ids: List of campaign_ids that could not be retrieved
            - errors: List of error messages (if any)

    Example minimal response:
        {
            "success": true,
            "campaigns": {
                "uuid-1": {"status": "running", "iteration": 5, "n_results": 10},
                "uuid-2": {"status": "completed", "iteration": 12, "n_results": 25}
            },
            "failed_ids": [],
            "errors": []
        }
    """
    logger.info(
        "Batch getting status for %d campaigns, verbosity=%s",
        len(campaign_ids),
        verbosity,
    )

    # Validate verbosity
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    # Validate campaign_ids list
    if not campaign_ids:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="At least one campaign_id is required",
        )

    if len(campaign_ids) > MAX_BATCH_SIZE:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Too many campaign_ids ({len(campaign_ids)}). Maximum is {MAX_BATCH_SIZE}.",
            details={"max_batch_size": MAX_BATCH_SIZE, "requested": len(campaign_ids)},
        )

    # Parse and validate UUIDs
    valid_uuids: list[tuple[str, UUID]] = []
    invalid_ids: list[str] = []

    for cid in campaign_ids:
        try:
            valid_uuids.append((cid, UUID(cid)))
        except ValueError:
            invalid_ids.append(cid)

    errors: list[str] = []
    if invalid_ids:
        errors.append(f"Invalid UUID format: {invalid_ids}")

    campaigns_info: dict[str, dict[str, Any]] = {}
    failed_ids: list[str] = list(invalid_ids)  # Start with invalid IDs

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)
        suggestion_repo = SuggestionRepository(session)

        for cid_str, cid_uuid in valid_uuids:
            campaign = await campaign_repo.get(cid_uuid)

            if campaign is None:
                failed_ids.append(cid_str)
                continue

            # Get spec for campaign name
            spec = await spec_repo.get(campaign.spec_id)
            campaign_name = spec.name if spec else "Unknown"

            # Get result count
            results = await result_repo.list_by_campaign(campaign.id)
            n_results = len(results)

            if verbosity_level == VerbosityLevel.MINIMAL:
                campaigns_info[cid_str] = {
                    "name": campaign_name,
                    "status": campaign.status.value,
                    "iteration": campaign.iteration,
                    "n_results": n_results,
                }
            elif verbosity_level == VerbosityLevel.STANDARD:
                # Get pending suggestions count
                suggestions = await suggestion_repo.list_by_campaign(campaign.id)
                n_pending = len([s for s in suggestions if s.status == SuggestionStatus.PENDING])

                # Compute simple health indicator
                health = "healthy"
                if campaign.status == CampaignStatus.FAILED:
                    health = "critical"
                elif campaign.status == CampaignStatus.PAUSED:
                    health = "paused"
                elif n_results == 0 and campaign.iteration > 1:
                    health = "warning"

                # Compute simple key metric
                key_metric: dict[str, Any] = {}
                if spec and len(spec.objectives) == 1 and results:
                    obj = spec.objectives[0]
                    values = [
                        r.objective_values.get(obj.name)
                        for r in results
                        if obj.name in r.objective_values
                    ]
                    if values:
                        valid_values = [v for v in values if v is not None]
                        if valid_values:
                            if obj.is_minimize:
                                key_metric["best_value"] = min(valid_values)
                            else:
                                key_metric["best_value"] = max(valid_values)
                elif spec and len(spec.objectives) >= 2 and campaign.hypervolume_history:
                    key_metric["hypervolume"] = campaign.hypervolume_history[-1]

                campaigns_info[cid_str] = {
                    "name": campaign_name,
                    "status": campaign.status.value,
                    "iteration": campaign.iteration,
                    "n_results": n_results,
                    "n_pending_suggestions": n_pending,
                    "health": health,
                    "key_metric": key_metric,
                }
            else:  # DETAILED
                suggestions = await suggestion_repo.list_by_campaign(campaign.id)
                n_pending = len([s for s in suggestions if s.status == SuggestionStatus.PENDING])

                # Health status
                health = "healthy"
                if campaign.status == CampaignStatus.FAILED:
                    health = "critical"
                elif campaign.status == CampaignStatus.PAUSED:
                    health = "paused"
                elif n_results == 0 and campaign.iteration > 1:
                    health = "warning"

                # Key metric
                key_metric = {}
                is_single_objective = spec is not None and len(spec.objectives) == 1
                if is_single_objective and results and spec is not None:
                    obj = spec.objectives[0]
                    values = [
                        r.objective_values.get(obj.name)
                        for r in results
                        if obj.name in r.objective_values
                    ]
                    valid_values = [v for v in values if v is not None]
                    if valid_values:
                        if obj.is_minimize:
                            key_metric["best_value"] = min(valid_values)
                        else:
                            key_metric["best_value"] = max(valid_values)
                elif spec and len(spec.objectives) >= 2:
                    if campaign.hypervolume_history:
                        key_metric["hypervolume"] = campaign.hypervolume_history[-1]
                    key_metric["n_pareto_points"] = None  # Would need to compute

                # Convergence info (simplified)
                convergence_info: dict[str, Any] = {"converged": False}
                if campaign.hypervolume_history and len(campaign.hypervolume_history) >= 5:
                    # Simple convergence check
                    recent = campaign.hypervolume_history[-5:]
                    if len(set(recent)) == 1 or (max(recent) - min(recent)) < 0.001:
                        convergence_info["converged"] = True
                        convergence_info["reason"] = "Hypervolume stable"

                campaigns_info[cid_str] = {
                    "name": campaign_name,
                    "status": campaign.status.value,
                    "iteration": campaign.iteration,
                    "n_results": n_results,
                    "n_pending_suggestions": n_pending,
                    "health": health,
                    "key_metric": key_metric,
                    "convergence": convergence_info,
                    "created_at": campaign.created_at.isoformat(),
                    "owner_id": str(campaign.owner_id),
                }

    if failed_ids:
        errors.append(f"Could not retrieve campaigns: {failed_ids}")

    logger.info(
        "Batch status complete: %d succeeded, %d failed",
        len(campaigns_info),
        len(failed_ids),
    )

    return {
        "success": len(campaigns_info) > 0 or len(failed_ids) == 0,
        "campaigns": campaigns_info,
        "failed_ids": failed_ids,
        "errors": errors,
    }
