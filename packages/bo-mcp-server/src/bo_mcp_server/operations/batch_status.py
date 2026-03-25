"""Shared batch status operations."""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import CampaignStatus, SuggestionStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    SuggestionRepository,
    get_session,
)

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 20


async def batch_get_status_operation(
    campaign_ids: list[str],
    verbosity: str = "minimal",
) -> dict[str, Any]:
    """Get status for multiple campaigns in one call."""
    logger.info(
        "Batch getting status for %d campaigns, verbosity=%s",
        len(campaign_ids),
        verbosity,
    )

    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

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

    valid_uuids: list[tuple[str, UUID]] = []
    invalid_ids: list[str] = []
    for campaign_id in campaign_ids:
        try:
            valid_uuids.append((campaign_id, UUID(campaign_id)))
        except ValueError:
            invalid_ids.append(campaign_id)

    errors: list[str] = []
    if invalid_ids:
        errors.append(f"Invalid UUID format: {invalid_ids}")

    campaigns_info: dict[str, dict[str, Any]] = {}
    failed_ids: list[str] = list(invalid_ids)

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)
        suggestion_repo = SuggestionRepository(session)

        for campaign_id_str, campaign_uuid in valid_uuids:
            campaign = await campaign_repo.get(campaign_uuid)
            if campaign is None:
                failed_ids.append(campaign_id_str)
                continue

            spec = await spec_repo.get(campaign.spec_id)
            campaign_name = spec.name if spec else "Unknown"
            results = await result_repo.list_by_campaign(campaign.id)
            n_results = len(results)

            if verbosity_level == VerbosityLevel.MINIMAL:
                campaigns_info[campaign_id_str] = {
                    "name": campaign_name,
                    "status": campaign.status.value,
                    "iteration": campaign.iteration,
                    "n_results": n_results,
                }
                continue

            suggestions = await suggestion_repo.list_by_campaign(campaign.id)
            n_pending = len([s for s in suggestions if s.status == SuggestionStatus.PENDING])

            health = "healthy"
            if campaign.status == CampaignStatus.FAILED:
                health = "critical"
            elif campaign.status == CampaignStatus.PAUSED:
                health = "paused"
            elif n_results == 0 and campaign.iteration > 1:
                health = "warning"

            key_metric: dict[str, Any] = {}
            if spec and len(spec.objectives) == 1 and results:
                objective = spec.objectives[0]
                values = [
                    result.objective_values.get(objective.name)
                    for result in results
                    if objective.name in result.objective_values
                ]
                valid_values = [value for value in values if value is not None]
                if valid_values:
                    key_metric["best_value"] = (
                        min(valid_values) if objective.is_minimize else max(valid_values)
                    )
            elif spec and len(spec.objectives) >= 2 and campaign.hypervolume_history:
                key_metric["hypervolume"] = campaign.hypervolume_history[-1]

            campaigns_info[campaign_id_str] = {
                "name": campaign_name,
                "status": campaign.status.value,
                "iteration": campaign.iteration,
                "n_results": n_results,
                "n_pending_suggestions": n_pending,
                "health": health,
                "key_metric": key_metric,
            }

            if verbosity_level == VerbosityLevel.DETAILED:
                convergence_info: dict[str, Any] = {"converged": False}
                if campaign.hypervolume_history and len(campaign.hypervolume_history) >= 5:
                    recent = campaign.hypervolume_history[-5:]
                    if len(set(recent)) == 1 or (max(recent) - min(recent)) < 0.001:
                        convergence_info["converged"] = True
                        convergence_info["reason"] = "Hypervolume stable"

                campaigns_info[campaign_id_str].update(
                    {
                        "convergence": convergence_info,
                        "created_at": campaign.created_at.isoformat(),
                        "owner_id": str(campaign.owner_id),
                    }
                )

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
