"""Shared batch status operations."""

import logging
import math
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import Campaign, CampaignSpec, CampaignStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.helpers import parse_verbosity
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


def _determine_health(status: CampaignStatus, n_results: int, iteration: int) -> str:
    if status == CampaignStatus.FAILED:
        return "critical"
    if status == CampaignStatus.PAUSED:
        return "paused"
    if n_results == 0 and iteration > 1:
        return "warning"
    return "healthy"


def _compute_key_metric(
    spec: CampaignSpec | None, hypervolume_history: list[float]
) -> dict[str, Any]:
    if spec and len(spec.objectives) >= 2 and hypervolume_history:
        return {"hypervolume": hypervolume_history[-1]}
    return {}


def _compute_convergence(hypervolume_history: list[float]) -> dict[str, Any]:
    convergence_info: dict[str, Any] = {"converged": False}
    if hypervolume_history and len(hypervolume_history) >= 5:
        recent = hypervolume_history[-5:]
        if (
            all(math.isclose(r, recent[0], rel_tol=1e-9) for r in recent)
            or (max(recent) - min(recent)) < 0.001
        ):
            convergence_info["converged"] = True
            convergence_info["reason"] = "Hypervolume stable"
    return convergence_info


def _build_minimal_info(name: str, campaign: Campaign, n_results: int) -> dict[str, Any]:
    return {
        "name": name,
        "status": campaign.status.value,
        "iteration": campaign.iteration,
        "n_results": n_results,
    }


def _build_standard_info(
    name: str,
    campaign: Campaign,
    n_results: int,
    n_pending: int,
    spec: CampaignSpec | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": campaign.status.value,
        "iteration": campaign.iteration,
        "n_results": n_results,
        "n_pending_suggestions": n_pending,
        "health": _determine_health(campaign.status, n_results, campaign.iteration),
        "key_metric": _compute_key_metric(spec, campaign.hypervolume_history),
    }


def _build_detailed_info(
    name: str,
    campaign: Campaign,
    n_results: int,
    n_pending: int,
    spec: CampaignSpec | None,
) -> dict[str, Any]:
    info = _build_standard_info(name, campaign, n_results, n_pending, spec)
    info.update(
        {
            "convergence": _compute_convergence(campaign.hypervolume_history),
            "created_at": campaign.created_at.isoformat(),
            "owner_id": str(campaign.owner_id),
        }
    )
    return info


def _parse_campaign_ids(
    campaign_ids: list[str],
) -> tuple[list[tuple[str, UUID]], list[str]]:
    """Parse and validate campaign ID strings into UUIDs."""
    valid: list[tuple[str, UUID]] = []
    invalid: list[str] = []
    for cid in campaign_ids:
        try:
            valid.append((cid, UUID(cid)))
        except ValueError:
            invalid.append(cid)
    return valid, invalid


def _validate_batch_request(
    campaign_ids: list[str], verbosity: str
) -> dict[str, Any] | VerbosityLevel:
    """Validate batch request parameters. Returns error dict or VerbosityLevel."""
    verbosity_result = parse_verbosity(verbosity)
    if isinstance(verbosity_result, dict):
        return verbosity_result
    verbosity_level = verbosity_result

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

    return verbosity_level


def _build_campaign_info(
    verbosity_level: VerbosityLevel,
    name: str,
    campaign: Campaign,
    n_results: int,
    n_pending: int,
    spec: CampaignSpec | None,
) -> dict[str, Any]:
    """Build campaign info dict based on verbosity."""
    if verbosity_level == VerbosityLevel.MINIMAL:
        return _build_minimal_info(name, campaign, n_results)
    if verbosity_level == VerbosityLevel.STANDARD:
        return _build_standard_info(name, campaign, n_results, n_pending, spec)
    return _build_detailed_info(name, campaign, n_results, n_pending, spec)


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

    validated = _validate_batch_request(campaign_ids, verbosity)
    if isinstance(validated, dict):
        return validated
    verbosity_level = validated

    valid_uuids, invalid_ids = _parse_campaign_ids(campaign_ids)
    errors: list[str] = []
    if invalid_ids:
        errors.append(f"Invalid UUID format: {invalid_ids}")

    failed_ids: list[str] = list(invalid_ids)
    campaigns_info: dict[str, dict[str, Any]] = {}

    if not valid_uuids:
        return {
            "success": False,
            "campaigns": campaigns_info,
            "failed_ids": failed_ids,
            "errors": errors,
        }

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)
        suggestion_repo = SuggestionRepository(session)

        # Batch-fetch all data
        id_str_map = {uuid: id_str for id_str, uuid in valid_uuids}
        uuid_list = list(id_str_map.keys())
        campaigns = await campaign_repo.get_by_ids(uuid_list)

        for uuid in uuid_list:
            if uuid not in campaigns:
                failed_ids.append(id_str_map[uuid])

        found_uuids = list(campaigns.keys())
        spec_ids = list({c.spec_id for c in campaigns.values()})
        specs = await spec_repo.get_by_ids(spec_ids)
        result_counts = await result_repo.count_by_campaigns(found_uuids)

        pending_counts: dict[UUID, int] = {}
        if verbosity_level != VerbosityLevel.MINIMAL:
            pending_counts = await suggestion_repo.count_pending_by_campaigns(found_uuids)

        for campaign_uuid, campaign in campaigns.items():
            cid = id_str_map[campaign_uuid]
            spec = specs.get(campaign.spec_id)
            name = spec.name if spec else "Unknown"
            campaigns_info[cid] = _build_campaign_info(
                verbosity_level,
                name,
                campaign,
                result_counts.get(campaign_uuid, 0),
                pending_counts.get(campaign_uuid, 0),
                spec,
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
