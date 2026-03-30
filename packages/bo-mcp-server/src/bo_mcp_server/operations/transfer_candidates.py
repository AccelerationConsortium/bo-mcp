"""Shared transfer candidate discovery operations."""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import CampaignSpec, CampaignStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel, format_transfer_candidates_response
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)


def _compute_parameter_similarity(source_spec: CampaignSpec, target_spec: CampaignSpec) -> float:
    source_params = {(parameter.name, parameter.type.value) for parameter in source_spec.parameters}
    target_params = {(parameter.name, parameter.type.value) for parameter in target_spec.parameters}
    if not source_params or not target_params:
        return 0.0
    return len(source_params & target_params) / len(source_params | target_params)


def _compute_objective_similarity(source_spec: CampaignSpec, target_spec: CampaignSpec) -> float:
    source_objectives = {
        (objective.name, objective.direction) for objective in source_spec.objectives
    }
    target_objectives = {
        (objective.name, objective.direction) for objective in target_spec.objectives
    }
    if not source_objectives or not target_objectives:
        return 0.0
    return len(source_objectives & target_objectives) / len(source_objectives | target_objectives)


def _compute_bounds_overlap(source_spec: CampaignSpec, target_spec: CampaignSpec) -> float:
    source_params = {parameter.name: parameter for parameter in source_spec.parameters}
    target_params = {parameter.name: parameter for parameter in target_spec.parameters}

    common_params = set(source_params) & set(target_params)
    if not common_params:
        return 0.0

    overlaps: list[float] = []
    for name in common_params:
        source_param = source_params[name]
        target_param = target_params[name]
        if source_param.bounds is None or target_param.bounds is None:
            continue

        source_lower, source_upper = source_param.bounds.lower, source_param.bounds.upper
        target_lower, target_upper = target_param.bounds.lower, target_param.bounds.upper

        overlap_lower = max(source_lower, target_lower)
        overlap_upper = min(source_upper, target_upper)
        if overlap_lower >= overlap_upper:
            overlaps.append(0.0)
            continue

        overlap_length = overlap_upper - overlap_lower
        union_length = max(source_upper, target_upper) - min(source_lower, target_lower)
        overlaps.append(overlap_length / union_length if union_length > 0 else 0.0)

    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def _compute_overall_similarity(
    source_spec: CampaignSpec,
    target_spec: CampaignSpec,
    n_results: int,
) -> tuple[float, dict[str, float]]:
    parameter_similarity = _compute_parameter_similarity(source_spec, target_spec)
    objective_similarity = _compute_objective_similarity(source_spec, target_spec)
    bounds_overlap = _compute_bounds_overlap(source_spec, target_spec)
    data_richness = min(1.0, n_results / 20)

    overall = (
        0.4 * parameter_similarity
        + 0.3 * objective_similarity
        + 0.2 * bounds_overlap
        + 0.1 * data_richness
    )

    return overall, {
        "parameter_similarity": round(parameter_similarity, 4),
        "objective_similarity": round(objective_similarity, 4),
        "bounds_overlap": round(bounds_overlap, 4),
        "data_richness": round(data_richness, 4),
    }


def _generate_transfer_recommendation(
    similarity: float,
    source_name: str,
    n_results: int,
) -> str:
    if similarity >= 0.8:
        return (
            f"Highly recommended for transfer learning. '{source_name}' has "
            f"excellent similarity ({similarity:.0%}) and {n_results} results. "
            "Expected significant benefit from transfer."
        )
    if similarity >= 0.6:
        return (
            f"Recommended for transfer learning. '{source_name}' has "
            f"good similarity ({similarity:.0%}). Transfer may accelerate "
            "early optimization but may have limited benefit later."
        )
    if similarity >= 0.4:
        return (
            f"Possible candidate. '{source_name}' has moderate similarity "
            f"({similarity:.0%}). Consider transfer if few other options exist."
        )
    return (
        f"Low similarity ({similarity:.0%}). Transfer from '{source_name}' "
        "may not be beneficial and could potentially harm optimization."
    )


async def discover_transfer_candidates_operation(  # noqa: C901
    campaign_id: str,
    similarity_threshold: float = 0.5,
    max_candidates: int = 5,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Discover campaigns suitable for transfer learning."""
    logger.info(
        "Discovering transfer candidates for campaign %s (threshold=%.2f, verbosity=%s)",
        campaign_id,
        similarity_threshold,
        verbosity,
    )

    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )
        response.update({"target_campaign": None, "candidates": []})
        return response

    try:
        target_uuid = UUID(campaign_id)
    except ValueError:
        response = make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )
        response.update({"target_campaign": None, "candidates": []})
        return response

    if not (0.0 <= similarity_threshold <= 1.0):
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="similarity_threshold must be between 0 and 1",
            details={"similarity_threshold": similarity_threshold},
        )
        response.update({"target_campaign": None, "candidates": []})
        return response

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)

        target_campaign = await campaign_repo.get(target_uuid)
        if target_campaign is None:
            return {
                "success": False,
                "target_campaign": None,
                "candidates": [],
                "errors": [f"Campaign {campaign_id} not found"],
            }

        target_spec = await spec_repo.get(target_campaign.spec_id)
        if target_spec is None:
            return {
                "success": False,
                "target_campaign": None,
                "candidates": [],
                "errors": ["Campaign spec not found"],
            }

        target_info = {
            "campaign_id": str(target_uuid),
            "name": target_spec.name,
            "n_parameters": len(target_spec.parameters),
            "n_objectives": len(target_spec.objectives),
            "parameter_names": [parameter.name for parameter in target_spec.parameters],
            "objective_names": [objective.name for objective in target_spec.objectives],
        }

        candidates: list[dict[str, Any]] = []
        for candidate_campaign in await campaign_repo.list_all():
            if candidate_campaign.id == target_uuid:
                continue

            candidate_results = await result_repo.list_by_campaign(candidate_campaign.id)
            if len(candidate_results) < 3 or candidate_campaign.status == CampaignStatus.FAILED:
                continue

            candidate_spec = await spec_repo.get(candidate_campaign.spec_id)
            if candidate_spec is None:
                continue

            overall_similarity, component_scores = _compute_overall_similarity(
                candidate_spec,
                target_spec,
                len(candidate_results),
            )
            if overall_similarity < similarity_threshold:
                continue

            candidates.append(
                {
                    "campaign_id": str(candidate_campaign.id),
                    "name": candidate_spec.name,
                    "status": candidate_campaign.status.value,
                    "n_results": len(candidate_results),
                    "iteration": candidate_campaign.iteration,
                    "similarity_score": round(overall_similarity, 4),
                    "component_scores": component_scores,
                    "recommendation": _generate_transfer_recommendation(
                        overall_similarity,
                        candidate_spec.name,
                        len(candidate_results),
                    ),
                }
            )

        candidates.sort(key=lambda candidate: candidate["similarity_score"], reverse=True)
        candidates = candidates[:max_candidates]

        if not candidates:
            overall_recommendation = (
                "No suitable transfer candidates found. The target campaign's parameter "
                "space and objectives are sufficiently different from existing campaigns. "
                "Optimization will proceed without transfer learning."
            )
        elif candidates[0]["similarity_score"] >= 0.7:
            top_candidate = candidates[0]
            overall_recommendation = (
                f"Recommend transferring from '{top_candidate['name']}' (similarity: "
                f"{top_candidate['similarity_score']:.0%}, {top_candidate['n_results']} results). "
                f'Use prior_campaign_ids: ["{top_candidate["campaign_id"]}"] when creating '
                "the campaign spec to enable transfer learning."
            )
        else:
            overall_recommendation = (
                "Moderate transfer candidates found. Transfer learning may provide "
                "some benefit but is not guaranteed. Consider running initial design "
                "without transfer and comparing performance."
            )

        full_response = {
            "success": True,
            "target_campaign": target_info,
            "candidates": candidates,
            "overall_recommendation": overall_recommendation,
            "errors": [],
        }
        return format_transfer_candidates_response(full_response, verbosity_level)
