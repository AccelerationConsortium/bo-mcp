"""Discover transfer learning candidates tool for MCP.

Helps AI agents automatically discover relevant prior campaigns
for transfer learning.
"""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.domain import CampaignSpec, CampaignStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel, format_transfer_candidates_response
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)


def _compute_parameter_similarity(
    source_spec: CampaignSpec,
    target_spec: CampaignSpec,
) -> float:
    """Compute similarity between parameter spaces.

    Uses Jaccard-like similarity based on parameter names and types.

    Returns:
        Similarity score between 0 and 1
    """
    source_params = {(p.name, p.type.value) for p in source_spec.parameters}
    target_params = {(p.name, p.type.value) for p in target_spec.parameters}

    if not source_params or not target_params:
        return 0.0

    # Jaccard similarity on parameter (name, type) tuples
    intersection = len(source_params & target_params)
    union = len(source_params | target_params)

    return intersection / union if union > 0 else 0.0


def _compute_objective_similarity(
    source_spec: CampaignSpec,
    target_spec: CampaignSpec,
) -> float:
    """Compute similarity between objective configurations.

    Returns:
        Similarity score between 0 and 1
    """
    source_objs = {(o.name, o.direction) for o in source_spec.objectives}
    target_objs = {(o.name, o.direction) for o in target_spec.objectives}

    if not source_objs or not target_objs:
        return 0.0

    # Jaccard similarity on objective (name, direction) tuples
    intersection = len(source_objs & target_objs)
    union = len(source_objs | target_objs)

    return intersection / union if union > 0 else 0.0


def _compute_bounds_overlap(
    source_spec: CampaignSpec,
    target_spec: CampaignSpec,
) -> float:
    """Compute overlap between parameter bounds.

    Only considers parameters that exist in both specs.

    Returns:
        Average overlap score between 0 and 1
    """
    source_params = {p.name: p for p in source_spec.parameters}
    target_params = {p.name: p for p in target_spec.parameters}

    common_params = set(source_params.keys()) & set(target_params.keys())
    if not common_params:
        return 0.0

    overlaps = []
    for name in common_params:
        sp = source_params[name]
        tp = target_params[name]

        # Skip categorical parameters (no bounds)
        if sp.bounds is None or tp.bounds is None:
            continue

        # Compute 1D interval overlap
        s_lower, s_upper = sp.bounds
        t_lower, t_upper = tp.bounds

        overlap_lower = max(s_lower, t_lower)
        overlap_upper = min(s_upper, t_upper)

        if overlap_lower >= overlap_upper:
            overlaps.append(0.0)
        else:
            overlap_length = overlap_upper - overlap_lower
            union_length = max(s_upper, t_upper) - min(s_lower, t_lower)
            overlaps.append(overlap_length / union_length if union_length > 0 else 0.0)

    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def _compute_overall_similarity(
    source_spec: CampaignSpec,
    target_spec: CampaignSpec,
    n_results: int,
) -> tuple[float, dict[str, float]]:
    """Compute overall similarity score for transfer learning.

    Returns:
        Tuple of (overall_score, component_scores)
    """
    param_sim = _compute_parameter_similarity(source_spec, target_spec)
    obj_sim = _compute_objective_similarity(source_spec, target_spec)
    bounds_overlap = _compute_bounds_overlap(source_spec, target_spec)

    # Data richness factor (more data = more valuable for transfer)
    data_factor = min(1.0, n_results / 20)  # Cap at 20 results

    # Weighted combination
    # Parameters are most important, then objectives, then bounds
    weights = {"parameters": 0.4, "objectives": 0.3, "bounds": 0.2, "data": 0.1}

    overall = (
        weights["parameters"] * param_sim
        + weights["objectives"] * obj_sim
        + weights["bounds"] * bounds_overlap
        + weights["data"] * data_factor
    )

    return overall, {
        "parameter_similarity": round(param_sim, 4),
        "objective_similarity": round(obj_sim, 4),
        "bounds_overlap": round(bounds_overlap, 4),
        "data_richness": round(data_factor, 4),
    }


def _generate_transfer_recommendation(
    similarity: float,
    component_scores: dict[str, float],
    source_name: str,
    n_results: int,
) -> str:
    """Generate agent-friendly recommendation for transfer candidate."""
    if similarity >= 0.8:
        return (
            f"Highly recommended for transfer learning. '{source_name}' has "
            f"excellent similarity ({similarity:.0%}) and {n_results} results. "
            "Expected significant benefit from transfer."
        )
    elif similarity >= 0.6:
        return (
            f"Recommended for transfer learning. '{source_name}' has "
            f"good similarity ({similarity:.0%}). Transfer may accelerate "
            "early optimization but may have limited benefit later."
        )
    elif similarity >= 0.4:
        return (
            f"Possible candidate. '{source_name}' has moderate similarity "
            f"({similarity:.0%}). Consider transfer if few other options exist."
        )
    else:
        return (
            f"Low similarity ({similarity:.0%}). Transfer from '{source_name}' "
            "may not be beneficial and could potentially harm optimization."
        )


@mcp.tool()
async def discover_transfer_candidates(
    campaign_id: str,
    similarity_threshold: float = 0.5,
    max_candidates: int = 5,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Discover campaigns suitable for transfer learning.

    Analyzes other campaigns in the system to find those with similar
    parameter spaces and objectives that could accelerate optimization
    through transfer learning.

    Args:
        campaign_id: UUID of the target campaign to find transfer sources for
        similarity_threshold: Minimum similarity score (0-1) for candidates
        max_candidates: Maximum number of candidates to return
        verbosity: Response verbosity level. Options:
            - "minimal": ~50 tokens - success + top candidate + recommendation only
            - "standard": ~200 tokens - simplified candidate info
            - "detailed": ~500+ tokens - all fields including component scores

    Returns:
        Dictionary with:
            - success: Boolean indicating if discovery succeeded
            - target_campaign: Information about the target campaign
            - candidates: Ranked list of transfer candidates
            - errors: List of error messages (if failed)
    """
    logger.info(
        "Discovering transfer candidates for campaign %s (threshold=%.2f, verbosity=%s)",
        campaign_id,
        similarity_threshold,
        verbosity,
    )

    # Validate verbosity parameter
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
        logger.warning("Invalid campaign_id format: %s", campaign_id)
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

        # Get target campaign
        target_campaign = await campaign_repo.get(target_uuid)
        if target_campaign is None:
            return {
                "success": False,
                "target_campaign": None,
                "candidates": [],
                "errors": [f"Campaign {campaign_id} not found"],
            }

        # Get target spec
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
            "parameter_names": [p.name for p in target_spec.parameters],
            "objective_names": [o.name for o in target_spec.objectives],
        }

        # Get all campaigns to compare
        all_campaigns = await campaign_repo.list_all()

        candidates = []
        for candidate_campaign in all_campaigns:
            # Skip the target campaign itself
            if candidate_campaign.id == target_uuid:
                continue

            # Skip campaigns without results (nothing to transfer)
            candidate_results = await result_repo.list_by_campaign(candidate_campaign.id)
            if len(candidate_results) < 3:  # Need minimum data for meaningful transfer
                continue

            # Skip failed or very early campaigns
            if candidate_campaign.status == CampaignStatus.FAILED:
                continue

            # Get candidate spec
            candidate_spec = await spec_repo.get(candidate_campaign.spec_id)
            if candidate_spec is None:
                continue

            # Compute similarity
            overall_sim, component_scores = _compute_overall_similarity(
                candidate_spec, target_spec, len(candidate_results)
            )

            # Filter by threshold
            if overall_sim < similarity_threshold:
                continue

            candidates.append(
                {
                    "campaign_id": str(candidate_campaign.id),
                    "name": candidate_spec.name,
                    "status": candidate_campaign.status.value,
                    "n_results": len(candidate_results),
                    "iteration": candidate_campaign.iteration,
                    "similarity_score": round(overall_sim, 4),
                    "component_scores": component_scores,
                    "recommendation": _generate_transfer_recommendation(
                        overall_sim,
                        component_scores,
                        candidate_spec.name,
                        len(candidate_results),
                    ),
                }
            )

        # Sort by similarity score and limit
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        candidates = candidates[:max_candidates]

        # Generate overall recommendation
        if not candidates:
            overall_recommendation = (
                "No suitable transfer candidates found. The target campaign's parameter "
                "space and objectives are sufficiently different from existing campaigns. "
                "Optimization will proceed without transfer learning."
            )
        elif candidates[0]["similarity_score"] >= 0.7:
            top = candidates[0]
            overall_recommendation = (
                f"Recommend transferring from '{top['name']}' (similarity: "
                f"{top['similarity_score']:.0%}, {top['n_results']} results). "
                f'Use prior_campaign_ids: ["{top["campaign_id"]}"] when creating '
                "the campaign spec to enable transfer learning."
            )
        else:
            overall_recommendation = (
                "Moderate transfer candidates found. Transfer learning may provide "
                "some benefit but is not guaranteed. Consider running initial design "
                "without transfer and comparing performance."
            )

        logger.info(
            "Transfer candidate discovery completed: found %d candidates",
            len(candidates),
        )
        full_response = {
            "success": True,
            "target_campaign": target_info,
            "candidates": candidates,
            "overall_recommendation": overall_recommendation,
            "errors": [],
        }
        return format_transfer_candidates_response(full_response, verbosity_level)
