"""Shared transfer candidate discovery operations."""

import logging
from typing import Any
from uuid import UUID

from bo_engine.constants import (
    INITIAL_DESIGN_MULTIPLIER,
    TRANSFER_WEIGHT_BOUNDS,
    TRANSFER_WEIGHT_DATA_RICHNESS,
    TRANSFER_WEIGHT_OBJECTIVE,
    TRANSFER_WEIGHT_PARAMETER,
)
from bo_mcp_server.constants import (
    TRANSFER_SIMILARITY_GOOD,
    TRANSFER_SIMILARITY_MODERATE,
    TRANSFER_SIMILARITY_STRONG,
)
from bo_mcp_server.domain import CampaignSpec, CampaignStatus
from bo_mcp_server.domain.campaign_spec import InputParameter
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.helpers import (
    objective_identity,
    parse_campaign_id,
    parse_verbosity,
)
from bo_mcp_server.response_formatter import VerbosityLevel, format_transfer_candidates_response
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)


def _build_alias_index(
    parameter_aliases: dict[str, list[str]] | None,
) -> dict[str, str]:
    """Flatten ``{canonical: [synonym, ...]}`` into a synonym → canonical map.

    Includes the canonical name itself as a self-mapping so the lookup is
    total: both ``"temperature"`` and ``"temp_c"`` resolve to ``"temperature"``
    when the user supplies ``{"temperature": ["temp_c"]}``.
    """
    if not parameter_aliases:
        return {}
    flat: dict[str, str] = {}
    for canonical, synonyms in parameter_aliases.items():
        flat[canonical] = canonical
        for synonym in synonyms:
            flat[synonym] = canonical
    return flat


def _canonical_name(name: str, alias_index: dict[str, str]) -> str:
    """Return the canonical parameter name (alias-aware, falling back to ``name``)."""
    return alias_index.get(name, name)


def _compute_parameter_similarity(
    source_spec: CampaignSpec,
    target_spec: CampaignSpec,
    alias_index: dict[str, str],
) -> float:
    source_params = {
        (_canonical_name(parameter.name, alias_index), parameter.type.value)
        for parameter in source_spec.parameters
    }
    target_params = {
        (_canonical_name(parameter.name, alias_index), parameter.type.value)
        for parameter in target_spec.parameters
    }
    if not source_params or not target_params:
        return 0.0
    return len(source_params & target_params) / len(source_params | target_params)


def _compute_objective_similarity(source_spec: CampaignSpec, target_spec: CampaignSpec) -> float:
    """Jaccard similarity of the campaigns' objective-goal identity sets.

    Identities come from :func:`objective_identity` (resolved mode + MATCH
    target), so legacy-``direction`` and ``target_mode`` spellings of the
    same goal match while MATCH objectives with different targets do not.
    """
    source_objectives = {objective_identity(objective) for objective in source_spec.objectives}
    target_objectives = {objective_identity(objective) for objective in target_spec.objectives}
    if not source_objectives or not target_objectives:
        return 0.0
    return len(source_objectives & target_objectives) / len(source_objectives | target_objectives)


def _parameter_pair_overlap(
    source_param: InputParameter, target_param: InputParameter
) -> float | None:
    """Search-space overlap for one common parameter, or ``None`` if unscorable.

    Numeric parameters score interval-overlap / interval-union; categorical
    parameters score the Jaccard similarity of their category sets — the
    natural analogue of bounds overlap for a discrete label space.
    """
    if source_param.bounds is not None and target_param.bounds is not None:
        source_lower, source_upper = source_param.bounds.lower, source_param.bounds.upper
        target_lower, target_upper = target_param.bounds.lower, target_param.bounds.upper

        overlap_lower = max(source_lower, target_lower)
        overlap_upper = min(source_upper, target_upper)
        if overlap_lower >= overlap_upper:
            return 0.0
        overlap_length = overlap_upper - overlap_lower
        union_length = max(source_upper, target_upper) - min(source_lower, target_lower)
        return overlap_length / union_length if union_length > 0 else 0.0

    if source_param.categories is not None and target_param.categories is not None:
        source_categories = set(source_param.categories)
        target_categories = set(target_param.categories)
        union = source_categories | target_categories
        if not union:
            return None
        return len(source_categories & target_categories) / len(union)

    return None


def _compute_bounds_overlap(
    source_spec: CampaignSpec,
    target_spec: CampaignSpec,
    alias_index: dict[str, str],
) -> float | None:
    """Mean per-parameter search-space overlap over the common parameters.

    Returns ``None`` when no common parameter is scorable (e.g. no common
    parameters at all) so the caller can renormalize the component weights
    instead of treating "not measurable" as zero overlap — which would
    structurally cap categorical-only campaigns below the recommendation
    thresholds even for identical specs.
    """
    source_params = {
        _canonical_name(parameter.name, alias_index): parameter
        for parameter in source_spec.parameters
    }
    target_params = {
        _canonical_name(parameter.name, alias_index): parameter
        for parameter in target_spec.parameters
    }

    common_params = set(source_params) & set(target_params)
    overlaps = [
        overlap
        for name in common_params
        if (overlap := _parameter_pair_overlap(source_params[name], target_params[name]))
        is not None
    ]
    return sum(overlaps) / len(overlaps) if overlaps else None


def _compute_overall_similarity(
    source_spec: CampaignSpec,
    target_spec: CampaignSpec,
    n_results: int,
    alias_index: dict[str, str],
) -> tuple[float, dict[str, float | None]]:
    parameter_similarity = _compute_parameter_similarity(source_spec, target_spec, alias_index)
    objective_similarity = _compute_objective_similarity(source_spec, target_spec)
    bounds_overlap = _compute_bounds_overlap(source_spec, target_spec, alias_index)

    # Scale data richness threshold with source problem dimensionality:
    # more parameters need more data to be considered "rich"
    n_params = len(source_spec.parameters)
    richness_threshold = INITIAL_DESIGN_MULTIPLIER * n_params + 1
    data_richness = min(1.0, n_results / richness_threshold)

    weighted_components = [
        (TRANSFER_WEIGHT_PARAMETER, parameter_similarity),
        (TRANSFER_WEIGHT_OBJECTIVE, objective_similarity),
        (TRANSFER_WEIGHT_DATA_RICHNESS, data_richness),
    ]
    if bounds_overlap is not None:
        weighted_components.append((TRANSFER_WEIGHT_BOUNDS, bounds_overlap))

    # Renormalize over the available components so an unscorable overlap
    # ("no comparable search-space information") does not act as a hidden
    # zero: two identical specs must be able to reach similarity 1.0.
    total_weight = sum(weight for weight, _ in weighted_components)
    overall = sum(weight * value for weight, value in weighted_components) / total_weight

    return overall, {
        "parameter_similarity": round(parameter_similarity, 4),
        "objective_similarity": round(objective_similarity, 4),
        "bounds_overlap": round(bounds_overlap, 4) if bounds_overlap is not None else None,
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
    if similarity >= TRANSFER_SIMILARITY_GOOD:
        return (
            f"Recommended for transfer learning. '{source_name}' has "
            f"good similarity ({similarity:.0%}). Transfer may accelerate "
            "early optimization but may have limited benefit later."
        )
    if similarity >= TRANSFER_SIMILARITY_MODERATE:
        return (
            f"Possible candidate. '{source_name}' has moderate similarity "
            f"({similarity:.0%}). Consider transfer if few other options exist."
        )
    return (
        f"Low similarity ({similarity:.0%}). Transfer from '{source_name}' "
        "may not be beneficial and could potentially harm optimization."
    )


def _validate_transfer_inputs(
    campaign_id: str,
    similarity_threshold: float,
    verbosity: str,
) -> tuple[VerbosityLevel, UUID] | dict[str, Any]:
    """Validate inputs for transfer candidate discovery.

    Returns (verbosity_level, target_uuid) on success, or an error response dict.
    """
    verbosity_result = parse_verbosity(verbosity)
    if isinstance(verbosity_result, dict):
        verbosity_result.update({"target_campaign": None, "candidates": []})
        return verbosity_result
    verbosity_level = verbosity_result

    campaign_id_result = parse_campaign_id(campaign_id)
    if isinstance(campaign_id_result, dict):
        campaign_id_result.update({"target_campaign": None, "candidates": []})
        return campaign_id_result
    target_uuid = campaign_id_result

    if not (0.0 <= similarity_threshold <= 1.0):
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="similarity_threshold must be between 0 and 1",
            details={"similarity_threshold": similarity_threshold},
        )
        response.update({"target_campaign": None, "candidates": []})
        return response

    return verbosity_level, target_uuid


async def _evaluate_candidates(
    campaign_repo: CampaignRepository,
    spec_repo: CampaignSpecRepository,
    result_repo: ResultRepository,
    target_uuid: UUID,
    target_spec: CampaignSpec,
    similarity_threshold: float,
    max_candidates: int,
    alias_index: dict[str, str],
) -> list[dict[str, Any]]:
    """Evaluate all campaigns for transfer learning similarity.

    Uses batch queries to avoid N+1: loads all campaigns, then batch-loads
    specs and result counts in two queries instead of per-campaign queries.
    """
    all_campaigns = await campaign_repo.list_all()

    # Filter out the target campaign and failed campaigns with too few results
    candidate_campaigns = [
        c for c in all_campaigns if c.id != target_uuid and c.status != CampaignStatus.FAILED
    ]
    if not candidate_campaigns:
        return []

    # Batch-load specs and result counts (2 queries instead of 2*N)
    campaign_ids = [c.id for c in candidate_campaigns]
    spec_ids = list({c.spec_id for c in candidate_campaigns})
    specs_by_id = await spec_repo.get_by_ids(spec_ids)
    result_counts = await result_repo.count_by_campaigns(campaign_ids)

    candidates: list[dict[str, Any]] = []
    for campaign in candidate_campaigns:
        n_results = result_counts.get(campaign.id, 0)
        if n_results < 3:
            continue

        candidate_spec = specs_by_id.get(campaign.spec_id)
        if candidate_spec is None:
            continue

        overall_similarity, component_scores = _compute_overall_similarity(
            candidate_spec,
            target_spec,
            n_results,
            alias_index,
        )
        if overall_similarity < similarity_threshold:
            continue

        candidates.append(
            {
                "campaign_id": str(campaign.id),
                "name": candidate_spec.name,
                "status": campaign.status.value,
                "n_results": n_results,
                "iteration": campaign.iteration,
                "similarity_score": round(overall_similarity, 4),
                "component_scores": component_scores,
                "recommendation": _generate_transfer_recommendation(
                    overall_similarity,
                    candidate_spec.name,
                    n_results,
                ),
            }
        )

    candidates.sort(key=lambda c: c["similarity_score"], reverse=True)
    return candidates[:max_candidates]


def _build_overall_recommendation(candidates: list[dict[str, Any]]) -> str:
    """Build the overall transfer learning recommendation string."""
    if not candidates:
        return (
            "No suitable transfer candidates found. The target campaign's parameter "
            "space and objectives are sufficiently different from existing campaigns. "
            "Optimization will proceed without transfer learning."
        )
    if candidates[0]["similarity_score"] >= TRANSFER_SIMILARITY_STRONG:
        top = candidates[0]
        return (
            f"Recommend transferring from '{top['name']}' (similarity: "
            f"{top['similarity_score']:.0%}, {top['n_results']} results). "
            f'Use prior_campaign_ids: ["{top["campaign_id"]}"] when creating '
            "the campaign spec to enable transfer learning."
        )
    return (
        "Moderate transfer candidates found. Transfer learning may provide "
        "some benefit but is not guaranteed. Consider running initial design "
        "without transfer and comparing performance."
    )


async def discover_transfer_candidates_operation(
    campaign_id: str,
    similarity_threshold: float = 0.5,
    max_candidates: int = 5,
    verbosity: str = "standard",
    parameter_aliases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Discover campaigns suitable for transfer learning.

    ``parameter_aliases`` lets callers bridge parameter-name drift across
    related campaigns. The mapping uses ``{canonical: [synonym, ...]}``
    semantics so e.g. ``{"temperature": ["temp_c", "temp_celsius"]}``
    treats all three names as the same physical parameter when computing
    parameter-set overlap and bounds overlap.
    """
    logger.info(
        "Discovering transfer candidates for campaign %s (threshold=%.2f, verbosity=%s)",
        campaign_id,
        similarity_threshold,
        verbosity,
    )

    validated = _validate_transfer_inputs(campaign_id, similarity_threshold, verbosity)
    if isinstance(validated, dict):
        return validated
    verbosity_level, target_uuid = validated

    alias_index = _build_alias_index(parameter_aliases)

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
            "parameter_names": [p.name for p in target_spec.parameters],
            "objective_names": [o.name for o in target_spec.objectives],
        }

        candidates = await _evaluate_candidates(
            campaign_repo,
            spec_repo,
            result_repo,
            target_uuid,
            target_spec,
            similarity_threshold,
            max_candidates,
            alias_index,
        )

        full_response = {
            "success": True,
            "target_campaign": target_info,
            "candidates": candidates,
            "overall_recommendation": _build_overall_recommendation(candidates),
            "errors": [],
        }
        return format_transfer_candidates_response(full_response, verbosity_level)
