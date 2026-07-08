"""Shared campaign comparison operations.

The backend's ``compute_hypervolume`` call runs on a worker thread so it
never blocks the event loop while other requests are served.
"""

import asyncio
import logging
from typing import Any
from uuid import UUID

from bo_engine.diagnostics import (
    compute_best_value,
    compute_improvement_history,
    compute_single_objective_improvement_rate,
)
from bo_mcp_server.backend import get_backend_async
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import CampaignSpec, Result
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.helpers import (
    objective_analysis_series,
    objective_identity,
    parse_campaign_id,
    parse_verbosity,
    results_to_observations,
)
from bo_mcp_server.response_formatter import VerbosityLevel, format_compare_campaigns_response
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)

# Inclusive bounds for the number of campaigns that can be compared in a
# single call. Exposed so the MCP tool schema can advertise the limit via
# ``minItems`` / ``maxItems`` instead of letting agents discover it by
# failing.
MIN_COMPARE_CAMPAIGNS = 2
MAX_COMPARE_CAMPAIGNS = 10


# Labels for the per-campaign ``sample_efficiency_basis`` field. The two
# metrics live in different units (a dimensionless improvement ratio vs a
# hypervolume rate in objective units), so cross-basis ranking is
# meaningless — the comparison section only ranks campaigns sharing a basis
# (and, for multi-objective, the same objective structure).
_EFFICIENCY_BASIS_SINGLE = "relative_improvement_per_result"
_EFFICIENCY_BASIS_MULTI = "hypervolume_per_result"


def _objective_signature(spec: CampaignSpec) -> list[str]:
    """Order-independent objective identity used to group comparable campaigns.

    Built from :func:`objective_identity` (resolved mode + MATCH target),
    so legacy-``direction`` and ``target_mode`` spellings of the same goal
    group together while MATCH objectives with different targets do not.
    """
    return sorted(objective_identity(objective) for objective in spec.objectives)


async def _compute_campaign_metrics(
    spec: CampaignSpec,
    results: list[Result],
    iteration: int,
) -> dict[str, Any]:
    is_multi_objective = len(spec.objectives) > 1
    metrics: dict[str, Any] = {
        "n_results": len(results),
        "iteration": iteration,
        "n_parameters": len(spec.parameters),
        "n_objectives": len(spec.objectives),
        "is_multi_objective": is_multi_objective,
        "objective_signature": _objective_signature(spec),
        "sample_efficiency_basis": (
            _EFFICIENCY_BASIS_MULTI if is_multi_objective else _EFFICIENCY_BASIS_SINGLE
        ),
    }

    if not results:
        metrics.update(
            {
                "best_value": None,
                "improvement_rate": 0.0,
                "sample_efficiency": 0.0,
                "hypervolume": None,
                "n_pareto_points": None,
            }
        )
        return metrics

    if len(spec.objectives) == 1:
        objective = spec.objectives[0]
        values = [result.objective_values[objective.name] for result in results]

        # MATCH objectives are analyzed on distance-to-target (best =
        # closest); other goals pass raw values through unchanged. The
        # reported best_value is always the raw observed value at the
        # best metric index.
        metric_values, metric_minimize = objective_analysis_series(objective, values)
        best_metric, best_idx = compute_best_value(metric_values, minimize=metric_minimize)
        best_value = values[best_idx]
        improvement_history = compute_improvement_history(metric_values, minimize=metric_minimize)
        improvement_rate = compute_single_objective_improvement_rate(improvement_history)

        if len(metric_values) >= 2 and abs(metric_values[0]) > 1e-10:
            sample_efficiency = (
                abs(best_metric - metric_values[0]) / abs(metric_values[0]) / len(metric_values)
            )
        else:
            sample_efficiency = 0.0

        metrics.update(
            {
                "best_value": best_value,
                "improvement_rate": round(improvement_rate, 4),
                "sample_efficiency": round(sample_efficiency, 6),
                "hypervolume": None,
                "n_pareto_points": None,
            }
        )
        return metrics

    backend = await get_backend_async(spec.backend)
    opt_spec = campaign_spec_to_optimization_spec(spec)
    observations = results_to_observations(results)
    hypervolume = await asyncio.to_thread(backend.compute_hypervolume, opt_spec, observations)

    metrics.update(
        {
            "best_value": None,
            "improvement_rate": None,
            "sample_efficiency": (round(hypervolume / len(results), 6) if hypervolume else 0.0),
            "hypervolume": round(hypervolume, 6) if hypervolume else None,
            "n_pareto_points": None,  # Computed internally by backend
        }
    )
    return metrics


def _sample_efficiency_groups(metrics_list: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Group campaign indices into sample-efficiency comparability classes.

    Single-objective campaigns share one class — their metric is a
    dimensionless improvement ratio. Multi-objective campaigns are grouped
    by objective signature, because a raw hypervolume rate is only
    meaningful against campaigns optimizing the same objectives.
    """
    groups: dict[str, list[int]] = {}
    for index, metrics in enumerate(metrics_list):
        if metrics["is_multi_objective"]:
            signature = "|".join(metrics.get("objective_signature", []))
            key = f"multi_objective[{signature}]"
        else:
            key = "single_objective"
        groups.setdefault(key, []).append(index)
    return groups


def _generate_comparison_recommendation(
    metrics_list: list[dict[str, Any]],
    campaign_names: list[str],
    most_efficient_campaign: str | None,
) -> str:
    if len(metrics_list) < 2:
        return "Need at least 2 campaigns to generate meaningful comparison."

    most_data_idx = max(
        range(len(metrics_list)),
        key=lambda index: metrics_list[index]["n_results"],
    )
    most_data_campaign = campaign_names[most_data_idx]

    if most_efficient_campaign is None:
        return (
            f"Campaign '{most_data_campaign}' has the most data. The compared campaigns "
            "differ in objective structure, so their sample-efficiency values are in "
            "different units and cannot be ranked against each other — see "
            "best_sample_efficiency_by_group for the per-group leaders."
        )

    if most_data_campaign == most_efficient_campaign:
        return (
            f"Campaign '{most_data_campaign}' has both the most data and best sample "
            "efficiency. Consider using its configuration as a baseline for future campaigns."
        )

    return (
        f"Campaign '{most_efficient_campaign}' has the best sample efficiency, while "
        f"'{most_data_campaign}' has the most data. Consider transferring learnings "
        f"from '{most_efficient_campaign}' to improve future campaigns."
    )


def _compare_metrics(
    metrics_list: list[dict[str, Any]],
    campaign_names: list[str],
) -> dict[str, Any]:
    # Sample efficiency is ranked only within comparability groups: the
    # single-objective ratio and the multi-objective hypervolume rate are
    # incommensurable (the raw HV rate carries the objectives' units, so a
    # global max() would let multi-objective campaigns win on magnitude
    # alone). The flat best_sample_efficiency is only populated when every
    # compared campaign falls into one group.
    groups = _sample_efficiency_groups(metrics_list)
    efficiency_by_group = {
        key: campaign_names[
            max(indices, key=lambda index: metrics_list[index].get("sample_efficiency", 0) or 0)
        ]
        for key, indices in groups.items()
    }
    best_sample_efficiency = (
        next(iter(efficiency_by_group.values())) if len(efficiency_by_group) == 1 else None
    )

    # Determine best single-objective performer (lowest best_value).
    # Guard: the generator may be empty if no campaign has results yet.
    best_single_objective = None
    single_with_values = [
        (index, metrics)
        for index, metrics in enumerate(metrics_list)
        if not metrics["is_multi_objective"] and metrics.get("best_value") is not None
    ]
    if single_with_values:
        best_single_objective_idx = min(
            (index for index, _ in single_with_values),
            key=lambda index: metrics_list[index]["best_value"],
        )
        best_single_objective = campaign_names[best_single_objective_idx]

    # Determine best multi-objective performer (highest hypervolume) — only
    # when every multi-objective campaign optimizes the same objectives; a
    # raw hypervolume carries the objectives' units, so ranking across
    # different objective structures would be a pure magnitude artifact.
    best_multi_objective = None
    multi_with_hv = [
        (index, metrics)
        for index, metrics in enumerate(metrics_list)
        if metrics["is_multi_objective"] and metrics.get("hypervolume") is not None
    ]
    multi_signatures = {
        tuple(metrics.get("objective_signature", [])) for _, metrics in multi_with_hv
    }
    if multi_with_hv and len(multi_signatures) == 1:
        best_multi_objective_idx = max(
            (index for index, _ in multi_with_hv),
            key=lambda index: metrics_list[index]["hypervolume"],
        )
        best_multi_objective = campaign_names[best_multi_objective_idx]

    return {
        "best_sample_efficiency": best_sample_efficiency,
        "best_sample_efficiency_by_group": efficiency_by_group,
        "sample_efficiency_note": (
            "Sample efficiency is comparable only between campaigns sharing a basis: "
            f"single-objective campaigns use {_EFFICIENCY_BASIS_SINGLE} (dimensionless), "
            f"multi-objective campaigns use {_EFFICIENCY_BASIS_MULTI} (objective units, "
            "grouped by objective signature)."
        ),
        "best_single_objective": best_single_objective,
        "best_multi_objective": best_multi_objective,
        "total_campaigns_compared": len(metrics_list),
        "recommendation": _generate_comparison_recommendation(
            metrics_list, campaign_names, best_sample_efficiency
        ),
    }


def _make_compare_error(
    code: ErrorCode,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compare-specific error response."""
    response = make_error_response(code, message=message, details=details)
    response.update({"campaigns": [], "comparison": None})
    return response


def _validate_compare_inputs(
    campaign_ids: list[str],
    verbosity: str,
) -> tuple[VerbosityLevel, list[UUID]] | dict[str, Any]:
    """Validate compare_campaigns inputs. Returns (level, uuids) or error dict."""
    verbosity_result = parse_verbosity(verbosity)
    if isinstance(verbosity_result, dict):
        verbosity_result.update({"campaigns": [], "comparison": None})
        return verbosity_result

    if len(campaign_ids) < MIN_COMPARE_CAMPAIGNS:
        return _make_compare_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Need at least {MIN_COMPARE_CAMPAIGNS} campaigns to compare",
            details={
                "provided_count": len(campaign_ids),
                "min_campaigns": MIN_COMPARE_CAMPAIGNS,
            },
        )

    if len(campaign_ids) > MAX_COMPARE_CAMPAIGNS:
        return _make_compare_error(
            ErrorCode.VALIDATION_FAILED,
            message=(f"Cannot compare more than {MAX_COMPARE_CAMPAIGNS} campaigns at once"),
            details={
                "provided_count": len(campaign_ids),
                "max_campaigns": MAX_COMPARE_CAMPAIGNS,
            },
        )

    campaign_uuids: list[UUID] = []
    for campaign_id in campaign_ids:
        parsed_id = parse_campaign_id(campaign_id)
        if isinstance(parsed_id, dict):
            parsed_id.update({"campaigns": [], "comparison": None})
            return parsed_id
        campaign_uuids.append(parsed_id)

    return verbosity_result, campaign_uuids


async def compare_campaigns_operation(
    campaign_ids: list[str],
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Compare multiple campaigns."""
    logger.info("Comparing %d campaigns, verbosity=%s", len(campaign_ids), verbosity)

    validated = _validate_compare_inputs(campaign_ids, verbosity)
    if isinstance(validated, dict):
        return validated
    verbosity_level, campaign_uuids = validated

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)

        campaign_metrics: list[dict[str, Any]] = []
        campaign_names: list[str] = []
        errors: list[str] = []

        # Batch-fetch all campaigns, specs, and results to avoid N+1 queries
        campaigns_by_id = await campaign_repo.get_by_ids(campaign_uuids)
        found_campaigns = {}
        for campaign_uuid in campaign_uuids:
            if campaign_uuid not in campaigns_by_id:
                errors.append(f"Campaign {campaign_uuid} not found")
            else:
                found_campaigns[campaign_uuid] = campaigns_by_id[campaign_uuid]

        spec_ids = list({c.spec_id for c in found_campaigns.values()})
        specs_by_id = await spec_repo.get_by_ids(spec_ids)
        results_by_campaign = await result_repo.list_by_campaigns(list(found_campaigns.keys()))

        for campaign_uuid, campaign in found_campaigns.items():
            spec = specs_by_id.get(campaign.spec_id)
            if spec is None:
                errors.append(f"Campaign spec for {campaign_uuid} not found")
                continue

            results = results_by_campaign.get(campaign_uuid, [])
            metrics = await _compute_campaign_metrics(spec, results, campaign.iteration)
            metrics["campaign_id"] = str(campaign_uuid)
            metrics["campaign_name"] = spec.name
            metrics["status"] = campaign.status.value

            campaign_metrics.append(metrics)
            campaign_names.append(spec.name)

        if len(campaign_metrics) < 2:
            return {
                "success": False,
                "campaigns": campaign_metrics,
                "comparison": None,
                "errors": [*errors, "Need at least 2 valid campaigns to compare"],
            }

        comparison = _compare_metrics(campaign_metrics, campaign_names)
        full_response = {
            "success": True,
            "campaigns": campaign_metrics,
            "comparison": comparison,
            "errors": errors,
        }
        return format_compare_campaigns_response(full_response, verbosity_level)
