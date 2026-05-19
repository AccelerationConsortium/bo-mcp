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

from bo_mcp_server.backend import get_backend
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import CampaignSpec, Result
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.helpers import (
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


async def _compute_campaign_metrics(
    spec: CampaignSpec,
    results: list[Result],
    iteration: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "n_results": len(results),
        "iteration": iteration,
        "n_parameters": len(spec.parameters),
        "n_objectives": len(spec.objectives),
        "is_multi_objective": len(spec.objectives) > 1,
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

        best_value, _ = compute_best_value(values, minimize=objective.is_minimize)
        improvement_history = compute_improvement_history(values, minimize=objective.is_minimize)
        improvement_rate = compute_single_objective_improvement_rate(improvement_history)

        if len(values) >= 2 and abs(values[0]) > 1e-10:
            sample_efficiency = abs(best_value - values[0]) / abs(values[0]) / len(values)
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

    backend = get_backend(spec.backend)
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


def _generate_comparison_recommendation(
    metrics_list: list[dict[str, Any]],
    campaign_names: list[str],
) -> str:
    if len(metrics_list) < 2:
        return "Need at least 2 campaigns to generate meaningful comparison."

    most_data_idx = max(
        range(len(metrics_list)),
        key=lambda index: metrics_list[index]["n_results"],
    )
    most_data_campaign = campaign_names[most_data_idx]

    most_efficient_idx = max(
        range(len(metrics_list)),
        key=lambda index: metrics_list[index].get("sample_efficiency", 0) or 0,
    )
    most_efficient_campaign = campaign_names[most_efficient_idx]

    if most_data_idx == most_efficient_idx:
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
    best_sample_efficiency_idx = max(
        range(len(metrics_list)),
        key=lambda index: metrics_list[index].get("sample_efficiency", 0) or 0,
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

    # Determine best multi-objective performer (highest hypervolume).
    best_multi_objective = None
    multi_with_hv = [
        (index, metrics)
        for index, metrics in enumerate(metrics_list)
        if metrics["is_multi_objective"] and metrics.get("hypervolume") is not None
    ]
    if multi_with_hv:
        best_multi_objective_idx = max(
            (index for index, _ in multi_with_hv),
            key=lambda index: metrics_list[index]["hypervolume"],
        )
        best_multi_objective = campaign_names[best_multi_objective_idx]

    return {
        "best_sample_efficiency": campaign_names[best_sample_efficiency_idx],
        "best_single_objective": best_single_objective,
        "best_multi_objective": best_multi_objective,
        "total_campaigns_compared": len(metrics_list),
        "recommendation": _generate_comparison_recommendation(metrics_list, campaign_names),
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
