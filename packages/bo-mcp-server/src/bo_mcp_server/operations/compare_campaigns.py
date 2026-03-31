"""Shared campaign comparison operations."""

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
from bo_mcp_server.response_formatter import format_compare_campaigns_response
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)


def _compute_campaign_metrics(
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

    backend = get_backend()
    opt_spec = campaign_spec_to_optimization_spec(spec)
    observations = results_to_observations(results)
    hypervolume = backend.compute_hypervolume(opt_spec, observations)

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

    single_objective_campaigns = [
        (index, metrics)
        for index, metrics in enumerate(metrics_list)
        if not metrics["is_multi_objective"]
    ]
    best_single_objective = None
    if single_objective_campaigns:
        best_single_objective_idx = min(
            (
                index
                for index, metrics in single_objective_campaigns
                if metrics.get("best_value") is not None
            ),
            key=lambda index: metrics_list[index]["best_value"],
            default=None,
        )
        if best_single_objective_idx is not None:
            best_single_objective = campaign_names[best_single_objective_idx]

    multi_objective_campaigns = [
        (index, metrics)
        for index, metrics in enumerate(metrics_list)
        if metrics["is_multi_objective"]
    ]
    best_multi_objective = None
    if multi_objective_campaigns:
        best_multi_objective_idx = max(
            (
                index
                for index, metrics in multi_objective_campaigns
                if metrics.get("hypervolume") is not None
            ),
            key=lambda index: metrics_list[index]["hypervolume"],
            default=None,
        )
        if best_multi_objective_idx is not None:
            best_multi_objective = campaign_names[best_multi_objective_idx]

    return {
        "best_sample_efficiency": campaign_names[best_sample_efficiency_idx],
        "best_single_objective": best_single_objective,
        "best_multi_objective": best_multi_objective,
        "total_campaigns_compared": len(metrics_list),
        "recommendation": _generate_comparison_recommendation(metrics_list, campaign_names),
    }


async def compare_campaigns_operation(
    campaign_ids: list[str],
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Compare multiple campaigns."""
    logger.info("Comparing %d campaigns, verbosity=%s", len(campaign_ids), verbosity)

    verbosity_result = parse_verbosity(verbosity)
    if isinstance(verbosity_result, dict):
        verbosity_result.update({"campaigns": [], "comparison": None})
        return verbosity_result
    verbosity_level = verbosity_result

    if len(campaign_ids) < 2:
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="Need at least 2 campaigns to compare",
            details={"provided_count": len(campaign_ids)},
        )
        response.update({"campaigns": [], "comparison": None})
        return response

    if len(campaign_ids) > 10:
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="Cannot compare more than 10 campaigns at once",
            details={"provided_count": len(campaign_ids)},
        )
        response.update({"campaigns": [], "comparison": None})
        return response

    campaign_uuids: list[UUID] = []
    for campaign_id in campaign_ids:
        parsed_id = parse_campaign_id(campaign_id)
        if isinstance(parsed_id, dict):
            parsed_id.update({"campaigns": [], "comparison": None})
            return parsed_id
        campaign_uuids.append(parsed_id)

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)

        campaign_metrics: list[dict[str, Any]] = []
        campaign_names: list[str] = []
        errors: list[str] = []

        for campaign_uuid in campaign_uuids:
            campaign = await campaign_repo.get(campaign_uuid)
            if campaign is None:
                errors.append(f"Campaign {campaign_uuid} not found")
                continue

            spec = await spec_repo.get(campaign.spec_id)
            if spec is None:
                errors.append(f"Campaign spec for {campaign_uuid} not found")
                continue

            results = await result_repo.list_by_campaign(campaign_uuid)
            metrics = _compute_campaign_metrics(spec, results, campaign.iteration)
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
                "errors": errors + ["Need at least 2 valid campaigns to compare"],
            }

        comparison = _compare_metrics(campaign_metrics, campaign_names)
        full_response = {
            "success": True,
            "campaigns": campaign_metrics,
            "comparison": comparison,
            "errors": errors,
        }
        return format_compare_campaigns_response(full_response, verbosity_level)
