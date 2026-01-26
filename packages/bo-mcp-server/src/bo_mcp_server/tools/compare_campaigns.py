"""Compare campaigns tool for MCP.

Provides campaign comparison functionality to help AI agents understand
relative performance across different optimization campaigns.
"""

import logging
from typing import Any
from uuid import UUID

import torch
from bo_engine.diagnostics import (
    compute_best_value,
    compute_hypervolume,
    compute_improvement_history,
    compute_pareto_front,
    compute_single_objective_improvement_rate,
)

from bo_mcp_server.domain import CampaignSpec, Result
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel, format_compare_campaigns_response
from bo_mcp_server.server import mcp
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
    """Compute comparison metrics for a single campaign.

    Returns:
        Dictionary with normalized metrics for comparison
    """
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

    is_single_objective = len(spec.objectives) == 1

    if is_single_objective:
        obj = spec.objectives[0]
        values = [r.objective_values[obj.name] for r in results]

        best_value, _ = compute_best_value(values, minimize=obj.is_minimize)
        improvement_history = compute_improvement_history(values, minimize=obj.is_minimize)
        improvement_rate = compute_single_objective_improvement_rate(improvement_history)

        # Sample efficiency: improvement per result
        if len(values) >= 2:
            initial = values[0]
            final = best_value
            if abs(initial) > 1e-10:
                total_improvement = abs(final - initial) / abs(initial)
                sample_efficiency = total_improvement / len(values)
            else:
                sample_efficiency = 0.0
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
    else:
        # Multi-objective metrics
        objective_names = [o.name for o in spec.objectives]
        minimize_mask = torch.tensor([o.is_minimize for o in spec.objectives], dtype=torch.bool)

        y_list = []
        for r in results:
            y = torch.tensor(
                [r.objective_values[name] for name in objective_names],
                dtype=torch.double,
            )
            y_list.append(y)

        y_tensor = torch.stack(y_list)

        # Negate maximization objectives
        y_bo = y_tensor.clone()
        y_bo[:, ~minimize_mask] = -y_bo[:, ~minimize_mask]

        # Compute Pareto front
        pareto_y, _ = compute_pareto_front(y_bo)

        # Compute hypervolume
        worst = y_bo.max(dim=0).values
        ranges = y_bo.max(dim=0).values - y_bo.min(dim=0).values
        ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)
        ref_point = worst + 0.1 * ranges

        hv = compute_hypervolume(pareto_y, ref_point)

        # Sample efficiency for multi-objective: hypervolume per result
        sample_efficiency = hv / len(results) if len(results) > 0 else 0.0

        metrics.update(
            {
                "best_value": None,
                "improvement_rate": None,
                "sample_efficiency": round(sample_efficiency, 6),
                "hypervolume": round(hv, 6),
                "n_pareto_points": pareto_y.shape[0],
            }
        )

    return metrics


def _compare_metrics(
    metrics_list: list[dict[str, Any]],
    campaign_names: list[str],
) -> dict[str, Any]:
    """Generate comparison analysis across campaigns.

    Returns:
        Dictionary with comparison insights
    """
    # Find best performers
    best_sample_efficiency_idx = max(
        range(len(metrics_list)),
        key=lambda i: metrics_list[i].get("sample_efficiency", 0) or 0,
    )

    # For single-objective: find best value (considering minimization)
    single_obj_campaigns = [
        (i, m) for i, m in enumerate(metrics_list) if not m["is_multi_objective"]
    ]
    best_single_obj = None
    if single_obj_campaigns:
        best_single_obj_idx = min(
            (i for i, m in single_obj_campaigns if m.get("best_value") is not None),
            key=lambda i: metrics_list[i]["best_value"],
            default=None,
        )
        if best_single_obj_idx is not None:
            best_single_obj = campaign_names[best_single_obj_idx]

    # For multi-objective: find best hypervolume
    multi_obj_campaigns = [(i, m) for i, m in enumerate(metrics_list) if m["is_multi_objective"]]
    best_multi_obj = None
    if multi_obj_campaigns:
        best_multi_obj_idx = max(
            (i for i, m in multi_obj_campaigns if m.get("hypervolume") is not None),
            key=lambda i: metrics_list[i]["hypervolume"],
            default=None,
        )
        if best_multi_obj_idx is not None:
            best_multi_obj = campaign_names[best_multi_obj_idx]

    return {
        "best_sample_efficiency": campaign_names[best_sample_efficiency_idx],
        "best_single_objective": best_single_obj,
        "best_multi_objective": best_multi_obj,
        "total_campaigns_compared": len(metrics_list),
        "recommendation": _generate_comparison_recommendation(metrics_list, campaign_names),
    }


def _generate_comparison_recommendation(
    metrics_list: list[dict[str, Any]],
    campaign_names: list[str],
) -> str:
    """Generate agent-friendly recommendation based on comparison."""
    if len(metrics_list) < 2:
        return "Need at least 2 campaigns to generate meaningful comparison."

    # Find the most data-rich campaign
    most_data_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i]["n_results"])
    most_data_campaign = campaign_names[most_data_idx]

    # Find the most efficient campaign
    most_efficient_idx = max(
        range(len(metrics_list)),
        key=lambda i: metrics_list[i].get("sample_efficiency", 0) or 0,
    )
    most_efficient_campaign = campaign_names[most_efficient_idx]

    if most_data_idx == most_efficient_idx:
        return (
            f"Campaign '{most_data_campaign}' has both the most data and best sample "
            "efficiency. Consider using its configuration as a baseline for future campaigns."
        )
    else:
        return (
            f"Campaign '{most_efficient_campaign}' has the best sample efficiency, while "
            f"'{most_data_campaign}' has the most data. Consider transferring learnings "
            f"from '{most_efficient_campaign}' to improve future campaigns."
        )


@mcp.tool()
async def compare_campaigns(
    campaign_ids: list[str],
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Compare multiple optimization campaigns.

    This tool helps AI agents understand relative performance across
    different campaigns to guide strategy decisions.

    Args:
        campaign_ids: List of campaign UUIDs to compare (2-10 campaigns)
        verbosity: Response verbosity level. Options:
            - "minimal": ~50 tokens - success + best performer + recommendation only
            - "standard": ~200 tokens - simplified campaign metrics
            - "detailed": ~500+ tokens - all metrics including improvement rates

    Returns:
        Dictionary with:
            - success: Boolean indicating if comparison succeeded
            - campaigns: List of campaign metrics
            - comparison: Cross-campaign analysis
            - errors: List of error messages (if failed)
    """
    logger.info("Comparing %d campaigns, verbosity=%s", len(campaign_ids), verbosity)

    # Validate verbosity parameter
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )
        response.update({"campaigns": [], "comparison": None})
        return response

    if len(campaign_ids) < 2:
        logger.warning("Not enough campaigns to compare: %d", len(campaign_ids))
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

    # Validate UUIDs
    campaign_uuids = []
    for cid in campaign_ids:
        try:
            campaign_uuids.append(UUID(cid))
        except ValueError:
            response = make_error_response(
                ErrorCode.INVALID_CAMPAIGN_ID,
                message=f"Invalid campaign_id format: {cid}",
                details={"campaign_id": cid},
            )
            response.update({"campaigns": [], "comparison": None})
            return response

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)

        campaign_metrics = []
        campaign_names = []
        errors = []

        for campaign_uuid in campaign_uuids:
            # Get campaign
            campaign = await campaign_repo.get(campaign_uuid)
            if campaign is None:
                errors.append(f"Campaign {campaign_uuid} not found")
                continue

            # Get spec
            spec = await spec_repo.get(campaign.spec_id)
            if spec is None:
                errors.append(f"Campaign spec for {campaign_uuid} not found")
                continue

            # Get results
            results = await result_repo.list_by_campaign(campaign_uuid)

            # Compute metrics
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

        # Generate comparison analysis
        comparison = _compare_metrics(campaign_metrics, campaign_names)

        logger.info(
            "Campaign comparison completed: %d campaigns, best efficiency=%s",
            len(campaign_metrics),
            comparison.get("best_sample_efficiency"),
        )
        full_response = {
            "success": True,
            "campaigns": campaign_metrics,
            "comparison": comparison,
            "errors": errors,
        }
        return format_compare_campaigns_response(full_response, verbosity_level)
