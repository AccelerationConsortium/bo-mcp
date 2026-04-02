"""Get diagnostics operation — protocol-neutral diagnostics computation.

Delegates model-based computation (GP fitting, LOO-CV, feature importance,
outlier detection, Pareto/hypervolume) to the BOBackend protocol. Server-side
concerns (campaign state, suggestion provenance, caching, formatting) stay here.
"""

import logging
from typing import Any
from uuid import UUID

import torch
from bo_engine.constants import MIN_IMPROVEMENT_RATE
from bo_engine.convergence import (
    detect_hypervolume_convergence,
    detect_single_objective_convergence,
)
from bo_engine.diagnostics import (
    compute_constraint_satisfaction,
    compute_exploration_exploitation_metrics,
    compute_suggestion_diversity,
    compute_uncertainty_trend,
    determine_health_status,
    determine_progress_status,
    determine_single_objective_health_status,
)
from bo_engine.transforms import encode_categorical
from bo_engine.types import OptimizationSpec

from bo_mcp_server.backend import get_backend
from bo_mcp_server.cache import diagnostics_cache
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import Campaign, CampaignSpec, Result, Suggestion, SuggestionStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.helpers import (
    parse_campaign_id,
    parse_verbosity,
    results_to_observations,
)
from bo_mcp_server.response_formatter import VerbosityLevel, format_diagnostics_response
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    SuggestionRepository,
    get_session,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Server-side helpers (no torch/model access)
# =============================================================================


def _compute_objective_ranges(
    spec: CampaignSpec,
    results: list[Result],
) -> dict[str, dict[str, Any]]:
    """Compute min/max ranges for each objective."""
    if not results:
        return {}
    objective_ranges = {}
    for obj in spec.objectives:
        values = [r.objective_values[obj.name] for r in results]
        objective_ranges[obj.name] = {
            "min": min(values),
            "max": max(values),
            "direction": obj.direction,
        }
    return objective_ranges


def _get_model_info(spec: CampaignSpec, is_single_objective: bool) -> dict[str, Any]:
    """Get model information summary."""
    acquisition_fn = spec.acquisition_method.value
    if acquisition_fn == "auto":
        acquisition_fn = (
            "noisy_expected_improvement" if is_single_objective else "hypervolume_improvement"
        )

    acquisition_desc = {
        "noisy_expected_improvement": "Log Noisy Expected Improvement (qLogNEI)",
        "expected_improvement": "Log Expected Improvement (qLogEI)",
        "hypervolume_improvement": "Log Expected Hypervolume Improvement (qLogNEHVI)",
        "scalarized_multi_objective": "Parallel EGO with Chebyshev (qLogNParEGO)",
        "cost_weighted_ei": "Expected Improvement per Unit cost (EIpu)",
        "multi_fidelity_kg": "Multi-Fidelity Knowledge Gradient (qMFKG)",
    }.get(acquisition_fn, acquisition_fn)

    model_type = (
        "SingleTaskGP (Gaussian Process)"
        if is_single_objective
        else "ModelListGP (Multi-output Gaussian Process)"
    )

    return {
        "type": model_type,
        "acquisition_function": acquisition_desc,
        "batch_strategy": "Sequential greedy optimization",
        "kernel": "Matern 5/2 with automatic relevance determination",
        "input_warping": spec.use_input_warping,
    }


# =============================================================================
# Health & Progress (uses plain Python from bo_engine.diagnostics)
# =============================================================================


def _compute_health_and_progress(
    _spec: CampaignSpec,
    results: list[Result],
    _campaign_iteration: int,
    is_single_objective: bool,
    model_correlation: float,
    hypervolume_history: list[float],
    diagnostics: dict[str, Any],
) -> None:
    """Compute health status and progress status."""
    if is_single_objective:
        health_status, warnings, progress_status = _single_obj_health(
            diagnostics,
            model_correlation,
        )
    else:
        health_status, warnings, progress_status = _multi_obj_health(
            results,
            diagnostics,
            model_correlation,
            hypervolume_history,
        )

    diagnostics["health_status"] = health_status
    diagnostics["warnings"] = warnings
    diagnostics["progress_status"] = progress_status
    diagnostics["hypervolume_history"] = hypervolume_history


def _single_obj_health(
    diagnostics: dict[str, Any],
    model_correlation: float,
) -> tuple[str, list[str], str]:
    """Compute health/progress for single-objective campaigns."""
    improvement_history = diagnostics.get("improvement_history", [])
    health_status, warnings = determine_single_objective_health_status(
        improvement_history=improvement_history,
        model_correlation=model_correlation,
    )
    improvement_rate = diagnostics.get("improvement_rate", 0)
    progress_status = "improving" if improvement_rate > MIN_IMPROVEMENT_RATE else "stable"
    return health_status, warnings, progress_status


def _multi_obj_health(
    results: list[Result],
    diagnostics: dict[str, Any],
    model_correlation: float,
    hypervolume_history: list[float],
) -> tuple[str, list[str], str]:
    """Compute health/progress for multi-objective campaigns."""
    hv_improvement, iters_stagnant = _analyze_hypervolume_history(
        hypervolume_history,
        results,
        diagnostics,
    )

    health_status, warnings = determine_health_status(
        n_results=len(results),
        hypervolume_improvement=hv_improvement,
        model_correlation=model_correlation,
        iterations_without_improvement=iters_stagnant,
    )

    hv_hist = hypervolume_history if hypervolume_history else [diagnostics.get("hypervolume", 0)]
    progress_status = determine_progress_status(hv_hist)
    return health_status, warnings, progress_status


def _analyze_hypervolume_history(
    hypervolume_history: list[float],
    results: list[Result],
    diagnostics: dict[str, Any],
) -> tuple[float, int]:
    """Analyze hypervolume history for improvement and stagnation."""
    if len(hypervolume_history) >= 2:
        recent_improvement = hypervolume_history[-1] - hypervolume_history[-2]
        hv_improvement = max(0.0, recent_improvement)

        threshold = 0.001 * abs(hypervolume_history[-1]) if hypervolume_history[-1] != 0 else 0.001
        iters_stagnant = 0
        for i in range(len(hypervolume_history) - 1, 0, -1):
            if hypervolume_history[i] - hypervolume_history[i - 1] < threshold:
                iters_stagnant += 1
            else:
                break
        return hv_improvement, iters_stagnant

    if len(results) >= 2:
        hv_improvement = 0.1 if diagnostics.get("hypervolume", 0) > 0 else 0.0
        return hv_improvement, 0

    return 0.0, 0


# =============================================================================
# Agent Usability Diagnostics (v2.4)
# =============================================================================


def _compute_uncertainty_trends(
    suggestions: list[Suggestion],
    diagnostics: dict[str, Any],
) -> None:
    """Compute uncertainty trends from suggestion history."""
    uncertainty_history = []
    for sugg in suggestions:
        if sugg.provenance.model_uncertainty is not None:
            uncertainty_history.append(sugg.provenance.model_uncertainty)

    if not uncertainty_history:
        diagnostics["uncertainty_trend"] = None
        return

    trend = compute_uncertainty_trend(uncertainty_history)
    diagnostics["uncertainty_trend"] = {
        "mean_uncertainty": round(trend.mean_uncertainty, 4),
        "std_uncertainty": round(trend.std_uncertainty, 4),
        "trend": trend.trend,
        "slope": round(trend.slope, 6),
        "interpretation": _interpret_uncertainty_trend(trend.trend),
    }


def _interpret_uncertainty_trend(trend: str) -> str:
    """Provide agent-friendly interpretation of uncertainty trend."""
    if trend == "decreasing":
        return (
            "Model uncertainty is decreasing. The model is becoming more confident "
            "as it learns from observations. This is expected behavior."
        )
    elif trend == "increasing":
        return (
            "Model uncertainty is increasing. This may indicate the optimization is "
            "exploring new regions, or the model is struggling to fit the data."
        )
    else:
        return (
            "Model uncertainty is stable. The model maintains consistent confidence "
            "across recent suggestions."
        )


def _compute_exploration_exploitation(
    suggestions: list[Suggestion],
    results: list[Result],
    spec: CampaignSpec,
    opt_spec: OptimizationSpec,
    diagnostics: dict[str, Any],
) -> None:
    """Compute exploration/exploitation balance metrics."""
    if len(suggestions) < 2:
        diagnostics["exploration_exploitation"] = None
        return

    try:
        # Get recent pending suggestions
        pending = [s for s in suggestions if s.status == SuggestionStatus.PENDING]
        if not pending:
            pending = suggestions[-5:]  # Use recent suggestions if no pending

        # Build suggestion tensor
        sugg_list = []
        uncertainties = []
        for s in pending:
            x_encoded = encode_categorical(s.parameter_values, opt_spec)
            sugg_list.append(x_encoded)
            if s.provenance.model_uncertainty is not None:
                uncertainties.append(s.provenance.model_uncertainty)

        if not sugg_list:
            diagnostics["exploration_exploitation"] = None
            return

        from bo_engine.diagnostics import compute_best_value
        from bo_engine.transforms import get_bounds_tensor

        suggestions_tensor = torch.stack(sugg_list)

        # Get best point from results
        best_point = None
        if results:
            obj = spec.objectives[0]
            values = [r.objective_values[obj.name] for r in results]
            if values:
                _, best_idx = compute_best_value(values, minimize=obj.is_minimize)
                if best_idx >= 0:
                    best_encoded = encode_categorical(results[best_idx].parameter_values, opt_spec)
                    best_point = best_encoded

        bounds = get_bounds_tensor(opt_spec)

        metrics = compute_exploration_exploitation_metrics(
            suggestions=suggestions_tensor,
            best_point=best_point,
            uncertainties=uncertainties,
            bounds=bounds,
        )

        diagnostics["exploration_exploitation"] = {
            "exploration_ratio": round(metrics.exploration_ratio, 4),
            "diversity_score": round(metrics.diversity_score, 4),
            "average_distance_to_best": round(metrics.average_distance_to_best, 4),
            "balance_assessment": metrics.balance_assessment,
            "recommendation": metrics.recommendation,
        }
    except (RuntimeError, ValueError, TypeError) as e:
        logger.debug("Exploration/exploitation metrics computation failed: %s", e)
        diagnostics["exploration_exploitation"] = None


def _compute_suggestion_diversity_metrics(
    suggestions: list[Suggestion],
    opt_spec: OptimizationSpec,
    diagnostics: dict[str, Any],
) -> None:
    """Compute diversity metrics for recent suggestions."""
    pending = [s for s in suggestions if s.status == SuggestionStatus.PENDING]
    if len(pending) < 2:
        diagnostics["suggestion_diversity"] = None
        return

    try:
        sugg_list = []
        for s in pending:
            x_encoded = encode_categorical(s.parameter_values, opt_spec)
            sugg_list.append(x_encoded)

        if len(sugg_list) < 2:
            diagnostics["suggestion_diversity"] = None
            return

        suggestions_tensor = torch.stack(sugg_list)
        diversity = compute_suggestion_diversity(suggestions_tensor)

        diagnostics["suggestion_diversity"] = {
            "diversity_score": round(diversity, 4),
            "n_suggestions": len(sugg_list),
            "interpretation": _interpret_diversity(diversity),
        }
    except (RuntimeError, ValueError, TypeError) as e:
        logger.debug("Suggestion diversity computation failed: %s", e)
        diagnostics["suggestion_diversity"] = None


def _interpret_diversity(diversity: float) -> str:
    """Provide agent-friendly interpretation of diversity score."""
    if diversity >= 0.7:
        return "High diversity. Suggestions explore diverse regions of the parameter space."
    elif diversity >= 0.4:
        return "Moderate diversity. Suggestions cover a reasonable range of the parameter space."
    else:
        return (
            "Low diversity. Suggestions are clustered together, which may indicate "
            "convergence or over-exploitation. Consider increasing exploration if stuck."
        )


def _interpret_lengthscales(lengthscales: dict[str, float]) -> str:
    """Provide agent-friendly interpretation of lengthscales."""
    if not lengthscales:
        return "No lengthscale information available."

    sorted_params = sorted(lengthscales.items(), key=lambda x: x[1])

    most_important = sorted_params[0][0] if sorted_params else "unknown"
    least_important = sorted_params[-1][0] if sorted_params else "unknown"

    if len(sorted_params) == 1:
        return f"The parameter '{most_important}' is the only tunable parameter."

    return (
        f"Based on lengthscales, '{most_important}' has the strongest influence on the "
        f"objective, while '{least_important}' has the weakest influence. "
        "Smaller lengthscales indicate more sensitivity to changes."
    )


# =============================================================================
# Constraint Satisfaction (plain Python, no torch)
# =============================================================================


def _compute_constraint_satisfaction_metrics(
    results: list[Result],
    spec: CampaignSpec,
    diagnostics: dict[str, Any],
) -> None:
    """Compute constraint satisfaction rate over time."""
    if not spec.constraints:
        diagnostics["constraint_satisfaction"] = None
        return

    try:
        result_dicts = [r.parameter_values for r in results]
        constraint_dicts = [
            {
                "type": c.type.value if hasattr(c.type, "value") else str(c.type),
                "parameters": c.parameters,
                "value": c.value,
                "coefficients": c.coefficients if hasattr(c, "coefficients") else None,
            }
            for c in spec.constraints
        ]

        metrics = compute_constraint_satisfaction(result_dicts, constraint_dicts)

        diagnostics["constraint_satisfaction"] = {
            "satisfaction_rate": metrics.satisfaction_rate,
            "recent_satisfaction_rate": metrics.recent_satisfaction_rate,
            "feasible_count": metrics.feasible_count,
            "infeasible_count": metrics.infeasible_count,
            "trend": metrics.trend,
            "interpretation": _interpret_constraint_satisfaction(metrics),
        }
    except (RuntimeError, ValueError, TypeError) as e:
        logger.debug("Constraint satisfaction computation failed: %s", e)
        diagnostics["constraint_satisfaction"] = None


def _interpret_constraint_satisfaction(metrics: Any) -> str:
    """Provide agent-friendly interpretation of constraint satisfaction."""
    if metrics.satisfaction_rate >= 0.95:
        return "Excellent constraint satisfaction. Almost all suggestions are feasible."
    elif metrics.satisfaction_rate >= 0.8:
        return "Good constraint satisfaction. Most suggestions are feasible."
    elif metrics.satisfaction_rate >= 0.5:
        return (
            "Moderate constraint satisfaction. Consider reviewing constraint values "
            "or expanding the feasible region."
        )
    else:
        return (
            "Low constraint satisfaction. The constraints may be too restrictive. "
            "Consider relaxing constraints or using a different optimization approach."
        )


# =============================================================================
# Convergence Detection (Section 1.4)
# =============================================================================


def _compute_convergence_diagnostics(
    spec: CampaignSpec,
    is_single_objective: bool,
    hypervolume_history: list[float],
    diagnostics: dict[str, Any],
) -> None:
    """Compute convergence/early stopping detection metrics."""
    try:
        if is_single_objective:
            _compute_single_obj_convergence(spec, diagnostics)
        else:
            _compute_multi_obj_convergence(hypervolume_history, diagnostics)
    except (RuntimeError, ValueError, TypeError) as e:
        logger.debug("Convergence detection failed: %s", e)
        diagnostics["convergence"] = None


def _compute_single_obj_convergence(
    spec: CampaignSpec,
    diagnostics: dict[str, Any],
) -> None:
    """Compute convergence for single-objective campaigns."""
    improvement_history = diagnostics.get("improvement_history", [])
    if len(improvement_history) >= 5:
        obj = spec.objectives[0]
        report = detect_single_objective_convergence(
            best_value_history=improvement_history,
            minimize=obj.is_minimize,
        )
        diagnostics["convergence"] = {
            "converged": report.converged,
            "convergence_score": round(report.convergence_score, 4),
            "reason": report.reason,
            "avg_improvement": round(report.avg_improvement, 6),
            "iterations_without_improvement": report.iterations_without_improvement,
            "recommendation": report.recommendation,
        }
    else:
        diagnostics["convergence"] = {
            "converged": False,
            "convergence_score": 0.0,
            "reason": "Insufficient data for convergence detection",
            "recommendation": "Continue optimization to gather more data.",
        }


def _compute_multi_obj_convergence(
    hypervolume_history: list[float],
    diagnostics: dict[str, Any],
) -> None:
    """Compute convergence for multi-objective campaigns."""
    hv_history = hypervolume_history if hypervolume_history else []
    if len(hv_history) >= 5:
        report = detect_hypervolume_convergence(hv_history)
        diagnostics["convergence"] = {
            "converged": report.converged,
            "convergence_score": round(report.convergence_score, 4),
            "reason": report.reason,
            "avg_improvement": round(report.avg_improvement, 6),
            "iterations_without_improvement": report.iterations_without_improvement,
            "recommendation": report.recommendation,
        }
    elif hv_history:
        diagnostics["convergence"] = {
            "converged": False,
            "convergence_score": 0.0,
            "reason": f"Insufficient history ({len(hv_history)} points, need 5)",
            "recommendation": (
                "Continue optimization to gather more data for convergence detection."
            ),
        }
    else:
        diagnostics["convergence"] = {
            "converged": False,
            "convergence_score": 0.0,
            "reason": "No hypervolume history available",
            "recommendation": (
                "Submit results to build hypervolume history for convergence detection."
            ),
        }


# =============================================================================
# Next Action Recommendation (v3.3)
# =============================================================================


def _compute_next_action_recommendation(
    diagnostics: dict[str, Any],
    n_pending_suggestions: int,
    campaign_status: str,
) -> None:
    """Compute proactive next action recommendation for agents."""
    n_results = diagnostics.get("n_results", 0)
    health_status = diagnostics.get("health_status", "unknown")
    convergence = diagnostics.get("convergence", {})
    converged = convergence.get("converged", False)
    outliers = diagnostics.get("outliers", {})
    outlier_count = outliers.get("count", 0) if outliers else 0

    action, reason, urgency = _determine_next_action(
        campaign_status,
        converged,
        convergence,
        outlier_count,
        n_results,
        health_status,
        diagnostics,
        n_pending_suggestions,
    )

    diagnostics["next_action_recommendation"] = {
        "action": action,
        "reason": reason,
        "urgency": urgency,
    }


def _health_status_action(
    health_status: str,
    diagnostics: dict[str, Any],
) -> tuple[str, str, str]:
    """Return action tuple for critical/warning health status."""
    if health_status == "critical":
        warnings = diagnostics.get("warnings", [])
        return (
            "investigate_issues",
            f"Campaign health is critical. Issues: {warnings[:2] if warnings else 'Unknown'}",
            "high",
        )
    return (
        "monitor_progress",
        "Campaign health has warnings. Continue but monitor closely.",
        "normal",
    )


def _determine_next_action(
    campaign_status: str,
    converged: bool,
    convergence: dict[str, Any],
    outlier_count: int,
    n_results: int,
    health_status: str,
    diagnostics: dict[str, Any],
    n_pending_suggestions: int,
) -> tuple[str, str, str]:
    """Determine the next action, reason, and urgency."""
    if campaign_status in ("paused", "completed", "failed"):
        return (
            "review_campaign_status",
            f"Campaign is {campaign_status}. Resume or create new campaign if needed.",
            "low",
        )
    if converged:
        reason = convergence.get("reason", "Optimization has converged")
        return "consider_stopping", f"{reason}. Consider terminating the campaign.", "normal"
    if outlier_count > 0 and n_results > 5:
        return (
            "review_outliers",
            f"Detected {outlier_count} potential outlier(s). Verify measurements for errors.",
            "normal",
        )
    if health_status in ("critical", "warning"):
        return _health_status_action(health_status, diagnostics)
    if n_pending_suggestions > 0:
        return (
            "bo_submit_results",
            f"Campaign has {n_pending_suggestions} pending suggestion(s) awaiting results.",
            "normal",
        )
    reason = (
        "No results yet. Generate initial suggestions to start optimization."
        if n_results == 0
        else f"Campaign healthy with {n_results} results. Ready for next batch of suggestions."
    )
    return "bo_generate_suggestions", reason, "normal"


# =============================================================================
# Outlier result enrichment (adds result IDs from server domain)
# =============================================================================


def _enrich_outlier_results(
    outlier_data: dict[str, Any] | None,
    results: list[Result],
) -> dict[str, Any] | None:
    """Add server-side result IDs and recommendation to backend outlier data."""
    if outlier_data is None:
        return None

    outlier_results = outlier_data.get("outlier_results", [])
    for info in outlier_results:
        idx = info.get("result_index", -1)
        if 0 <= idx < len(results):
            info["result_id"] = str(results[idx].id)

    count = outlier_data.get("count", 0)
    if count > 0:
        outlier_data["recommendation"] = (
            f"Detected {count} potential outlier(s). "
            "These results deviate significantly from model predictions. "
            "Consider verifying these measurements for errors."
        )
    else:
        outlier_data["recommendation"] = "No outliers detected. Results appear consistent."

    return outlier_data


# =============================================================================
# Main Operation Function
# =============================================================================


ALL_SECTIONS = frozenset(
    ["health", "objectives", "model", "convergence", "suggestions", "outliers", "constraints"]
)


def _validate_diagnostics_inputs(
    campaign_id: str,
    verbosity: str,
    sections: list[str] | None,
) -> tuple[VerbosityLevel, UUID, frozenset[str]] | dict[str, Any]:
    """Validate diagnostics inputs. Returns (level, uuid, sections) or error dict."""
    verbosity_result = parse_verbosity(verbosity)
    if isinstance(verbosity_result, dict):
        return verbosity_result
    verbosity_level = verbosity_result

    requested = ALL_SECTIONS if sections is None else frozenset(sections)
    invalid_sections = requested - ALL_SECTIONS
    if invalid_sections:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(
                f"Invalid sections: {sorted(invalid_sections)}. Valid: {sorted(ALL_SECTIONS)}"
            ),
        )

    campaign_id_result = parse_campaign_id(campaign_id)
    if isinstance(campaign_id_result, dict):
        return campaign_id_result
    campaign_uuid = campaign_id_result

    return verbosity_level, campaign_uuid, requested


async def get_diagnostics_operation(
    campaign_id: str,
    use_cache: bool = True,
    verbosity: str = "standard",
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Compute diagnostic information for a campaign.

    Protocol-neutral operation used by MCP tools and REST routes.
    Delegates model-based computation to the BOBackend protocol.

    Args:
        campaign_id: UUID of the campaign
        use_cache: Whether to use cached results (default True).
        verbosity: Response verbosity level (minimal, standard, detailed).
        sections: Optional list of sections to compute. When omitted, all
            sections are computed. Valid: health, objectives, model,
            convergence, suggestions, outliers, constraints.

    Returns:
        Formatted diagnostics dictionary.
    """
    logger.info(
        "Getting diagnostics for campaign_id=%s, use_cache=%s, verbosity=%s, sections=%s",
        campaign_id,
        use_cache,
        verbosity,
        sections,
    )

    validated = _validate_diagnostics_inputs(campaign_id, verbosity, sections)
    if isinstance(validated, dict):
        return validated
    verbosity_level, campaign_uuid, requested = validated

    is_full = requested == ALL_SECTIONS

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)
        suggestion_repo = SuggestionRepository(session)

        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            return make_error_response(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )

        cache_key = f"diagnostics:{campaign_id}:{campaign.version}"
        if use_cache and is_full:
            cached = await diagnostics_cache.get(cache_key)
            if cached is not None:
                logger.debug("Returning cached diagnostics for campaign %s", campaign_id)
                return format_diagnostics_response(cached, verbosity_level)

        spec = await spec_repo.get(campaign.spec_id)
        if spec is None:
            return make_error_response(
                ErrorCode.DATABASE_ERROR,
                message="Campaign spec not found",
                details={"spec_id": str(campaign.spec_id)},
            )

        results = await result_repo.list_by_campaign(campaign_uuid)
        all_suggestions = await suggestion_repo.list_by_campaign(campaign_uuid)
        pending_suggestions = [s for s in all_suggestions if s.status == SuggestionStatus.PENDING]

        diagnostics = _compute_sections(
            requested,
            spec,
            results,
            all_suggestions,
            pending_suggestions,
            campaign,
        )

        logger.info(
            "Diagnostics computed for campaign %s: results=%d, health=%s",
            campaign_id,
            len(results),
            diagnostics.get("health_status", "unknown"),
        )

        if is_full:
            await diagnostics_cache.set(cache_key, diagnostics)

        return format_diagnostics_response(diagnostics, verbosity_level)


def _map_backend_sections(requested: frozenset[str]) -> frozenset[str]:
    """Map server-side section names to backend section names."""
    mapping = {
        "objectives": "objectives",
        "health": "objectives",  # health needs objective data
        "model": "model",
        "outliers": "outliers",
        "suggestions": "suggestions_tensor",
    }
    return frozenset(mapping[s] for s in requested if s in mapping)


def _enrich_diagnostics(
    diagnostics: dict[str, Any],
    spec: CampaignSpec,
    results: list[Result],
    is_single_objective: bool,
) -> None:
    """Add server-side enrichments to backend diagnostics."""
    diagnostics["objective_ranges"] = _compute_objective_ranges(spec, results)
    diagnostics["model_info"] = _get_model_info(spec, is_single_objective)

    if diagnostics.get("hyperparameters") is not None:
        hp = diagnostics["hyperparameters"]
        hp["interpretation"] = _interpret_lengthscales(hp.get("lengthscales", {}))


def _compute_sections(
    requested: frozenset[str],
    spec: CampaignSpec,
    results: list[Result],
    all_suggestions: list[Suggestion],
    pending_suggestions: list[Suggestion],
    campaign: Campaign,
) -> dict[str, Any]:
    """Compute only the requested diagnostic sections."""
    is_single_objective = len(spec.objectives) == 1
    opt_spec = campaign_spec_to_optimization_spec(spec)

    diagnostics: dict[str, Any] = {
        "success": True,
        "campaign_status": campaign.status.value,
        "iteration": campaign.iteration,
        "n_results": len(results),
        "n_pending_suggestions": len(pending_suggestions),
        "errors": [],
    }

    # Delegate model-based computation to the backend
    backend_sections = _map_backend_sections(requested)
    if backend_sections:
        backend = get_backend(spec.backend)
        observations = results_to_observations(results)
        diagnostics.update(backend.compute_diagnostics(opt_spec, observations, backend_sections))

    # Enrich with server-side model info
    if "objectives" in requested or "health" in requested:
        _enrich_diagnostics(diagnostics, spec, results, is_single_objective)

    # Health (plain-Python functions + backend correlation)
    model_correlation = diagnostics.get("model_correlation") or 0.5
    if "health" in requested:
        _compute_health_and_progress(
            spec,
            results,
            campaign.iteration,
            is_single_objective,
            model_correlation,
            campaign.hypervolume_history,
            diagnostics,
        )
        _compute_next_action_recommendation(
            diagnostics,
            len(pending_suggestions),
            campaign.status.value,
        )

    # Suggestions — server-side parts (provenance-based)
    if "suggestions" in requested:
        _compute_uncertainty_trends(all_suggestions, diagnostics)
        _compute_exploration_exploitation(all_suggestions, results, spec, opt_spec, diagnostics)
        _compute_suggestion_diversity_metrics(all_suggestions, opt_spec, diagnostics)

    if "constraints" in requested:
        _compute_constraint_satisfaction_metrics(results, spec, diagnostics)

    if "convergence" in requested:
        _compute_convergence_diagnostics(
            spec,
            is_single_objective,
            campaign.hypervolume_history,
            diagnostics,
        )

    if "outliers" in requested:
        diagnostics["outliers"] = _enrich_outlier_results(
            diagnostics.get("outliers"),
            results,
        )

    return diagnostics
