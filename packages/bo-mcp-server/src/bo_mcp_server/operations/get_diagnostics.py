"""Get diagnostics operation — protocol-neutral diagnostics computation."""

import logging
from typing import Any, cast
from uuid import UUID

import torch
from bo_engine.convergence import (
    detect_hypervolume_convergence,
    detect_single_objective_convergence,
)
from bo_engine.diagnostics import (
    LOOCVMetrics,
    compute_best_value,
    compute_constraint_satisfaction,
    compute_exploration_exploitation_metrics,
    compute_hypervolume,
    compute_improvement_history,
    compute_loo_cv_for_model,
    compute_pareto_front,
    compute_rank_correlation,
    compute_single_objective_improvement_rate,
    compute_suggestion_diversity,
    compute_uncertainty_trend,
    determine_health_status,
    determine_progress_status,
    determine_single_objective_health_status,
    extract_hyperparameters,
    summarize_pareto_front,
)
from bo_engine.feature_importance import compute_feature_importance
from bo_engine.models import create_and_fit_model, create_and_fit_single_task_model
from bo_engine.result_validation import detect_outliers
from bo_engine.transforms import encode_categorical, get_bounds_tensor
from torch import Tensor

from bo_mcp_server.cache import diagnostics_cache
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import CampaignSpec, Result, Suggestion, SuggestionStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
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
# Helper Functions for Diagnostics Computation
# =============================================================================


def _compute_single_objective_diagnostics(
    spec: CampaignSpec,
    results: list[Result],
    diagnostics: dict[str, Any],
) -> None:
    """Compute diagnostics for single-objective optimization.

    Modifies diagnostics dict in place.
    """
    obj = spec.objectives[0]
    values = [r.objective_values[obj.name] for r in results]

    if values:
        best_value, best_idx = compute_best_value(values, minimize=obj.is_minimize)
        improvement_history = compute_improvement_history(values, minimize=obj.is_minimize)
        improvement_rate = compute_single_objective_improvement_rate(improvement_history)

        diagnostics["best_value"] = best_value
        diagnostics["best_parameters"] = results[best_idx].parameter_values
        diagnostics["improvement_history"] = improvement_history
        diagnostics["improvement_rate"] = improvement_rate
    else:
        diagnostics["best_value"] = None
        diagnostics["best_parameters"] = None
        diagnostics["improvement_history"] = []
        diagnostics["improvement_rate"] = 0.0

    # Multi-objective fields set to null for single-objective
    diagnostics["pareto_front"] = None
    diagnostics["hypervolume"] = None
    diagnostics["n_pareto_points"] = None


def _compute_multi_objective_diagnostics(
    spec: CampaignSpec,
    results: list[Result],
    diagnostics: dict[str, Any],
) -> None:
    """Compute diagnostics for multi-objective optimization.

    Modifies diagnostics dict in place.
    """
    objective_names = [o.name for o in spec.objectives]

    # Single-objective fields set to null for multi-objective
    diagnostics["best_value"] = None
    diagnostics["best_parameters"] = None
    diagnostics["improvement_history"] = None
    diagnostics["improvement_rate"] = None

    if len(results) >= 2:
        # Build objective tensor
        minimize_mask = torch.tensor([o.is_minimize for o in spec.objectives], dtype=torch.bool)

        y_list = []
        for r in results:
            y = torch.tensor(
                [r.objective_values[name] for name in objective_names],
                dtype=torch.double,
            )
            y_list.append(y)

        y_tensor = torch.stack(y_list)

        # Negate maximization objectives for internal computation
        y_bo = y_tensor.clone()
        y_bo[:, ~minimize_mask] = -y_bo[:, ~minimize_mask]

        # Compute Pareto front
        pareto_y, _pareto_mask = compute_pareto_front(y_bo)

        # Transform back for display
        pareto_display = pareto_y.clone()
        pareto_display[:, ~minimize_mask] = -pareto_display[:, ~minimize_mask]

        # Compute reference point and hypervolume
        worst = y_bo.max(dim=0).values
        ranges = y_bo.max(dim=0).values - y_bo.min(dim=0).values
        ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)
        ref_point = worst + 0.1 * ranges

        hv = compute_hypervolume(pareto_y, ref_point)

        # Summarize Pareto front
        pareto_summary = summarize_pareto_front(pareto_display, objective_names)

        diagnostics["pareto_front"] = pareto_summary
        diagnostics["hypervolume"] = hv
        diagnostics["n_pareto_points"] = len(pareto_summary)
    else:
        diagnostics["pareto_front"] = []
        diagnostics["hypervolume"] = 0.0
        diagnostics["n_pareto_points"] = 0


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


def _prepare_training_data(
    spec: CampaignSpec,
    results: list[Result],
    opt_spec: Any,
) -> tuple[Tensor, Tensor, list[str], list[str]]:
    """Prepare training data tensors from results.

    Returns:
        Tuple of (train_x, train_y, param_names, objective_names)
    """
    param_names = [p.name for p in spec.parameters]
    objective_names = [o.name for o in spec.objectives]

    # Build train_x tensor (parameters)
    x_list = []
    for r in results:
        x_encoded = encode_categorical(r.parameter_values, opt_spec)
        x_list.append(x_encoded)
    train_x = torch.stack(x_list)

    # Build train_y tensor (objectives, negated for maximization)
    y_list = []
    for r in results:
        y = [r.objective_values[name] for name in objective_names]
        y_list.append(torch.tensor(y, dtype=torch.double))
    train_y = torch.stack(y_list)

    # Negate maximization objectives
    minimize_mask = torch.tensor([o.is_minimize for o in spec.objectives], dtype=torch.bool)
    train_y[:, ~minimize_mask] = -train_y[:, ~minimize_mask]

    return train_x, train_y, param_names, objective_names


def _compute_model_correlation(
    model: Any,
    train_x: Tensor,
    train_y: Tensor,
    is_single_objective: bool,
) -> float:
    """Compute model correlation (predictions vs actuals)."""
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(train_x)
        predictions = posterior.mean

    if is_single_objective:
        return compute_rank_correlation(predictions.squeeze(), train_y.squeeze())
    else:
        correlations = []
        for obj_idx in range(train_y.shape[1]):
            corr = compute_rank_correlation(predictions[:, obj_idx], train_y[:, obj_idx])
            correlations.append(corr)
        return sum(correlations) / len(correlations)


def _loo_cv_metrics_to_dict(m: LOOCVMetrics) -> dict[str, float]:
    return {"rmse": m.rmse, "mae": m.mae, "r_squared": m.r_squared}


def _compute_model_diagnostics(
    spec: CampaignSpec,
    results: list[Result],
    is_single_objective: bool,
    diagnostics: dict[str, Any],
) -> float:
    """Compute model-based diagnostics (feature importance, LOO-CV, correlation).

    Modifies diagnostics dict in place.

    Returns:
        Computed model correlation (or default 0.5 if model couldn't be fitted)
    """
    n_params = len(spec.parameters)
    min_data_for_model = max(3, 2 * n_params)  # At least 3 for LOO-CV
    model_correlation = 0.5  # Default fallback if model cannot be fitted

    if len(results) < min_data_for_model:
        diagnostics["feature_importance"] = None
        diagnostics["loo_cv_metrics"] = None
        diagnostics["model_correlation"] = None
        return model_correlation

    try:
        # Convert to bo-engine types
        opt_spec = campaign_spec_to_optimization_spec(spec)

        # Prepare training data
        train_x, train_y, param_names, objective_names = _prepare_training_data(
            spec, results, opt_spec
        )

        # Build bounds
        bounds = get_bounds_tensor(opt_spec)

        # Fit model based on problem type
        if is_single_objective:
            model = create_and_fit_single_task_model(
                train_x,
                train_y,
                bounds,
                use_input_warping=spec.use_input_warping,
            )
        else:
            model = create_and_fit_model(
                train_x,
                train_y,
                bounds,
                use_input_warping=spec.use_input_warping,
            )

        # Compute model correlation
        model_correlation = _compute_model_correlation(model, train_x, train_y, is_single_objective)
        diagnostics["model_correlation"] = model_correlation

        # Compute feature importance
        feature_importance = compute_feature_importance(
            model,
            train_x,
            param_names,
            include_shap=False,  # Skip SHAP for speed
        )
        diagnostics["feature_importance"] = feature_importance

        # Compute LOO-CV metrics (v1.1)
        if len(results) >= 5:  # Need enough data for meaningful LOO-CV
            try:
                loo_metrics = compute_loo_cv_for_model(model, train_x, train_y)
                # Format LOO-CV metrics by objective name
                loo_cv_by_objective: dict[str, dict[str, float]] = {}
                if isinstance(loo_metrics, dict):
                    metrics_by_idx = cast(dict[int, LOOCVMetrics], loo_metrics)
                    for idx, obj_name in enumerate(objective_names):
                        if idx in metrics_by_idx:
                            loo_cv_by_objective[obj_name] = _loo_cv_metrics_to_dict(
                                metrics_by_idx[idx]
                            )
                elif objective_names:
                    loo_cv_by_objective[objective_names[0]] = _loo_cv_metrics_to_dict(loo_metrics)
                diagnostics["loo_cv_metrics"] = loo_cv_by_objective
            except Exception as e:
                logger.debug("LOO-CV computation failed: %s", e)
                diagnostics["loo_cv_metrics"] = None
        else:
            diagnostics["loo_cv_metrics"] = None
    except Exception as e:
        # Model-based diagnostics are optional - don't fail
        logger.debug("Model-based diagnostics computation failed: %s", e)
        diagnostics["feature_importance"] = None
        diagnostics["loo_cv_metrics"] = None
        diagnostics["model_correlation"] = None

    return model_correlation


def _compute_health_and_progress(
    spec: CampaignSpec,
    results: list[Result],
    campaign_iteration: int,
    is_single_objective: bool,
    model_correlation: float,
    hypervolume_history: list[float],
    diagnostics: dict[str, Any],
) -> None:
    """Compute health status and progress status.

    Args:
        spec: Campaign specification
        results: List of results
        campaign_iteration: Current iteration number
        is_single_objective: Whether this is single-objective optimization
        model_correlation: Model correlation score
        hypervolume_history: Stored hypervolume history from Campaign (Section 4.2)
        diagnostics: Dictionary to modify in place
    """
    if is_single_objective:
        improvement_history = diagnostics.get("improvement_history", [])
        health_status, warnings = determine_single_objective_health_status(
            improvement_history=improvement_history,
            model_correlation=model_correlation,
        )
        progress_status = "improving" if diagnostics.get("improvement_rate", 0) > 0.1 else "stable"
    else:
        # Multi-objective health status using persisted hypervolume history (Section 4.2)
        hypervolume_improvement = 0.0
        iterations_without_improvement = 0

        if len(hypervolume_history) >= 2:
            # Compute improvement from last two hypervolume values
            recent_improvement = hypervolume_history[-1] - hypervolume_history[-2]
            hypervolume_improvement = max(0.0, recent_improvement)

            # Count consecutive iterations without significant improvement
            threshold = (
                0.001 * abs(hypervolume_history[-1]) if hypervolume_history[-1] != 0 else 0.001
            )
            for i in range(len(hypervolume_history) - 1, 0, -1):
                improvement = hypervolume_history[i] - hypervolume_history[i - 1]
                if improvement < threshold:
                    iterations_without_improvement += 1
                else:
                    break
        elif len(results) >= 2:
            # Fallback to simple estimate if no history
            hypervolume_improvement = 0.1 if diagnostics.get("hypervolume", 0) > 0 else 0.0

        health_status, warnings = determine_health_status(
            n_results=len(results),
            hypervolume_improvement=hypervolume_improvement,
            model_correlation=model_correlation,
            iterations_without_improvement=iterations_without_improvement,
        )

        # Compute progress status using stored hypervolume history
        hv_history = (
            hypervolume_history if hypervolume_history else [diagnostics.get("hypervolume", 0)]
        )
        progress_status = determine_progress_status(hv_history)

    diagnostics["health_status"] = health_status
    diagnostics["warnings"] = warnings
    diagnostics["progress_status"] = progress_status
    diagnostics["hypervolume_history"] = hypervolume_history


# =============================================================================
# Agent Usability Diagnostics (v2.4)
# =============================================================================


def _compute_uncertainty_trends(
    suggestions: list[Suggestion],
    diagnostics: dict[str, Any],
) -> None:
    """Compute uncertainty trends from suggestion history.

    Modifies diagnostics dict in place.
    """
    # Collect uncertainty values from suggestion provenance
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
    opt_spec: Any,
    diagnostics: dict[str, Any],
) -> None:
    """Compute exploration/exploitation balance metrics.

    Modifies diagnostics dict in place.
    """
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

        # Get bounds
        bounds = get_bounds_tensor(opt_spec)

        # Compute metrics
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
    except Exception as e:
        logger.debug("Exploration/exploitation metrics computation failed: %s", e)
        diagnostics["exploration_exploitation"] = None


def _compute_hyperparameters(
    model: Any,
    param_names: list[str],
    diagnostics: dict[str, Any],
) -> None:
    """Extract and expose GP hyperparameters.

    Modifies diagnostics dict in place.
    """
    try:
        hp_info = extract_hyperparameters(model, param_names)
        diagnostics["hyperparameters"] = {
            "lengthscales": hp_info.lengthscales,
            "noise_variance": hp_info.noise_variance,
            "output_scale": hp_info.output_scale,
            "kernel_type": hp_info.kernel_type,
            "model_type": hp_info.model_type,
            "interpretation": _interpret_lengthscales(hp_info.lengthscales),
        }
    except Exception as e:
        logger.debug("Hyperparameter extraction failed: %s", e)
        diagnostics["hyperparameters"] = None


def _interpret_lengthscales(lengthscales: dict[str, float]) -> str:
    """Provide agent-friendly interpretation of lengthscales."""
    if not lengthscales:
        return "No lengthscale information available."

    # Sort by importance (smaller lengthscale = more important)
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


def _compute_constraint_satisfaction_metrics(
    results: list[Result],
    spec: CampaignSpec,
    diagnostics: dict[str, Any],
) -> None:
    """Compute constraint satisfaction rate over time.

    Modifies diagnostics dict in place.
    """
    if not spec.constraints:
        diagnostics["constraint_satisfaction"] = None
        return

    try:
        # Convert results to dict format
        result_dicts = [r.parameter_values for r in results]

        # Convert constraints to dict format
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
    except Exception as e:
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


def _compute_suggestion_diversity_metrics(
    suggestions: list[Suggestion],
    opt_spec: Any,
    diagnostics: dict[str, Any],
) -> None:
    """Compute diversity metrics for recent suggestions.

    Modifies diagnostics dict in place.
    """
    pending = [s for s in suggestions if s.status == SuggestionStatus.PENDING]
    if len(pending) < 2:
        diagnostics["suggestion_diversity"] = None
        return

    try:
        # Build suggestion tensor
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
    except Exception as e:
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


# =============================================================================
# Convergence Detection (Section 1.4)
# =============================================================================


def _compute_convergence_diagnostics(
    spec: CampaignSpec,
    is_single_objective: bool,
    hypervolume_history: list[float],
    diagnostics: dict[str, Any],
) -> None:
    """Compute convergence/early stopping detection metrics.

    Section 1.4 of Implementation Plan - Early Stopping Detection.
    Section 4.2: Now uses persisted hypervolume_history for multi-objective.

    Args:
        spec: Campaign specification
        is_single_objective: Whether this is single-objective optimization
        hypervolume_history: Stored hypervolume history from Campaign
        diagnostics: Dictionary to modify in place
    """
    try:
        if is_single_objective:
            # Use improvement history for single-objective
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
        else:
            # Use persisted hypervolume history for multi-objective (Section 4.2)
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
                # Have some data but not enough for convergence detection
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
    except Exception as e:
        logger.debug("Convergence detection failed: %s", e)
        diagnostics["convergence"] = None


# =============================================================================
# Next Action Recommendation (v3.3)
# =============================================================================


def _compute_next_action_recommendation(
    diagnostics: dict[str, Any],
    n_pending_suggestions: int,
    campaign_status: str,
) -> None:
    """Compute proactive next action recommendation for agents.

    This helps agents decide what to do next without manual interpretation.

    Args:
        diagnostics: Dictionary to modify in place
        n_pending_suggestions: Number of pending suggestions
        campaign_status: Current campaign status
    """
    # Extract key metrics
    n_results = diagnostics.get("n_results", 0)
    health_status = diagnostics.get("health_status", "unknown")
    convergence = diagnostics.get("convergence", {})
    converged = convergence.get("converged", False)
    outliers = diagnostics.get("outliers", {})
    outlier_count = outliers.get("count", 0) if outliers else 0

    # Determine recommendation based on state
    action: str
    reason: str
    urgency: str

    # Check campaign status first
    if campaign_status in ("paused", "completed", "failed"):
        action = "review_campaign_status"
        reason = f"Campaign is {campaign_status}. Resume or create new campaign if needed."
        urgency = "low"
    # Check convergence
    elif converged:
        action = "consider_stopping"
        convergence_reason = convergence.get("reason", "Optimization has converged")
        reason = f"{convergence_reason}. Consider terminating the campaign."
        urgency = "normal"
    # Check for outliers that need attention
    elif outlier_count > 0 and n_results > 5:
        action = "review_outliers"
        reason = f"Detected {outlier_count} potential outlier(s). Verify measurements for errors."
        urgency = "normal"
    # Check health status
    elif health_status == "critical":
        action = "investigate_issues"
        warnings = diagnostics.get("warnings", [])
        reason = f"Campaign health is critical. Issues: {warnings[:2] if warnings else 'Unknown'}"
        urgency = "high"
    elif health_status == "warning":
        action = "monitor_progress"
        reason = "Campaign health has warnings. Continue but monitor closely."
        urgency = "normal"
    # Check pending suggestions
    elif n_pending_suggestions > 0:
        action = "bo_submit_results"
        reason = f"Campaign has {n_pending_suggestions} pending suggestion(s) awaiting results."
        urgency = "normal"
    # Need more suggestions
    elif n_results == 0:
        action = "bo_generate_suggestions"
        reason = "No results yet. Generate initial suggestions to start optimization."
        urgency = "normal"
    else:
        action = "bo_generate_suggestions"
        reason = f"Campaign healthy with {n_results} results. Ready for next batch of suggestions."
        urgency = "normal"

    diagnostics["next_action_recommendation"] = {
        "action": action,
        "reason": reason,
        "urgency": urgency,
    }


# =============================================================================
# Outlier Detection (Section 1.3)
# =============================================================================


def _compute_outlier_diagnostics(
    spec: CampaignSpec,
    results: list[Result],
    opt_spec: Any,
    diagnostics: dict[str, Any],
) -> None:
    """Detect potential outliers in the results.

    Section 1.3 of Implementation Plan - Outlier Detection.

    Modifies diagnostics dict in place.
    """
    if len(results) < 5:
        diagnostics["outliers"] = None
        return

    try:
        # Prepare training data
        objective_names = [o.name for o in spec.objectives]

        # Build train_x tensor
        x_list = []
        for r in results:
            x_encoded = encode_categorical(r.parameter_values, opt_spec)
            x_list.append(x_encoded)
        train_x = torch.stack(x_list)

        # Build train_y tensor
        y_list = []
        for r in results:
            y = [r.objective_values[name] for name in objective_names]
            y_list.append(torch.tensor(y, dtype=torch.double))
        train_y = torch.stack(y_list)

        # Negate maximization objectives
        minimize_mask = torch.tensor([o.is_minimize for o in spec.objectives], dtype=torch.bool)
        train_y_bo = train_y.clone()
        train_y_bo[:, ~minimize_mask] = -train_y_bo[:, ~minimize_mask]

        # Get bounds
        bounds = get_bounds_tensor(opt_spec)

        # Detect outliers
        outliers = detect_outliers(
            train_x=train_x,
            train_y=train_y_bo,
            bounds=bounds,
            objective_names=objective_names,
        )

        if outliers:
            outlier_info = []
            for o in outliers:
                result = results[o.index]
                outlier_info.append(
                    {
                        "result_index": o.index,
                        "result_id": str(result.id),
                        "standardized_error": round(o.standardized_error, 2),
                        "actual_value": round(o.actual_value, 4),
                        "predicted_value": round(o.predicted_value, 4),
                        "objective": o.objective_name,
                        "parameter_values": result.parameter_values,
                    }
                )

            diagnostics["outliers"] = {
                "count": len(outliers),
                "outlier_results": outlier_info,
                "recommendation": (
                    f"Detected {len(outliers)} potential outlier(s). "
                    "These results deviate significantly from model predictions. "
                    "Consider verifying these measurements for errors."
                ),
            }
        else:
            diagnostics["outliers"] = {
                "count": 0,
                "outlier_results": [],
                "recommendation": "No outliers detected. Results appear consistent.",
            }
    except Exception as e:
        logger.debug("Outlier detection failed: %s", e)
        diagnostics["outliers"] = None


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
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(
                f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed"
            ),
        )

    requested = ALL_SECTIONS if sections is None else frozenset(sections)
    invalid_sections = requested - ALL_SECTIONS
    if invalid_sections:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(
                f"Invalid sections: {sorted(invalid_sections)}. Valid: {sorted(ALL_SECTIONS)}"
            ),
        )

    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        logger.warning("Invalid campaign_id format: %s", campaign_id)
        return make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    return verbosity_level, campaign_uuid, requested


async def get_diagnostics_operation(
    campaign_id: str,
    use_cache: bool = True,
    verbosity: str = "standard",
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Compute diagnostic information for a campaign.

    Protocol-neutral operation used by MCP tools and REST routes.

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
            cached = diagnostics_cache.get(cache_key)
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
            diagnostics_cache.set(cache_key, diagnostics)

        return format_diagnostics_response(diagnostics, verbosity_level)


def _compute_sections(
    requested: frozenset[str],
    spec: CampaignSpec,
    results: list[Result],
    all_suggestions: list[Suggestion],
    pending_suggestions: list[Suggestion],
    campaign: Any,
) -> dict[str, Any]:
    """Compute only the requested diagnostic sections."""
    is_single_objective = len(spec.objectives) == 1

    diagnostics: dict[str, Any] = {
        "success": True,
        "campaign_status": campaign.status.value,
        "iteration": campaign.iteration,
        "n_results": len(results),
        "n_pending_suggestions": len(pending_suggestions),
        "errors": [],
    }

    # Objectives section (cheap)
    if "objectives" in requested or "health" in requested:
        if is_single_objective:
            _compute_single_objective_diagnostics(spec, results, diagnostics)
        else:
            _compute_multi_objective_diagnostics(spec, results, diagnostics)
        diagnostics["objective_ranges"] = _compute_objective_ranges(spec, results)
        diagnostics["model_info"] = _get_model_info(spec, is_single_objective)

    # Model section (expensive — involves GP fitting)
    model_correlation = 0.5
    if "model" in requested:
        model_correlation = _compute_model_diagnostics(
            spec, results, is_single_objective, diagnostics
        )

    # Health section
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
            n_pending_suggestions=len(pending_suggestions),
            campaign_status=campaign.status.value,
        )

    # Suggestions section (moderate — needs encoding)
    if "suggestions" in requested:
        opt_spec = campaign_spec_to_optimization_spec(spec)
        _compute_uncertainty_trends(all_suggestions, diagnostics)
        _compute_exploration_exploitation(all_suggestions, results, spec, opt_spec, diagnostics)
        _compute_suggestion_diversity_metrics(all_suggestions, opt_spec, diagnostics)
        _compute_hyperparameters_section(spec, results, is_single_objective, diagnostics)

    # Constraints section (cheap)
    if "constraints" in requested:
        _compute_constraint_satisfaction_metrics(results, spec, diagnostics)

    # Convergence section (cheap — uses stored history)
    if "convergence" in requested:
        _compute_convergence_diagnostics(
            spec,
            is_single_objective,
            campaign.hypervolume_history,
            diagnostics,
        )

    # Outliers section (expensive — needs model fitting)
    if "outliers" in requested:
        opt_spec = campaign_spec_to_optimization_spec(spec)
        _compute_outlier_diagnostics(spec, results, opt_spec, diagnostics)

    return diagnostics


def _compute_hyperparameters_section(
    spec: CampaignSpec,
    results: list[Result],
    is_single_objective: bool,
    diagnostics: dict[str, Any],
) -> None:
    """Compute hyperparameter visibility."""
    n_params = len(spec.parameters)
    min_data_for_model = max(3, 2 * n_params)
    if len(results) < min_data_for_model:
        diagnostics["hyperparameters"] = None
        return

    try:
        opt_spec = campaign_spec_to_optimization_spec(spec)
        param_names = [p.name for p in spec.parameters]
        train_x, train_y, _, _ = _prepare_training_data(spec, results, opt_spec)
        bounds = get_bounds_tensor(opt_spec)

        if is_single_objective:
            model = create_and_fit_single_task_model(
                train_x,
                train_y,
                bounds,
                use_input_warping=spec.use_input_warping,
            )
        else:
            model = create_and_fit_model(
                train_x,
                train_y,
                bounds,
                use_input_warping=spec.use_input_warping,
            )
        _compute_hyperparameters(model, param_names, diagnostics)
    except Exception as e:
        logger.debug("Model fitting for hyperparameters failed: %s", e)
        diagnostics["hyperparameters"] = None
