"""Agent-usability diagnostics: uncertainty trend, hyperparameters, constraints.

Split from :mod:`bo_engine.diagnostics` so the campaign-health module
does not also have to host the per-tool readouts agents consume to make
adjustment decisions (uncertainty trend, exploration/exploitation
metrics, GP hyperparameters, constraint satisfaction history).

The exploration/exploitation metric reuses
:func:`bo_engine.diagnostics.compute_suggestion_diversity` so the two
modules share one diversity definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from botorch.models import ModelListGP, SingleTaskGP
from torch import Tensor

from bo_engine.constants import (
    EXPLOITATION_HEAVY_THRESHOLD,
    EXPLORATION_HEAVY_THRESHOLD,
    EXPLORATION_RATIO_MULTIPLIER,
    NUMERICAL_EPSILON,
    SAFE_DIVISION_EPSILON,
    SATISFACTION_TREND_THRESHOLD,
    UNCERTAINTY_TREND_SLOPE_THRESHOLD,
)

# Equality tolerance for sum_equals feasibility checks. ``1e-6`` matches the
# default used elsewhere (e.g. BayBE's ThresholdCondition for discrete
# equality constraints) and is intentionally looser than NUMERICAL_EPSILON
# because real-world parameter values are rarely meaningful below 1e-6.
_FEASIBILITY_EQUALITY_TOLERANCE = 1e-6


@dataclass
class UncertaintyTrend:
    """Uncertainty trend over iterations.

    Tracks how model uncertainty is evolving to help agents understand
    if the model is becoming more confident over time.
    """

    mean_uncertainty: float
    std_uncertainty: float
    trend: str  # "decreasing", "stable", "increasing"
    slope: float  # Linear regression slope (negative = decreasing)
    history: list[float]  # Uncertainty values over iterations


@dataclass
class ExplorationExploitationMetrics:
    """Exploration vs exploitation balance metrics.

    Helps agents understand whether the optimization is exploring
    new regions or exploiting known good areas.
    """

    exploration_ratio: float  # 0-1, higher = more exploration
    diversity_score: float  # 0-1, higher = more diverse suggestions
    average_distance_to_best: float  # Normalized distance to best point
    balance_assessment: str  # "exploration_heavy", "balanced", "exploitation_heavy"
    recommendation: str  # Agent-friendly recommendation


@dataclass
class HyperparameterInfo:
    """Exposed GP hyperparameters for agent visibility.

    Provides transparency into model configuration for debugging
    and advanced agent decision-making.
    """

    lengthscales: dict[str, float]  # Parameter name -> lengthscale
    noise_variance: float  # Observation noise estimate
    output_scale: float  # Kernel output scale
    kernel_type: str  # e.g., "Matern52ARD"
    model_type: str  # e.g., "SingleTaskGP", "ModelListGP"


@dataclass
class ConstraintSatisfactionMetrics:
    """Constraint satisfaction tracking over time.

    Helps agents monitor if constraints are being respected
    and identify potential feasibility issues.
    """

    satisfaction_rate: float  # 0-1, fraction of feasible points
    recent_satisfaction_rate: float  # 0-1, in last 10 points
    feasible_count: int
    infeasible_count: int
    trend: str  # "improving", "stable", "worsening"


def compute_uncertainty_trend(
    uncertainty_history: list[float],
    window: int = 5,
) -> UncertaintyTrend:
    """Compute uncertainty trend from history.

    Analyzes model uncertainty over iterations to determine if
    the model is becoming more confident.

    Args:
        uncertainty_history: List of mean uncertainties per iteration
        window: Window size for trend analysis (uses last ``window`` points for slope)

    Returns:
        UncertaintyTrend with trend assessment
    """
    if len(uncertainty_history) < 2:
        return UncertaintyTrend(
            mean_uncertainty=uncertainty_history[0] if uncertainty_history else 0.0,
            std_uncertainty=0.0,
            trend="stable",
            slope=0.0,
            history=uncertainty_history,
        )

    mean_unc = sum(uncertainty_history) / len(uncertainty_history)
    variance = sum((u - mean_unc) ** 2 for u in uncertainty_history) / len(uncertainty_history)
    std_unc = variance**0.5

    recent = uncertainty_history[-window:]
    n = len(recent)
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n

    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    slope = numerator / denominator if denominator > NUMERICAL_EPSILON else 0.0

    relative_slope = slope / (mean_unc + NUMERICAL_EPSILON)
    if relative_slope < -UNCERTAINTY_TREND_SLOPE_THRESHOLD:
        trend = "decreasing"
    elif relative_slope > UNCERTAINTY_TREND_SLOPE_THRESHOLD:
        trend = "increasing"
    else:
        trend = "stable"

    return UncertaintyTrend(
        mean_uncertainty=mean_unc,
        std_uncertainty=std_unc,
        trend=trend,
        slope=slope,
        history=uncertainty_history,
    )


def compute_exploration_exploitation_metrics(
    suggestions: Tensor,
    best_point: Tensor | None,
    uncertainties: list[float],
    bounds: Tensor,
) -> ExplorationExploitationMetrics:
    """Compute exploration/exploitation balance metrics.

    Analyzes recent suggestions to determine the balance between
    exploring new regions and exploiting known good areas.

    Args:
        suggestions: Recent suggestion parameter values (n_suggestions, n_params)
        best_point: Current best point (n_params,) or None
        uncertainties: Model uncertainties at suggestion points
        bounds: Parameter bounds (2, n_params)

    Returns:
        ExplorationExploitationMetrics with balance assessment
    """
    # Import locally so a future re-export shuffle does not create a
    # cycle on the diversity helper.
    from bo_engine.diagnostics import compute_suggestion_diversity  # noqa: PLC0415

    diversity = compute_suggestion_diversity(suggestions)

    avg_distance_to_best = 0.0
    if best_point is not None and suggestions.shape[0] > 0:
        ranges = bounds[1] - bounds[0]
        # Replace degenerate (~zero-width) bounds with 1 so the per-dim
        # normalization below does not produce inf/NaN. ``SAFE_DIVISION_EPSILON``
        # is the project-wide "treat as zero for division" threshold.
        ranges = torch.where(ranges < SAFE_DIVISION_EPSILON, torch.ones_like(ranges), ranges)

        normalized_suggestions = (suggestions - bounds[0]) / ranges
        normalized_best = (best_point - bounds[0]) / ranges

        distances = torch.norm(normalized_suggestions - normalized_best, dim=1)
        avg_distance_to_best = distances.mean().item()

    if uncertainties:
        avg_uncertainty = sum(uncertainties) / len(uncertainties)
        exploration_ratio = min(1.0, avg_uncertainty * EXPLORATION_RATIO_MULTIPLIER)
    else:
        exploration_ratio = 0.5

    balance, recommendation = _classify_exploration_balance(exploration_ratio, diversity)

    return ExplorationExploitationMetrics(
        exploration_ratio=exploration_ratio,
        diversity_score=diversity,
        average_distance_to_best=avg_distance_to_best,
        balance_assessment=balance,
        recommendation=recommendation,
    )


def _classify_exploration_balance(
    exploration_ratio: float,
    diversity: float,
) -> tuple[str, str]:
    """Classify the exploration/exploitation balance and return (label, recommendation)."""
    combined_score = (exploration_ratio + diversity) / 2

    if combined_score > EXPLORATION_HEAVY_THRESHOLD:
        return (
            "exploration_heavy",
            "Suggestions are primarily exploring new regions. "
            "If optimization is mature, consider reducing exploration.",
        )
    if combined_score < EXPLOITATION_HEAVY_THRESHOLD:
        return (
            "exploitation_heavy",
            "Suggestions are focused near known good points. "
            "If stuck in local optima, consider increasing exploration.",
        )
    return ("balanced", "Good balance between exploration and exploitation.")


def _extract_gp_kernel_info(
    gp: Any,
) -> tuple[str, Tensor, float, float]:
    """Extract kernel type, lengthscales, noise variance, and output scale from a single GP."""
    covar = gp.covar_module
    kernel = getattr(covar, "base_kernel", covar)
    kernel_type = type(kernel).__name__

    ls = kernel.lengthscale.detach().squeeze()

    noise_variance = 0.0
    if hasattr(gp, "likelihood") and hasattr(gp.likelihood, "noise"):
        noise_variance = float(gp.likelihood.noise.item())

    output_scale = 1.0
    if hasattr(covar, "outputscale"):
        output_scale = float(covar.outputscale.item())

    return kernel_type, ls, noise_variance, output_scale


def _lengthscales_to_dict(ls: Tensor, param_names: list[str]) -> dict[str, float]:
    """Convert a lengthscale tensor to a named dict."""
    result: dict[str, float] = {}
    for i, name in enumerate(param_names):
        if ls.numel() == 1:
            result[name] = round(float(ls.item()), 4)
        elif i < ls.numel():
            result[name] = round(float(ls[i].item()), 4)
    return result


def extract_hyperparameters(
    model: SingleTaskGP | ModelListGP,
    param_names: list[str],
) -> HyperparameterInfo:
    """Extract GP hyperparameters for visibility.

    Provides transparency into model configuration for debugging
    and advanced agent decision-making.

    Args:
        model: Fitted GP model
        param_names: Names of input parameters

    Returns:
        HyperparameterInfo with extracted hyperparameters
    """
    model_type = type(model).__name__

    if isinstance(model, ModelListGP):
        all_info = [_extract_gp_kernel_info(gp) for gp in model.models]
        kernel_type = all_info[-1][0] if all_info else "Unknown"
        all_ls = [info[1] for info in all_info]
        avg_ls = torch.stack(all_ls).mean(dim=0) if all_ls else torch.tensor([])
        lengthscales_dict = _lengthscales_to_dict(avg_ls, param_names)
        noise_values = [info[2] for info in all_info]
        noise_variance = sum(noise_values) / len(noise_values) if noise_values else 0.0
        scale_values = [info[3] for info in all_info]
        output_scale = sum(scale_values) / len(scale_values) if scale_values else 1.0
    else:
        kernel_type, ls, noise_variance, output_scale = _extract_gp_kernel_info(model)
        lengthscales_dict = _lengthscales_to_dict(ls, param_names)

    return HyperparameterInfo(
        lengthscales=lengthscales_dict,
        noise_variance=round(noise_variance, 6),
        output_scale=round(output_scale, 4),
        kernel_type=kernel_type,
        model_type=model_type,
    )


def _check_constraint_feasibility(result: dict[str, float], constraint: dict) -> bool:
    """Check if a single result satisfies a single constraint."""
    ctype = constraint.get("type", "")
    params = constraint.get("parameters", [])
    value = constraint.get("value", 0.0)
    coefficients = constraint.get("coefficients")

    param_values = [result.get(p, 0.0) for p in params]

    if ctype == "sum_equals":
        return abs(sum(param_values) - value) < _FEASIBILITY_EQUALITY_TOLERANCE
    if ctype == "sum_less_than":
        return sum(param_values) <= value
    if ctype == "sum_greater_than":
        return sum(param_values) >= value
    if ctype == "linear" and coefficients:
        weighted_sum = sum(c * v for c, v in zip(coefficients, param_values, strict=True))
        return weighted_sum <= value
    return True


def _compute_satisfaction_trend(
    feasibility_history: list[bool],
    recent_rate: float,
    window: int,
) -> str:
    """Determine the satisfaction trend from history."""
    if len(feasibility_history) < 2 * window:
        return "stable"
    previous = feasibility_history[-2 * window : -window]
    previous_rate = sum(previous) / len(previous)
    if recent_rate > previous_rate + SATISFACTION_TREND_THRESHOLD:
        return "improving"
    if recent_rate < previous_rate - SATISFACTION_TREND_THRESHOLD:
        return "worsening"
    return "stable"


def compute_constraint_satisfaction(
    results: list[dict[str, float]],
    constraints: list[dict],
    window: int = 10,
) -> ConstraintSatisfactionMetrics:
    """Compute constraint satisfaction metrics over time.

    Tracks feasibility rate to help agents identify constraint issues.

    Args:
        results: List of result dicts with parameter values
        constraints: List of constraint definitions
        window: Window for recent satisfaction rate

    Returns:
        ConstraintSatisfactionMetrics
    """
    if not results or not constraints:
        return ConstraintSatisfactionMetrics(
            satisfaction_rate=1.0,
            recent_satisfaction_rate=1.0,
            feasible_count=len(results),
            infeasible_count=0,
            trend="stable",
        )

    feasibility_history = [
        all(_check_constraint_feasibility(result, c) for c in constraints) for result in results
    ]

    feasible_count = sum(feasibility_history)
    infeasible_count = len(feasibility_history) - feasible_count
    satisfaction_rate = feasible_count / len(feasibility_history)

    recent = feasibility_history[-window:]
    recent_satisfaction_rate = sum(recent) / len(recent) if recent else 1.0

    trend = _compute_satisfaction_trend(feasibility_history, recent_satisfaction_rate, window)

    return ConstraintSatisfactionMetrics(
        satisfaction_rate=round(satisfaction_rate, 4),
        recent_satisfaction_rate=round(recent_satisfaction_rate, 4),
        feasible_count=feasible_count,
        infeasible_count=infeasible_count,
        trend=trend,
    )
