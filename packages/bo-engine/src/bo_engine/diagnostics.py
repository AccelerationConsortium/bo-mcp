"""Diagnostics and metrics for BO campaigns.

Supports:
- Multi-objective: Pareto front, hypervolume, improvement tracking
- Single-objective: Best value tracking, improvement history (split into
  :mod:`bo_engine.diagnostics_single`)
- Model quality: LOO cross-validation, calibration, coverage (split into
  :mod:`bo_engine.diagnostics_loo`)
- Agent usability: uncertainty trend, hyperparameter readout, constraint
  satisfaction (split into :mod:`bo_engine.diagnostics_usability`)

v1.0.1: Added single-objective diagnostic functions
v1.1: Added LOO cross-validation for model quality assessment
v2.3: Added GPU auto-detection and acceleration
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from scipy import stats as scipy_stats
from torch import Tensor

from bo_engine.constants import (
    DIAGNOSTICS_CRITICAL_STAGNATION_ITERATIONS,
    DIAGNOSTICS_HYPERVOLUME_DECREASE_WARNING,
    DIAGNOSTICS_MIN_RESULTS,
    DIAGNOSTICS_MIN_RESULTS_FOR_CORRELATION_WARNING,
    DIAGNOSTICS_MIN_RESULTS_FOR_CRITICAL,
    DIAGNOSTICS_MODEL_CORRELATION_WARNING,
    DIAGNOSTICS_MODEL_CORRELATION_WARNING_STATUS,
    DIAGNOSTICS_WARNING_STAGNATION_ITERATIONS,
    EXPECTED_DISTANCE_HYPERCUBE_DIVISOR,
    EXPLORATION_EXPLOITATION_OFFSET,
    FALLBACK_HYPERVOLUME_IMPROVEMENT,
    HYPERVOLUME_STABILITY_THRESHOLD,
    MIN_IMPROVEMENT_RATE,
    NUMERICAL_EPSILON,
    PROGRESS_IMPROVING_THRESHOLD,
    PROGRESS_REGRESSING_THRESHOLD,
    is_zero,
)
from bo_engine.device import ensure_device, to_device

logger = logging.getLogger(__name__)


@dataclass
class LOOCVMetrics:
    """Leave-one-out cross-validation metrics."""

    rmse: float
    mae: float
    r_squared: float
    mean_standardized_error: float
    per_fold_errors: list[float]
    # Fraction of points whose held-out target falls within the 95% predictive
    # interval. Defaults to NaN ("not measured") rather than the perfect 0.95 so
    # callers can distinguish a computed-perfect calibration from an unset field.
    coverage_95: float = float("nan")


def compute_pareto_front(
    y: Tensor,
    minimize: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute Pareto front from objective values.

    By default assumes minimization (BoTorch convention).

    Args:
        y: Objective values of shape (n_samples, n_objectives)
        minimize: If True (default), treat as minimization problem.
            If False, treat as maximization.

    Returns:
        Tuple of:
            - pareto_y: Tensor of shape (n_pareto, n_objectives) containing Pareto-optimal points
            - pareto_mask: Boolean mask of shape (n_samples,) indicating Pareto-optimal points
    """
    y = to_device(y)

    # BoTorch is_non_dominated expects maximization, so we negate for minimization
    pareto_mask = is_non_dominated(-y) if minimize else is_non_dominated(y)

    pareto_y = y[pareto_mask]
    return pareto_y, pareto_mask


def compute_hypervolume(
    pareto_y: Tensor,
    ref_point: Tensor,
) -> float:
    """Compute hypervolume indicator for Pareto front.

    Both ``pareto_y`` and ``ref_point`` are expected in the canonical
    minimization form (lower = better, with maximization columns
    pre-negated by the caller) documented in :mod:`bo_engine.types`.  The
    function negates both internally before passing them to BoTorch's
    maximization-oriented ``Hypervolume`` implementation.

    Args:
        pareto_y: Pareto front points of shape (n_pareto, n_objectives),
            in minimization form
        ref_point: Reference point of shape (n_objectives,), in
            minimization form

    Returns:
        Hypervolume value
    """
    if pareto_y.shape[0] == 0:
        return 0.0

    pareto_y, ref_point = ensure_device(pareto_y, ref_point)

    # BoTorch Hypervolume expects maximization, so negate both
    hv = Hypervolume(ref_point=-ref_point)
    result = hv.compute(-pareto_y)
    # Handle both Tensor and float return types
    if hasattr(result, "item"):
        return float(result.item())  # type: ignore[union-attr]  # ty: ignore[call-non-callable]
    return float(result)


def compute_hypervolume_improvement(
    current_hv: float,
    previous_hv: float,
) -> float:
    """Compute relative hypervolume improvement.

    Args:
        current_hv: Current hypervolume
        previous_hv: Previous hypervolume

    Returns:
        Relative improvement (0.0 if no previous)
    """
    if previous_hv <= NUMERICAL_EPSILON:
        return 0.0
    return (current_hv - previous_hv) / previous_hv


def assess_model_health(
    predictions: Tensor,
    actuals: Tensor,
    uncertainties: Tensor,
) -> dict[str, float]:
    """Assess model health by comparing predictions to actuals.

    Args:
        predictions: Predicted means of shape (n_samples, n_objectives)
        actuals: Actual observations of shape (n_samples, n_objectives)
        uncertainties: Predicted variances of shape (n_samples, n_objectives)

    Returns:
        Dictionary with health metrics:
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - coverage: Fraction of actuals within 2 sigma of predictions
        - calibration: Ratio of observed to predicted variance
    """
    # Compute errors
    errors = predictions - actuals
    squared_errors = errors**2
    abs_errors = errors.abs()

    rmse = squared_errors.mean().sqrt().item()
    mae = abs_errors.mean().item()

    # Compute coverage (fraction within 2 sigma)
    std = uncertainties.sqrt()
    within_2sigma = abs_errors < 2 * std
    coverage = within_2sigma.float().mean().item()

    # Compute calibration (observed variance / predicted variance)
    observed_variance = squared_errors.mean()
    predicted_variance = uncertainties.mean()
    calibration = (observed_variance / (predicted_variance + NUMERICAL_EPSILON)).item()

    return {
        "rmse": rmse,
        "mae": mae,
        "coverage": coverage,
        "calibration": calibration,
    }


def compute_convergence_metric(
    hypervolumes: list[float],
    window: int = 5,
) -> float:
    """Compute convergence metric based on hypervolume history.

    Returns relative improvement in the last `window` iterations
    compared to the previous `window` iterations.

    Args:
        hypervolumes: List of hypervolume values over iterations
        window: Number of iterations to consider

    Returns:
        Convergence metric (near 0 = converged, > 0 = still improving)
    """
    if len(hypervolumes) < 2 * window:
        return 1.0  # Not enough data, assume not converged

    recent = hypervolumes[-window:]
    previous = hypervolumes[-2 * window : -window]

    recent_avg = sum(recent) / len(recent)
    previous_avg = sum(previous) / len(previous)

    if previous_avg <= NUMERICAL_EPSILON:
        return 1.0

    return (recent_avg - previous_avg) / previous_avg


def summarize_pareto_front(
    pareto_y: Tensor,
    objective_names: list[str],
) -> list[dict[str, float]]:
    """Create summary of Pareto front points.

    Args:
        pareto_y: Pareto front of shape (n_pareto, n_objectives)
        objective_names: Names of objectives

    Returns:
        List of dicts, each containing objective values for one Pareto point
    """
    n_objectives = len(objective_names)
    summaries = []

    for i in range(pareto_y.shape[0]):
        point = {}
        for j in range(n_objectives):
            point[objective_names[j]] = pareto_y[i, j].item()
        summaries.append(point)

    return summaries


def compute_rank_correlation(
    predictions: Tensor,
    actuals: Tensor,
) -> float:
    """Compute Spearman rank correlation between predictions and actuals.

    Used to assess whether model predictions match observed ordering.

    Args:
        predictions: Predicted values of shape (n_samples,)
        actuals: Actual values of shape (n_samples,)

    Returns:
        Spearman rank correlation coefficient (-1 to 1)
    """
    if predictions.numel() < 3:
        return 0.0

    pred_np = predictions.detach().cpu().numpy().flatten()
    actual_np = actuals.detach().cpu().numpy().flatten()

    try:
        result = scipy_stats.spearmanr(pred_np, actual_np)
        corr = float(result.statistic)  # type: ignore[union-attr]
    except (RuntimeError, ValueError, TypeError) as e:
        logger.debug("Rank correlation calculation failed with %d samples: %r", len(pred_np), e)
        return 0.0
    return corr if corr == corr else 0.0  # Handle NaN


def determine_health_status(
    n_results: int,
    hypervolume_improvement: float,
    model_correlation: float,
    iterations_without_improvement: int,
) -> tuple[str, list[str]]:
    """Determine overall health status and generate warnings.

    Args:
        n_results: Number of results collected
        hypervolume_improvement: Recent hypervolume improvement rate
        model_correlation: Rank correlation between predictions and actuals
        iterations_without_improvement: Number of iterations without improvement

    Returns:
        Tuple of (status, warnings) where status is 'healthy', 'warning', or 'critical'
    """
    warnings = []

    # Not enough data yet
    if n_results < DIAGNOSTICS_MIN_RESULTS:
        return "healthy", ["Collecting initial data - diagnostics will improve with more results"]

    # Check for critical issues
    if iterations_without_improvement >= DIAGNOSTICS_CRITICAL_STAGNATION_ITERATIONS:
        warnings.append(
            f"Optimization has not improved in {iterations_without_improvement} iterations. "
            "Consider: reviewing constraints, expanding search space, or stopping."
        )

    if (
        model_correlation < DIAGNOSTICS_MODEL_CORRELATION_WARNING
        and n_results >= DIAGNOSTICS_MIN_RESULTS_FOR_CORRELATION_WARNING
    ):
        warnings.append(
            "Model predictions are not matching experimental results (low correlation). "
            "The model may need more data or the problem may not suit BO."
        )

    if (
        hypervolume_improvement < DIAGNOSTICS_HYPERVOLUME_DECREASE_WARNING
        and n_results >= DIAGNOSTICS_MIN_RESULTS_FOR_CORRELATION_WARNING
    ):
        warnings.append("Hypervolume is decreasing. This may indicate model instability.")

    # Determine overall status
    if iterations_without_improvement >= DIAGNOSTICS_CRITICAL_STAGNATION_ITERATIONS or (
        model_correlation < DIAGNOSTICS_MODEL_CORRELATION_WARNING
        and n_results >= DIAGNOSTICS_MIN_RESULTS_FOR_CRITICAL
    ):
        return "critical", warnings
    if (
        iterations_without_improvement >= DIAGNOSTICS_WARNING_STAGNATION_ITERATIONS
        or model_correlation < DIAGNOSTICS_MODEL_CORRELATION_WARNING_STATUS
    ):
        return "warning", warnings
    return "healthy", warnings


def _classify_improvement(rate: float) -> str:
    """Classify an improvement rate as improving, stagnant, or regressing."""
    if rate > PROGRESS_IMPROVING_THRESHOLD:
        return "improving"
    if rate < PROGRESS_REGRESSING_THRESHOLD:
        return "regressing"
    return "stagnant"


def determine_progress_status(
    hypervolume_history: list[float],
    window: int = 3,
) -> str:
    """Determine optimization progress status.

    Args:
        hypervolume_history: List of hypervolume values over iterations
        window: Window size for comparison

    Returns:
        Progress status: 'improving', 'stagnant', or 'regressing'
    """
    if len(hypervolume_history) < 2:
        return "improving"

    if len(hypervolume_history) < window * 2:
        last, prev = hypervolume_history[-1], hypervolume_history[-2]
        rate = (last - prev) / abs(prev) if not is_zero(prev) else 1.0
        return _classify_improvement(rate)

    recent_avg = sum(hypervolume_history[-window:]) / window
    previous_avg = sum(hypervolume_history[-2 * window : -window]) / window

    if previous_avg <= NUMERICAL_EPSILON:
        return "improving"

    improvement_rate = (recent_avg - previous_avg) / previous_avg
    return _classify_improvement(improvement_rate)


def analyze_hypervolume_history(
    hypervolume_history: list[float],
    n_results: int,
    current_hypervolume: float = 0.0,
) -> tuple[float, int]:
    """Summarize a hypervolume trajectory for multi-objective health checks.

    Returns the **signed** last-step delta so
    :func:`determine_health_status` can detect a genuine hypervolume
    decrease (its ``DIAGNOSTICS_HYPERVOLUME_DECREASE_WARNING`` threshold
    is negative, so a clamp to ``>= 0`` would silently suppress the
    warning). Stagnation is reported separately as a count of recent
    iterations whose step was below
    :data:`HYPERVOLUME_STABILITY_THRESHOLD` (scaled by the magnitude of
    the last value so flat-but-large hypervolumes are not mis-flagged).

    When fewer than two history samples exist but enough results have
    accumulated to expect a hypervolume reading, ``current_hypervolume``
    is used as a presence signal: a non-zero value yields the synthetic
    :data:`FALLBACK_HYPERVOLUME_IMPROVEMENT` so the campaign is not
    driven into ``critical`` purely on missing-history grounds.

    Returns:
        ``(recent_delta, iterations_stagnant)`` ready to hand to
        :func:`determine_health_status`. ``recent_delta`` is signed —
        negative when the last step regressed.
    """
    if len(hypervolume_history) >= 2:
        recent_delta = hypervolume_history[-1] - hypervolume_history[-2]
        threshold = (
            HYPERVOLUME_STABILITY_THRESHOLD * abs(hypervolume_history[-1])
            if not math.isclose(hypervolume_history[-1], 0.0, abs_tol=1e-12)
            else HYPERVOLUME_STABILITY_THRESHOLD
        )
        iters_stagnant = 0
        for i in range(len(hypervolume_history) - 1, 0, -1):
            if hypervolume_history[i] - hypervolume_history[i - 1] < threshold:
                iters_stagnant += 1
            else:
                break
        return recent_delta, iters_stagnant

    if n_results >= 2:
        has_hv = current_hypervolume > NUMERICAL_EPSILON
        hv_improvement = FALLBACK_HYPERVOLUME_IMPROVEMENT if has_hv else 0.0
        return hv_improvement, 0

    return 0.0, 0


def compute_single_objective_progress_status(improvement_rate: float) -> str:
    """Classify a single-objective campaign's progress from a scalar rate.

    A rate strictly greater than :data:`MIN_IMPROVEMENT_RATE` is reported as
    ``"improving"``; anything at or below is ``"stable"``. The thresholds
    live in bo-engine so server / API layers cannot diverge on what counts
    as "still making progress" for a single-objective campaign.
    """
    return "improving" if improvement_rate > MIN_IMPROVEMENT_RATE else "stable"


def compute_campaign_health(
    *,
    is_single_objective: bool,
    n_results: int,
    diagnostics: dict[str, Any],
    model_correlation: float,
    hypervolume_history: list[float],
) -> tuple[str, list[str], str]:
    """End-to-end campaign-health computation.

    Combines the existing single- and multi-objective status helpers with
    the hypervolume-history analysis so the server / API layers do not have
    to assemble (status, warnings, progress) themselves — and cannot drift
    in *how* they assemble it. The function does not mutate
    ``diagnostics``; callers persist the returned tuple under their own
    transport keys.

    Args:
        is_single_objective: True for single-objective campaigns.
        n_results: Number of observed results (used by both branches).
        diagnostics: Diagnostics dict from
            :func:`compute_improvement_history` / multi-objective
            counterparts. Reads ``improvement_history`` (single-obj),
            ``improvement_rate`` (single-obj), and ``hypervolume``
            (multi-obj fallback).
        model_correlation: Spearman rank correlation between GP
            predictions and observed objectives.
        hypervolume_history: Hypervolume samples per iteration (newest
            last). Empty list signals no history yet.

    Returns:
        ``(status, warnings, progress_status)`` triple. Statuses are
        ``"healthy"``/``"warning"``/``"critical"`` and progress is
        ``"improving"``/``"stable"`` (single-obj) or
        ``"improving"``/``"stagnant"``/``"regressing"`` (multi-obj).
    """
    if is_single_objective:
        improvement_history = diagnostics.get("improvement_history", [])
        status, warnings = determine_single_objective_health_status(
            improvement_history=improvement_history,
            model_correlation=model_correlation,
        )
        progress = compute_single_objective_progress_status(
            float(diagnostics.get("improvement_rate", 0.0))
        )
        return status, warnings, progress

    hv_improvement, iters_stagnant = analyze_hypervolume_history(
        hypervolume_history,
        n_results=n_results,
        current_hypervolume=float(diagnostics.get("hypervolume", 0.0)),
    )
    status, warnings = determine_health_status(
        n_results=n_results,
        hypervolume_improvement=hv_improvement,
        model_correlation=model_correlation,
        iterations_without_improvement=iters_stagnant,
    )
    hv_for_progress = (
        hypervolume_history if hypervolume_history else [float(diagnostics.get("hypervolume", 0.0))]
    )
    progress = determine_progress_status(hv_for_progress)
    return status, warnings, progress


def compute_exploration_exploitation_ratio(
    suggestion_distances: list[float],
    uncertainties: list[float],
) -> float:
    """Compute exploration vs exploitation ratio.

    Higher values indicate more exploration (diverse suggestions).
    Lower values indicate more exploitation (focused suggestions).

    Args:
        suggestion_distances: Distances between consecutive suggestions
        uncertainties: Model uncertainties at suggestion points

    Returns:
        Exploration/exploitation ratio (0 to 1, higher = more exploration)
    """
    if not suggestion_distances or not uncertainties:
        return 0.5  # Neutral

    avg_distance = sum(suggestion_distances) / len(suggestion_distances)
    avg_uncertainty = sum(uncertainties) / len(uncertainties)

    # Normalize: high distance + high uncertainty = exploration
    # Simple heuristic: if uncertainty is high relative to distance, it's exploration
    if avg_distance > NUMERICAL_EPSILON:
        ratio = min(1.0, avg_uncertainty / (avg_distance + EXPLORATION_EXPLOITATION_OFFSET))
    else:
        ratio = 0.5

    return ratio


def compute_suggestion_diversity(
    suggestions: Tensor,
) -> float:
    """Compute diversity of suggestions (spread in parameter space).

    Args:
        suggestions: Suggestion parameter values of shape (n_suggestions, n_params)

    Returns:
        Diversity score (0 to 1, higher = more diverse)
    """
    if suggestions.shape[0] < 2:
        return 1.0

    # Compute pairwise distances
    n = suggestions.shape[0]
    total_distance = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.linalg.vector_norm(suggestions[i] - suggestions[j]).item()
            total_distance += dist
            count += 1

    if count == 0:
        return 1.0

    avg_distance = total_distance / count

    # Normalize by expected distance in unit hypercube
    n_dims = suggestions.shape[1]
    expected_distance = (n_dims / EXPECTED_DISTANCE_HYPERCUBE_DIVISOR) ** 0.5

    return min(1.0, avg_distance / (expected_distance + EXPLORATION_EXPLOITATION_OFFSET))


# ---------------------------------------------------------------------------
# Re-exports for backwards compatibility with the original god-module API.
# The implementations live in topic-specific modules; importing them here
# keeps ``from bo_engine.diagnostics import X`` working for every existing
# caller while reducing the cognitive load on this file.
# ---------------------------------------------------------------------------

from bo_engine.diagnostics_loo import (  # noqa: E402
    compute_loo_cv_for_model,
    compute_loo_cv_metrics,
)
from bo_engine.diagnostics_single import (  # noqa: E402
    SingleObjectiveDiagnostics,
    compute_best_value,
    compute_improvement_history,
    compute_single_objective_improvement_rate,
    determine_single_objective_health_status,
)
from bo_engine.diagnostics_usability import (  # noqa: E402
    ConstraintSatisfactionMetrics,
    ExplorationExploitationMetrics,
    HyperparameterInfo,
    UncertaintyTrend,
    compute_constraint_satisfaction,
    compute_exploration_exploitation_metrics,
    compute_uncertainty_trend,
    extract_hyperparameters,
)

__all__ = [
    # Data classes
    "ConstraintSatisfactionMetrics",
    "ExplorationExploitationMetrics",
    "HyperparameterInfo",
    "LOOCVMetrics",
    "SingleObjectiveDiagnostics",
    "UncertaintyTrend",
    # Multi-objective + campaign health (defined in this module)
    "analyze_hypervolume_history",
    "assess_model_health",
    # Single-objective diagnostics
    "compute_best_value",
    "compute_campaign_health",
    # Agent-usability diagnostics
    "compute_constraint_satisfaction",
    "compute_convergence_metric",
    "compute_exploration_exploitation_metrics",
    "compute_exploration_exploitation_ratio",
    "compute_hypervolume",
    "compute_hypervolume_improvement",
    "compute_improvement_history",
    # LOO cross-validation
    "compute_loo_cv_for_model",
    "compute_loo_cv_metrics",
    "compute_pareto_front",
    "compute_rank_correlation",
    "compute_single_objective_improvement_rate",
    "compute_single_objective_progress_status",
    "compute_suggestion_diversity",
    "compute_uncertainty_trend",
    "determine_health_status",
    "determine_progress_status",
    "determine_single_objective_health_status",
    "extract_hyperparameters",
    "summarize_pareto_front",
]
