"""Diagnostics and metrics for BO campaigns.

Supports:
- Multi-objective: Pareto front, hypervolume, improvement tracking
- Single-objective: Best value tracking, improvement history
- Model quality: LOO cross-validation, calibration, coverage

v1.0.1: Added single-objective diagnostic functions
v1.1: Added LOO cross-validation for model quality assessment
v2.3: Added GPU auto-detection and acceleration
"""

from dataclasses import dataclass
from typing import overload

import torch
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood
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
    IMPROVEMENT_TOLERANCE_ABSOLUTE,
    PROGRESS_IMPROVING_MULTIPLIER,
    PROGRESS_IMPROVING_THRESHOLD,
    PROGRESS_REGRESSING_MULTIPLIER,
    PROGRESS_REGRESSING_THRESHOLD,
)
from bo_engine.device import ensure_device, to_device


@dataclass
class LOOCVMetrics:
    """Leave-one-out cross-validation metrics."""

    rmse: float
    mae: float
    r_squared: float
    mean_standardized_error: float
    per_fold_errors: list[float]
    coverage_95: float = 0.95  # Fraction of points within 95% CI


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
    if minimize:
        pareto_mask = is_non_dominated(-y)
    else:
        pareto_mask = is_non_dominated(y)

    pareto_y = y[pareto_mask]
    return pareto_y, pareto_mask


def compute_hypervolume(
    pareto_y: Tensor,
    ref_point: Tensor,
) -> float:
    """Compute hypervolume indicator for Pareto front.

    Args:
        pareto_y: Pareto front points of shape (n_pareto, n_objectives)
        ref_point: Reference point of shape (n_objectives,)

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
        return float(result.item())  # type: ignore[union-attr]
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
    if previous_hv <= 0:
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
    calibration = (observed_variance / (predicted_variance + 1e-10)).item()

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

    if previous_avg <= 0:
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
        return corr if not (corr != corr) else 0.0  # Handle NaN
    except Exception as e:
        import logging

        logging.getLogger(__name__).debug(
            f"Rank correlation calculation failed with {len(pred_np)} samples: {e!r}"
        )
        return 0.0


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
    elif (
        iterations_without_improvement >= DIAGNOSTICS_WARNING_STAGNATION_ITERATIONS
        or model_correlation < DIAGNOSTICS_MODEL_CORRELATION_WARNING_STATUS
    ):
        return "warning", warnings
    else:
        return "healthy", warnings


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
        return "improving"  # Assume improving at start

    if len(hypervolume_history) < window * 2:
        # Compare just last two values
        if hypervolume_history[-1] > hypervolume_history[-2] * PROGRESS_IMPROVING_MULTIPLIER:
            return "improving"
        elif hypervolume_history[-1] < hypervolume_history[-2] * PROGRESS_REGRESSING_MULTIPLIER:
            return "regressing"
        else:
            return "stagnant"

    # Compare recent window to previous window
    recent_avg = sum(hypervolume_history[-window:]) / window
    previous_avg = sum(hypervolume_history[-2 * window : -window]) / window

    if previous_avg <= 0:
        return "improving"

    improvement_rate = (recent_avg - previous_avg) / previous_avg

    if improvement_rate > PROGRESS_IMPROVING_THRESHOLD:
        return "improving"
    elif improvement_rate < PROGRESS_REGRESSING_THRESHOLD:
        return "regressing"
    else:
        return "stagnant"


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
    if avg_distance > 0:
        ratio = min(1.0, avg_uncertainty / (avg_distance + 0.1))
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
            dist = torch.norm(suggestions[i] - suggestions[j]).item()
            total_distance += dist
            count += 1

    if count == 0:
        return 1.0

    avg_distance = total_distance / count

    # Normalize by expected distance in unit hypercube
    n_dims = suggestions.shape[1]
    expected_distance = (n_dims / 6) ** 0.5  # Rough expected distance in unit cube

    diversity = min(1.0, avg_distance / (expected_distance + 0.1))
    return diversity


# =============================================================================
# LOO Cross-Validation (v1.1)
# =============================================================================


def compute_loo_cv_metrics(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
) -> LOOCVMetrics:
    """Compute leave-one-out cross-validation metrics for model quality.

    LOO-CV provides an unbiased estimate of generalization error by training
    on n-1 points and predicting the held-out point, for all n points.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1) or (n_samples,)
        bounds: Parameter bounds of shape (2, n_dims)

    Returns:
        LOOCVMetrics with RMSE, MAE, R², and per-fold errors
    """
    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    # Ensure train_y is 2D
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    n_samples = train_x.shape[0]

    # Need at least 3 samples for meaningful CV
    if n_samples < 3:
        return LOOCVMetrics(
            rmse=float("nan"),
            mae=float("nan"),
            r_squared=float("nan"),
            mean_standardized_error=float("nan"),
            per_fold_errors=[],
        )

    # Generate LOO CV folds
    cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y)

    # Collect predictions and errors
    predictions = []
    actuals = []
    standardized_errors = []
    per_fold_errors = []

    for fold_idx in range(n_samples):
        train_fold = cv_folds.train_X[fold_idx]
        train_Y_fold = cv_folds.train_Y[fold_idx]
        test_fold = cv_folds.test_X[fold_idx]
        test_Y_fold = cv_folds.test_Y[fold_idx]

        # Skip if not enough training data
        if train_fold.shape[0] < 2:
            continue

        try:
            # Create and fit model on training fold
            from botorch.fit import fit_gpytorch_mll
            from botorch.models.transforms.input import Normalize
            from botorch.models.transforms.outcome import Standardize

            model = SingleTaskGP(
                train_X=train_fold,
                train_Y=train_Y_fold,
                input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Predict on test fold
            model.eval()
            with torch.no_grad():
                posterior = model.posterior(test_fold)
                pred_mean = posterior.mean
                pred_var = posterior.variance

            # Compute error
            error = (pred_mean - test_Y_fold).abs().item()
            per_fold_errors.append(error)
            predictions.append(pred_mean.squeeze().item())
            actuals.append(test_Y_fold.squeeze().item())

            # Standardized error (for calibration check)
            std_err = (pred_mean - test_Y_fold).abs() / (pred_var.sqrt() + 1e-10)
            standardized_errors.append(std_err.item())

        except Exception as e:  # noqa: S112 - intentionally skip failed folds
            # Skip folds that fail to fit, but log for debugging
            import logging

            logging.getLogger(__name__).debug(
                f"LOO-CV fold {fold_idx} failed to fit: {type(e).__name__}: {e}"
            )
            continue

    if len(predictions) < 2:
        return LOOCVMetrics(
            rmse=float("nan"),
            mae=float("nan"),
            r_squared=float("nan"),
            mean_standardized_error=float("nan"),
            per_fold_errors=per_fold_errors,
        )

    # Compute aggregate metrics
    predictions_t = torch.tensor(predictions)
    actuals_t = torch.tensor(actuals)
    errors = (predictions_t - actuals_t).abs()

    rmse = (errors**2).mean().sqrt().item()
    mae = errors.mean().item()

    # R² (coefficient of determination)
    ss_res = ((predictions_t - actuals_t) ** 2).sum()
    ss_tot = ((actuals_t - actuals_t.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / (ss_tot + 1e-10)).item() if ss_tot > 0 else 0.0

    mean_std_error = sum(standardized_errors) / len(standardized_errors)

    return LOOCVMetrics(
        rmse=rmse,
        mae=mae,
        r_squared=r_squared,
        mean_standardized_error=mean_std_error,
        per_fold_errors=per_fold_errors,
    )


@overload
def compute_loo_cv_for_model(
    model: SingleTaskGP,
    train_x: Tensor,
    train_y: Tensor,
) -> LOOCVMetrics: ...


@overload
def compute_loo_cv_for_model(
    model: ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
) -> dict[int, LOOCVMetrics]: ...


def compute_loo_cv_for_model(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
) -> LOOCVMetrics | dict[int, LOOCVMetrics]:
    """Compute LOO-CV metrics for a fitted model.

    Uses BoTorch's batch_cross_validation for efficiency.

    Args:
        model: Fitted GP model
        train_x: Training inputs
        train_y: Training outputs (n_samples, n_objectives)

    Returns:
        LOOCVMetrics for single-objective models, or
        Dictionary mapping objective index to LOOCVMetrics for multi-objective
    """
    if isinstance(model, SingleTaskGP):
        # Single model - return LOOCVMetrics directly
        cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y)
        try:
            cv_results = batch_cross_validation(
                model_cls=SingleTaskGP,
                mll_cls=ExactMarginalLogLikelihood,
                cv_folds=cv_folds,
            )
            # Extract predictions and compute metrics
            pred_mean = cv_results.posterior.mean.squeeze()
            pred_var = cv_results.posterior.variance.squeeze()
            actuals = train_y.squeeze()
            errors = (pred_mean - actuals).abs()

            rmse = (errors**2).mean().sqrt().item()
            mae = errors.mean().item()

            ss_res = ((pred_mean - actuals) ** 2).sum()
            ss_tot = ((actuals - actuals.mean()) ** 2).sum()
            r_squared = 1 - (ss_res / (ss_tot + 1e-10)).item() if ss_tot > 0 else 0.0

            # Compute standardized errors and coverage
            std = pred_var.sqrt().clamp(min=1e-6)
            standardized_errors = errors / std
            mean_std_error = standardized_errors.mean().item()

            # Coverage: fraction within 1.96 std (95% CI)
            within_95ci = standardized_errors < 1.96
            coverage_95 = within_95ci.float().mean().item()

            return LOOCVMetrics(
                rmse=rmse,
                mae=mae,
                r_squared=r_squared,
                mean_standardized_error=mean_std_error,
                per_fold_errors=errors.tolist(),
                coverage_95=coverage_95,
            )
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(
                f"Cross-validation for single-objective model failed: {type(e).__name__}: {e}"
            )
            return LOOCVMetrics(
                rmse=float("nan"),
                mae=float("nan"),
                r_squared=float("nan"),
                mean_standardized_error=float("nan"),
                per_fold_errors=[],
                coverage_95=float("nan"),
            )
    else:
        # ModelListGP - compute CV for each objective
        results: dict[int, LOOCVMetrics] = {}
        for i, _m in enumerate(model.models):
            train_y_i = train_y[:, i : i + 1]
            cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y_i)
            try:
                cv_results = batch_cross_validation(
                    model_cls=SingleTaskGP,
                    mll_cls=ExactMarginalLogLikelihood,
                    cv_folds=cv_folds,
                )
                pred_mean = cv_results.posterior.mean.squeeze()
                pred_var = cv_results.posterior.variance.squeeze()
                actuals = train_y_i.squeeze()
                errors = (pred_mean - actuals).abs()

                rmse = (errors**2).mean().sqrt().item()
                mae = errors.mean().item()

                ss_res = ((pred_mean - actuals) ** 2).sum()
                ss_tot = ((actuals - actuals.mean()) ** 2).sum()
                r_squared = 1 - (ss_res / (ss_tot + 1e-10)).item() if ss_tot > 0 else 0.0

                # Compute standardized errors and coverage
                std = pred_var.sqrt().clamp(min=1e-6)
                standardized_errors = errors / std
                mean_std_error = standardized_errors.mean().item()

                within_95ci = standardized_errors < 1.96
                coverage_95 = within_95ci.float().mean().item()

                results[i] = LOOCVMetrics(
                    rmse=rmse,
                    mae=mae,
                    r_squared=r_squared,
                    mean_standardized_error=mean_std_error,
                    per_fold_errors=errors.tolist(),
                    coverage_95=coverage_95,
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(
                    f"Cross-validation for objective {i} failed: {type(e).__name__}: {e}"
                )
                results[i] = LOOCVMetrics(
                    rmse=float("nan"),
                    mae=float("nan"),
                    r_squared=float("nan"),
                    mean_standardized_error=float("nan"),
                    per_fold_errors=[],
                    coverage_95=float("nan"),
                )

        return results


# =============================================================================
# Single-Objective Diagnostics (v1.0.1)
# =============================================================================


@dataclass
class SingleObjectiveDiagnostics:
    """Diagnostics for single-objective optimization."""

    best_value: float
    best_parameters: dict[str, float]
    n_evaluations: int
    improvement_history: list[float]
    improvement_rate: float
    health_status: str
    warnings: list[str]


def compute_best_value(
    objective_values: list[float],
    minimize: bool = True,
) -> tuple[float, int]:
    """Get best observed value and its index.

    Args:
        objective_values: List of observed objective values
        minimize: If True, best is minimum; else maximum

    Returns:
        Tuple of (best_value, best_index)
    """
    if not objective_values:
        return float("nan"), -1

    if minimize:
        best_idx = int(torch.tensor(objective_values).argmin().item())
    else:
        best_idx = int(torch.tensor(objective_values).argmax().item())

    return objective_values[best_idx], best_idx


def compute_improvement_history(
    objective_values: list[float],
    minimize: bool = True,
) -> list[float]:
    """Compute running best value over iterations.

    Args:
        objective_values: List of observed objective values
        minimize: If True, track running minimum; else running maximum

    Returns:
        List of running best values (same length as input)
    """
    if not objective_values:
        return []

    history = []
    if minimize:
        running_best = float("inf")
        for val in objective_values:
            running_best = min(running_best, val)
            history.append(running_best)
    else:
        running_best = float("-inf")
        for val in objective_values:
            running_best = max(running_best, val)
            history.append(running_best)

    return history


def compute_single_objective_improvement_rate(
    improvement_history: list[float],
    window: int = 5,
) -> float:
    """Compute recent improvement rate for single-objective optimization.

    Args:
        improvement_history: Running best values over iterations
        window: Window size for rate computation

    Returns:
        Improvement rate (positive = improving, near zero = stagnant)
    """
    if len(improvement_history) < 2:
        return 0.0

    if len(improvement_history) < window:
        # Compare first and last
        initial = improvement_history[0]
        final = improvement_history[-1]
    else:
        # Compare window ago to now
        initial = improvement_history[-window]
        final = improvement_history[-1]

    if abs(initial) < 1e-10:
        return 0.0

    return abs(final - initial) / abs(initial)


def determine_single_objective_health_status(
    improvement_history: list[float],
    model_correlation: float,
    stagnation_threshold: int = DIAGNOSTICS_CRITICAL_STAGNATION_ITERATIONS,
) -> tuple[str, list[str]]:
    """Determine health status for single-objective optimization.

    Args:
        improvement_history: Running best values over iterations
        model_correlation: Rank correlation between predictions and actuals
        stagnation_threshold: Iterations without improvement before warning

    Returns:
        Tuple of (status, warnings)
    """
    warnings = []
    n_results = len(improvement_history)

    if n_results < DIAGNOSTICS_MIN_RESULTS:
        return "healthy", ["Collecting initial data - diagnostics will improve with more results"]

    # Check for stagnation
    iterations_without_improvement = 0
    for i in range(1, min(stagnation_threshold + 1, n_results)):
        if (
            abs(improvement_history[-1] - improvement_history[-i - 1])
            < IMPROVEMENT_TOLERANCE_ABSOLUTE
        ):
            iterations_without_improvement += 1
        else:
            break

    if iterations_without_improvement >= stagnation_threshold:
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

    # Determine status
    if iterations_without_improvement >= stagnation_threshold or (
        model_correlation < DIAGNOSTICS_MODEL_CORRELATION_WARNING
        and n_results >= DIAGNOSTICS_MIN_RESULTS_FOR_CRITICAL
    ):
        return "critical", warnings
    elif (
        iterations_without_improvement >= DIAGNOSTICS_WARNING_STAGNATION_ITERATIONS
        or model_correlation < DIAGNOSTICS_MODEL_CORRELATION_WARNING_STATUS
    ):
        return "warning", warnings
    else:
        return "healthy", warnings


# =============================================================================
# Agent Usability Diagnostics (v2.4)
# =============================================================================


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
        window: Window size for trend analysis

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

    # Compute slope using simple linear regression
    n = len(uncertainty_history)
    x_mean = (n - 1) / 2
    y_mean = mean_unc

    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(uncertainty_history))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    slope = numerator / denominator if denominator > 0 else 0.0

    # Determine trend based on slope relative to mean
    relative_slope = slope / (mean_unc + 1e-10)
    if relative_slope < -0.05:
        trend = "decreasing"
    elif relative_slope > 0.05:
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
    # Compute diversity score
    diversity = compute_suggestion_diversity(suggestions)

    # Compute distance to best
    avg_distance_to_best = 0.0
    if best_point is not None and suggestions.shape[0] > 0:
        # Normalize by bounds range
        ranges = bounds[1] - bounds[0]
        ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)

        normalized_suggestions = (suggestions - bounds[0]) / ranges
        normalized_best = (best_point - bounds[0]) / ranges

        distances = torch.norm(normalized_suggestions - normalized_best, dim=1)
        avg_distance_to_best = distances.mean().item()

    # Compute exploration ratio from uncertainties
    if uncertainties:
        # Higher uncertainty at suggestion points = more exploration
        avg_uncertainty = sum(uncertainties) / len(uncertainties)
        # Normalize to 0-1 (assume uncertainty > 0.5 is high)
        exploration_ratio = min(1.0, avg_uncertainty * 2)
    else:
        exploration_ratio = 0.5

    # Combine metrics for balance assessment
    combined_score = (exploration_ratio + diversity) / 2

    if combined_score > 0.65:
        balance = "exploration_heavy"
        recommendation = (
            "Suggestions are primarily exploring new regions. "
            "If optimization is mature, consider reducing exploration."
        )
    elif combined_score < 0.35:
        balance = "exploitation_heavy"
        recommendation = (
            "Suggestions are focused near known good points. "
            "If stuck in local optima, consider increasing exploration."
        )
    else:
        balance = "balanced"
        recommendation = "Good balance between exploration and exploitation."

    return ExplorationExploitationMetrics(
        exploration_ratio=exploration_ratio,
        diversity_score=diversity,
        average_distance_to_best=avg_distance_to_best,
        balance_assessment=balance,
        recommendation=recommendation,
    )


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
    lengthscales_dict: dict[str, float] = {}
    noise_variance = 0.0
    output_scale = 1.0
    kernel_type = "Unknown"
    model_type = type(model).__name__

    if isinstance(model, ModelListGP):
        # Average across objectives for multi-objective
        all_lengthscales = []
        all_noise = []
        all_output_scale = []

        for gp in model.models:
            covar = gp.covar_module
            kernel = getattr(covar, "base_kernel", covar)
            kernel_type = type(kernel).__name__

            ls = kernel.lengthscale.detach().squeeze()  # type: ignore[union-attr]
            all_lengthscales.append(ls)

            # Get noise variance from likelihood
            if hasattr(gp, "likelihood") and hasattr(gp.likelihood, "noise"):
                all_noise.append(gp.likelihood.noise.item())  # type: ignore[union-attr]

            # Get output scale if present
            if hasattr(covar, "outputscale"):
                all_output_scale.append(covar.outputscale.item())  # type: ignore[union-attr]

        # Average lengthscales
        if all_lengthscales:
            avg_ls = torch.stack(all_lengthscales).mean(dim=0)
            for i, name in enumerate(param_names):
                if i < avg_ls.numel():
                    lengthscales_dict[name] = round(float(avg_ls[i].item()), 4)

        if all_noise:
            noise_variance = sum(all_noise) / len(all_noise)
        if all_output_scale:
            output_scale = sum(all_output_scale) / len(all_output_scale)
    else:
        # SingleTaskGP
        covar = model.covar_module
        kernel = getattr(covar, "base_kernel", covar)
        kernel_type = type(kernel).__name__

        ls = kernel.lengthscale.detach().squeeze()  # type: ignore[union-attr]
        for i, name in enumerate(param_names):
            if ls.numel() == 1:
                lengthscales_dict[name] = round(float(ls.item()), 4)
            elif i < ls.numel():
                lengthscales_dict[name] = round(float(ls[i].item()), 4)

        # Get noise variance from likelihood
        if hasattr(model, "likelihood") and hasattr(model.likelihood, "noise"):
            noise_variance = float(model.likelihood.noise.item())  # type: ignore[union-attr]

        # Get output scale if present
        if hasattr(covar, "outputscale"):
            output_scale = float(covar.outputscale.item())  # type: ignore[union-attr]

    return HyperparameterInfo(
        lengthscales=lengthscales_dict,
        noise_variance=round(noise_variance, 6),
        output_scale=round(output_scale, 4),
        kernel_type=kernel_type,
        model_type=model_type,
    )


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

    def check_feasibility(result: dict[str, float], constraint: dict) -> bool:
        """Check if result satisfies a constraint."""
        ctype = constraint.get("type", "")
        params = constraint.get("parameters", [])
        value = constraint.get("value", 0.0)
        coefficients = constraint.get("coefficients")

        param_values = [result.get(p, 0.0) for p in params]

        if ctype == "sum_equals":
            return abs(sum(param_values) - value) < 1e-6
        elif ctype == "sum_less_than":
            return sum(param_values) <= value
        elif ctype == "sum_greater_than":
            return sum(param_values) >= value
        elif ctype == "linear" and coefficients:
            weighted_sum = sum(c * v for c, v in zip(coefficients, param_values, strict=False))
            return weighted_sum <= value
        return True

    feasibility_history = []
    for result in results:
        is_feasible = all(check_feasibility(result, c) for c in constraints)
        feasibility_history.append(is_feasible)

    feasible_count = sum(feasibility_history)
    infeasible_count = len(feasibility_history) - feasible_count

    satisfaction_rate = feasible_count / len(feasibility_history) if feasibility_history else 1.0

    # Recent satisfaction rate
    if len(feasibility_history) >= window:
        recent = feasibility_history[-window:]
    else:
        recent = feasibility_history
    recent_satisfaction_rate = sum(recent) / len(recent) if recent else 1.0

    # Determine trend
    if len(feasibility_history) >= 2 * window:
        previous = feasibility_history[-2 * window : -window]
        previous_rate = sum(previous) / len(previous)

        if recent_satisfaction_rate > previous_rate + 0.1:
            trend = "improving"
        elif recent_satisfaction_rate < previous_rate - 0.1:
            trend = "worsening"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return ConstraintSatisfactionMetrics(
        satisfaction_rate=round(satisfaction_rate, 4),
        recent_satisfaction_rate=round(recent_satisfaction_rate, 4),
        feasible_count=feasible_count,
        infeasible_count=infeasible_count,
        trend=trend,
    )
