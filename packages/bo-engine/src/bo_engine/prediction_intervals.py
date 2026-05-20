"""Prediction intervals for Bayesian Optimization suggestions.

This module provides functions to compute prediction intervals for suggested
parameter configurations. Users need to know "what's the expected range of
outcomes if I run this experiment?" to make informed decisions.

Section 3.2 - Missing Trust-Building Features

References:
    - Rasmussen & Williams "Gaussian Processes for Machine Learning" (2006) Ch. 2
    - BoTorch Posteriors: https://botorch.org/docs/posteriors/
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from botorch.models import ModelListGP, SingleTaskGP
from scipy import stats as scipy_stats
from torch import Tensor

from bo_engine.constants import (
    PREDICTION_INTERVAL_DEFAULT_LEVELS,
    PREDICTION_INTERVAL_EPSILON,
)
from bo_engine.device import get_device, get_dtype


@dataclass
class PredictionInterval:
    """Prediction interval for a single point and objective.

    Attributes:
        lower: Lower bound of the interval.
        upper: Upper bound of the interval.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        mean: Predicted mean value.
        std: Predicted standard deviation.
    """

    lower: float
    upper: float
    confidence_level: float
    mean: float
    std: float


@dataclass
class SuggestionPrediction:
    """Complete prediction information for a single suggestion.

    Attributes:
        parameters: The suggested parameter values.
        objectives: Dict mapping objective name to prediction intervals.
        expected_improvement: Expected improvement over current best.
        probability_of_improvement: Probability of improving current best.
        risk_assessment: Categorical risk level ("low", "medium", "high").
    """

    parameters: dict[str, float]
    objectives: dict[str, list[PredictionInterval]]
    expected_improvement: float | None = None
    probability_of_improvement: float | None = None
    risk_assessment: str = "unknown"


@dataclass
class MultiObjectivePrediction:
    """Prediction for multi-objective suggestion with trade-off region.

    Attributes:
        parameters: The suggested parameter values.
        objective_predictions: Dict mapping objective name to prediction intervals.
        pareto_probability: Probability that this point is Pareto-optimal.
        trade_off_region: Corners of the predicted trade-off region.
        dominance_info: Information about dominated/non-dominated regions.
    """

    parameters: dict[str, float]
    objective_predictions: dict[str, list[PredictionInterval]]
    pareto_probability: float | None = None
    trade_off_region: dict[str, tuple[float, float]] | None = None
    dominance_info: str = ""


@dataclass
class BatchPredictions:
    """Predictions for a batch of suggestions.

    Attributes:
        suggestions: List of individual suggestion predictions.
        batch_diversity_note: Note about batch diversity.
        overall_expected_improvement: Total expected improvement from batch.
    """

    suggestions: list[SuggestionPrediction]
    batch_diversity_note: str = ""
    overall_expected_improvement: float | None = None


def compute_prediction_intervals(
    model: SingleTaskGP | ModelListGP,
    x: Tensor,
    confidence_levels: list[float] | None = None,
    objective_names: list[str] | None = None,
) -> list[dict[str, list[PredictionInterval]]]:
    """Compute prediction intervals for given points.

    Uses the GP posterior to compute prediction intervals at specified
    confidence levels. For each point and objective, returns intervals
    at multiple confidence levels (e.g., 50%, 90%, 95%).

    Args:
        model: Fitted GP model (SingleTaskGP or ModelListGP).
        x: Points at which to compute intervals (n x d tensor).
        confidence_levels: List of confidence levels (default: [0.5, 0.9, 0.95]).
        objective_names: Names for each objective. Defaults to obj_0, obj_1, etc.

    Returns:
        List of dicts (one per point), each mapping objective name to list
        of PredictionInterval objects (one per confidence level).

    Example:
        >>> from bo_engine import compute_prediction_intervals
        >>> intervals = compute_prediction_intervals(model, x_new)
        >>> for name, pi_list in intervals[0].items():
        ...     for pi in pi_list:
        ...         ci = f"[{pi.lower:.3f}, {pi.upper:.3f}]"
        ...         print(f"{name} {int(pi.confidence_level*100)}%: {ci}")

    References:
        - R&W GPML, Section 2.2 (Prediction with Noise-free Observations)
    """
    device = get_device()
    dtype = get_dtype()

    x = x.to(device=device, dtype=dtype)
    if x.dim() == 1:
        x = x.unsqueeze(0)

    if confidence_levels is None:
        confidence_levels = PREDICTION_INTERVAL_DEFAULT_LEVELS

    n_points = x.shape[0]

    # Get number of objectives
    n_objectives = len(model.models) if isinstance(model, ModelListGP) else 1

    if objective_names is None:
        objective_names = [f"obj_{i}" for i in range(n_objectives)]

    results: list[dict[str, list[PredictionInterval]]] = []

    for i in range(n_points):
        xi = x[i : i + 1]
        point_intervals: dict[str, list[PredictionInterval]] = {}

        for obj_idx, obj_name in enumerate(objective_names):
            # Get posterior for this objective
            with torch.no_grad():
                if isinstance(model, ModelListGP):
                    posterior = model.models[obj_idx].posterior(xi)  # ty: ignore[call-non-callable]
                else:
                    posterior = model.posterior(xi)

                mean = posterior.mean.item()
                std = posterior.variance.sqrt().item()

            # Compute intervals for each confidence level
            intervals: list[PredictionInterval] = []
            for level in confidence_levels:
                # z-score for two-tailed interval
                z_score = float(scipy_stats.norm.ppf((1 + level) / 2))
                lower = mean - z_score * std
                upper = mean + z_score * std

                intervals.append(
                    PredictionInterval(
                        lower=lower,
                        upper=upper,
                        confidence_level=level,
                        mean=mean,
                        std=std,
                    )
                )

            point_intervals[obj_name] = intervals

        results.append(point_intervals)

    return results


def _compute_improvement_metrics(
    intervals: dict[str, list],
    best_value: float | None,
    minimize: bool,
) -> tuple[float | None, float | None, str]:
    """Compute EI, PoI, and risk assessment for a single suggestion.

    Returns (expected_improvement, probability_of_improvement, risk_assessment).
    """
    if best_value is None or len(intervals) == 0:
        return None, None, "unknown"

    first_obj = next(iter(intervals.keys()))
    mean = intervals[first_obj][0].mean
    std = intervals[first_obj][0].std

    if std <= PREDICTION_INTERVAL_EPSILON:
        return None, None, "unknown"

    z = (best_value - mean) / std if minimize else (mean - best_value) / std
    poi = float(scipy_stats.norm.cdf(z))
    ei = _compute_expected_improvement(mean, std, best_value, minimize)

    if poi > 0.7:
        risk = "low"
    elif poi > 0.3:
        risk = "medium"
    else:
        risk = "high"

    return ei, poi, risk


def compute_suggestion_predictions(
    model: SingleTaskGP | ModelListGP,
    suggestions: list[dict[str, float]],
    parameter_names: list[str],
    best_value: float | None = None,
    minimize: bool = True,
    objective_names: list[str] | None = None,
    confidence_levels: list[float] | None = None,
) -> BatchPredictions:
    """Compute complete predictions for a batch of suggestions.

    Provides comprehensive prediction information including intervals,
    expected improvement, and risk assessment.

    Args:
        model: Fitted GP model.
        suggestions: List of parameter dictionaries.
        parameter_names: Names of parameters (for ordering).
        best_value: Current best observed value (for improvement calculation).
        minimize: Whether objective is being minimized.
        objective_names: Names for objectives.
        confidence_levels: Confidence levels for intervals.

    Returns:
        BatchPredictions containing predictions for each suggestion.
    """
    device = get_device()
    dtype = get_dtype()

    if confidence_levels is None:
        confidence_levels = PREDICTION_INTERVAL_DEFAULT_LEVELS

    x = torch.zeros(len(suggestions), len(parameter_names), device=device, dtype=dtype)
    for i, sugg in enumerate(suggestions):
        for j, pname in enumerate(parameter_names):
            x[i, j] = sugg[pname]

    all_intervals = compute_prediction_intervals(
        model=model,
        x=x,
        confidence_levels=confidence_levels,
        objective_names=objective_names,
    )

    suggestion_preds: list[SuggestionPrediction] = []
    for i, sugg in enumerate(suggestions):
        ei, poi, risk = _compute_improvement_metrics(all_intervals[i], best_value, minimize)
        suggestion_preds.append(
            SuggestionPrediction(
                parameters=sugg,
                objectives=all_intervals[i],
                expected_improvement=ei,
                probability_of_improvement=poi,
                risk_assessment=risk,
            )
        )

    total_ei = None
    if all(sp.expected_improvement is not None for sp in suggestion_preds):
        total_ei = sum(sp.expected_improvement for sp in suggestion_preds)

    return BatchPredictions(
        suggestions=suggestion_preds,
        batch_diversity_note=f"Batch contains {len(suggestions)} suggestions.",
        overall_expected_improvement=total_ei,
    )


def compute_multi_objective_predictions(
    model: ModelListGP,
    suggestions: list[dict[str, float]],
    parameter_names: list[str],
    objective_names: list[str],
    pareto_front: Tensor | None = None,
    confidence_levels: list[float] | None = None,
) -> list[MultiObjectivePrediction]:
    """Compute predictions for multi-objective suggestions with trade-off regions.

    For multi-objective problems, shows the predicted region in objective
    space and how it relates to the current Pareto front.

    Args:
        model: Fitted ModelListGP.
        suggestions: List of parameter dictionaries.
        parameter_names: Names of parameters.
        objective_names: Names of objectives.
        pareto_front: Current Pareto front (n_pareto x n_obj tensor).
        confidence_levels: Confidence levels for intervals.

    Returns:
        List of MultiObjectivePrediction objects.
    """
    device = get_device()
    dtype = get_dtype()

    if confidence_levels is None:
        confidence_levels = PREDICTION_INTERVAL_DEFAULT_LEVELS

    # Convert suggestions to tensor
    x = torch.zeros(len(suggestions), len(parameter_names), device=device, dtype=dtype)
    for i, sugg in enumerate(suggestions):
        for j, pname in enumerate(parameter_names):
            x[i, j] = sugg[pname]

    # Get prediction intervals
    all_intervals = compute_prediction_intervals(
        model=model,
        x=x,
        confidence_levels=confidence_levels,
        objective_names=objective_names,
    )

    results: list[MultiObjectivePrediction] = []

    for i, sugg in enumerate(suggestions):
        intervals = all_intervals[i]

        # Compute trade-off region (using 95% intervals)
        trade_off_region: dict[str, tuple[float, float]] = {}
        for obj_name in objective_names:
            # Use 95% interval (last in default list)
            pi = intervals[obj_name][-1]
            trade_off_region[obj_name] = (pi.lower, pi.upper)

        # Estimate probability of being Pareto-optimal
        pareto_prob = None
        dominance_info = ""

        if pareto_front is not None and pareto_front.shape[0] > 0:
            pareto_prob = _estimate_pareto_probability(
                model=model,
                x=x[i : i + 1],
                pareto_front=pareto_front,
            )

            if pareto_prob > 0.5:
                dominance_info = "Likely to extend the Pareto front"
            else:
                dominance_info = "May be dominated by existing Pareto points"

        results.append(
            MultiObjectivePrediction(
                parameters=sugg,
                objective_predictions=intervals,
                pareto_probability=pareto_prob,
                trade_off_region=trade_off_region,
                dominance_info=dominance_info,
            )
        )

    return results


def format_prediction_interval_string(
    intervals: dict[str, list[PredictionInterval]],
    include_mean: bool = True,
) -> str:
    """Format prediction intervals as a human-readable string.

    Args:
        intervals: Dict mapping objective name to list of PredictionInterval.
        include_mean: Whether to include the predicted mean.

    Returns:
        Formatted string representation.
    """
    lines: list[str] = []

    for obj_name, pi_list in intervals.items():
        obj_lines = [f"{obj_name}:"]

        if include_mean and pi_list:
            obj_lines.append(f"  Predicted mean: {pi_list[0].mean:.4f}")
            obj_lines.append(f"  Uncertainty (std): {pi_list[0].std:.4f}")

        for pi in pi_list:
            level_pct = int(pi.confidence_level * 100)
            obj_lines.append(f"  {level_pct}% CI: [{pi.lower:.4f}, {pi.upper:.4f}]")

        lines.extend(obj_lines)

    return "\n".join(lines)


def _compute_expected_improvement(
    mean: float,
    std: float,
    best_value: float,
    minimize: bool,
) -> float:
    """Compute expected improvement.

    Args:
        mean: Predicted mean.
        std: Predicted standard deviation.
        best_value: Current best value.
        minimize: Whether minimizing.

    Returns:
        Expected improvement value.
    """
    if std < PREDICTION_INTERVAL_EPSILON:
        return 0.0

    if minimize:
        z = (best_value - mean) / std
        ei = (best_value - mean) * float(scipy_stats.norm.cdf(z)) + std * float(
            scipy_stats.norm.pdf(z)
        )
    else:
        z = (mean - best_value) / std
        ei = (mean - best_value) * float(scipy_stats.norm.cdf(z)) + std * float(
            scipy_stats.norm.pdf(z)
        )

    return max(0.0, float(ei))


def _estimate_pareto_probability(
    model: ModelListGP,
    x: Tensor,
    pareto_front: Tensor,
    n_samples: int = 100,
) -> float:
    """Estimate probability that a point is Pareto-optimal via Monte Carlo.

    Args:
        model: Fitted ModelListGP.
        x: Point to evaluate (1 x d tensor).
        pareto_front: Current Pareto front (n_pareto x n_obj tensor).
        n_samples: Number of Monte Carlo samples.

    Returns:
        Estimated probability of being Pareto-optimal.
    """
    n_objectives = len(model.models)

    # Sample from posterior
    samples = torch.zeros(n_samples, n_objectives, device=x.device, dtype=x.dtype)

    with torch.no_grad():
        for obj_idx in range(n_objectives):
            posterior = model.models[obj_idx].posterior(x)  # ty: ignore[call-non-callable]
            sample = posterior.rsample(torch.Size([n_samples])).squeeze(-1).squeeze(-1)
            samples[:, obj_idx] = sample

    # Check how many samples are non-dominated by Pareto front
    non_dominated_count = 0

    for s in range(n_samples):
        sample_point = samples[s]
        is_dominated = False

        for p_idx in range(pareto_front.shape[0]):
            pareto_point = pareto_front[p_idx]
            # Assuming minimization: dominated if Pareto point <= sample in all obj
            # and strictly < in at least one
            if torch.all(pareto_point <= sample_point) and torch.any(pareto_point < sample_point):
                is_dominated = True
                break

        if not is_dominated:
            non_dominated_count += 1

    return non_dominated_count / n_samples
