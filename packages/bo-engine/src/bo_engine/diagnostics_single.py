"""Single-objective campaign diagnostics.

Split from :mod:`bo_engine.diagnostics` so the running-best trajectory,
improvement-rate computation, and health-status determination for
single-objective campaigns live in one focused module. Multi-objective
diagnostics (Pareto front, hypervolume) remain in the main module;
LOO-CV lives in :mod:`bo_engine.diagnostics_loo`.

These helpers are consumed by the diagnostics tool surface and by
``compute_campaign_health`` in :mod:`bo_engine.diagnostics`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from bo_engine.constants import (
    DIAGNOSTICS_CRITICAL_STAGNATION_ITERATIONS,
    DIAGNOSTICS_MIN_RESULTS,
    DIAGNOSTICS_MIN_RESULTS_FOR_CORRELATION_WARNING,
    DIAGNOSTICS_MIN_RESULTS_FOR_CRITICAL,
    DIAGNOSTICS_MODEL_CORRELATION_WARNING,
    DIAGNOSTICS_MODEL_CORRELATION_WARNING_STATUS,
    DIAGNOSTICS_WARNING_STAGNATION_ITERATIONS,
    STAGNATION_TOLERANCE_RELATIVE,
)
from bo_engine.convergence import _history_scale, _step_denominator


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

    The delta over the window is divided by the scale-invariant step
    denominator shared with :func:`bo_engine.convergence.detect_convergence`
    (``|initial|`` when meaningfully non-zero, otherwise the trajectory's
    robust IQR/std scale). Dividing by a bare ``|initial|`` is the audited
    scale-dependence bug: a micro-scale or zero-crossing objective would
    report a rate that flips classification with the objective's units or
    offset while the convergence verdict next to it stays invariant.

    Args:
        improvement_history: Running best values over iterations
        window: Window size for rate computation

    Returns:
        Improvement rate (positive = improving, near zero = stagnant)
    """
    if len(improvement_history) < 2:
        return 0.0

    if len(improvement_history) < window:
        initial = improvement_history[0]
        final = improvement_history[-1]
    else:
        initial = improvement_history[-window]
        final = improvement_history[-1]

    denominator = _step_denominator(initial, _history_scale(improvement_history))
    return abs(final - initial) / denominator


def _count_stagnant_iterations(
    improvement_history: list[float],
    max_lookback: int,
) -> int:
    """Count consecutive iterations without improvement from the end of history.

    The "no improvement" tolerance is proportional to the trajectory's
    robust scale (shared IQR/std chain from :mod:`bo_engine.convergence`)
    rather than an absolute constant, so the stagnation verdict survives a
    units change: a ~1e-7-scale objective that is actively improving must
    not read as "critically stagnant" while the scale-invariant convergence
    verdict next to it disagrees.
    """
    count = 0
    n = len(improvement_history)
    tolerance = STAGNATION_TOLERANCE_RELATIVE * _history_scale(improvement_history)
    for i in range(1, min(max_lookback + 1, n)):
        diff = abs(improvement_history[-1] - improvement_history[-i - 1])
        if diff < tolerance:
            count += 1
        else:
            break
    return count


def _collect_health_warnings(
    stagnant: int,
    stagnation_threshold: int,
    model_correlation: float,
    n_results: int,
) -> list[str]:
    """Build the warnings list for single-objective health.

    A ``NaN`` ``model_correlation`` means "not measurable" (see
    :func:`bo_engine.diagnostics.compute_rank_correlation`) and emits no
    low-correlation warning.
    """
    warnings: list[str] = []
    if stagnant >= stagnation_threshold:
        warnings.append(
            f"Optimization has not improved in {stagnant} iterations. "
            "Consider: reviewing constraints, expanding search space, or stopping."
        )
    if (
        not math.isnan(model_correlation)
        and model_correlation < DIAGNOSTICS_MODEL_CORRELATION_WARNING
        and n_results >= DIAGNOSTICS_MIN_RESULTS_FOR_CORRELATION_WARNING
    ):
        warnings.append(
            "Model predictions are not matching experimental results (low correlation). "
            "The model may need more data or the problem may not suit BO."
        )
    return warnings


def determine_single_objective_health_status(
    improvement_history: list[float],
    model_correlation: float,
    stagnation_threshold: int = DIAGNOSTICS_CRITICAL_STAGNATION_ITERATIONS,
) -> tuple[str, list[str]]:
    """Determine health status for single-objective optimization.

    ``model_correlation=NaN`` is treated as "unknown" — it never downgrades
    the status, mirroring :func:`bo_engine.diagnostics.determine_health_status`.

    Returns:
        Tuple of (status, warnings)
    """
    n_results = len(improvement_history)
    if n_results < DIAGNOSTICS_MIN_RESULTS:
        return "healthy", ["Collecting initial data - diagnostics will improve with more results"]

    stagnant = _count_stagnant_iterations(improvement_history, stagnation_threshold)
    warnings = _collect_health_warnings(
        stagnant, stagnation_threshold, model_correlation, n_results
    )
    correlation_known = not math.isnan(model_correlation)

    if stagnant >= stagnation_threshold or (
        correlation_known
        and model_correlation < DIAGNOSTICS_MODEL_CORRELATION_WARNING
        and n_results >= DIAGNOSTICS_MIN_RESULTS_FOR_CRITICAL
    ):
        return "critical", warnings
    if stagnant >= DIAGNOSTICS_WARNING_STAGNATION_ITERATIONS or (
        correlation_known and model_correlation < DIAGNOSTICS_MODEL_CORRELATION_WARNING_STATUS
    ):
        return "warning", warnings
    return "healthy", warnings
