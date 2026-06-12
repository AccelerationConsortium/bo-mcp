"""Convergence and early stopping detection for Bayesian Optimization.

Section 1.4 of Implementation Plan: Detects when optimization has converged
to help users avoid wasting experimental resources.

References:
- Convergence in BO: https://arxiv.org/abs/1906.08878
- Hypervolume improvement tracking: https://botorch.org/docs/multi_objective/
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bo_engine.constants import (
    CONVERGENCE_IMPROVEMENT_THRESHOLD,
    CONVERGENCE_MIN_OBSERVATIONS,
    CONVERGENCE_WINDOW_SIZE,
    ESTIMATE_REMAINING_MAX_ITERATIONS,
    IMPROVEMENT_TOLERANCE_ABSOLUTE,
)
from bo_engine.types import ObjectiveSpec, ObservationData, OptimizationSpec


@dataclass
class ConvergenceReport:
    """Report on optimization convergence status.

    Attributes:
        converged: Whether optimization appears to have converged
        convergence_score: Score from 0 (not converged) to 1 (fully converged)
        reason: Human-readable explanation of convergence status
        avg_improvement: Average improvement rate in recent window
        window_size: Window size used for detection
        iterations_without_improvement: Number of recent iterations without improvement
        recommendation: Suggested action based on convergence status
    """

    converged: bool
    convergence_score: float
    reason: str
    avg_improvement: float
    window_size: int
    iterations_without_improvement: int
    recommendation: str


def _history_scale(history: list[float]) -> float:
    """Compute a robust scale for the history that does not depend on absolute magnitude.

    Uses the inter-quartile range (IQR) of the recorded values; falls back
    to the sample standard deviation when the IQR collapses (rare on
    real-valued metric streams but possible on plateaus); finally falls
    back to the absolute value of the latest reading or
    ``IMPROVEMENT_TOLERANCE_ABSOLUTE``. The scale is only consulted by
    :func:`_step_denominator` when the per-step ``|prev|`` is below the
    absolute floor, so high-magnitude trajectories keep their per-step
    semantics and low-magnitude trajectories get a meaningful scale-based
    normalization (this is the audit's concrete fix; see :func:`detect_convergence`).
    """
    if not history:
        return IMPROVEMENT_TOLERANCE_ABSOLUTE
    sorted_history = sorted(history)
    n = len(sorted_history)
    if n >= 4:
        q1 = sorted_history[n // 4]
        q3 = sorted_history[(3 * n) // 4]
        iqr = q3 - q1
        if iqr > IMPROVEMENT_TOLERANCE_ABSOLUTE:
            return iqr
    if n >= 2:
        mean = sum(history) / n
        variance = sum((v - mean) ** 2 for v in history) / max(n - 1, 1)
        std = variance**0.5
        if std > IMPROVEMENT_TOLERANCE_ABSOLUTE:
            return std
    return max(abs(history[-1]), IMPROVEMENT_TOLERANCE_ABSOLUTE)


def _step_denominator(prev_value: float, history_scale: float) -> float:
    """Return the scale-invariant denominator for one step of the improvement loop.

    Uses ``|prev_value|`` when it is meaningfully above the absolute
    floor; otherwise falls back to the history-level scale (IQR / std).
    This keeps the historical per-step semantics for high-magnitude
    metrics (so existing convergence thresholds calibrated against a
    1.0-scale metric remain meaningful) and switches to a trajectory-
    derived scale only for low-magnitude metrics, which is exactly the
    case the audit called out as silently misbehaving under the old
    ``max(|prev|, ABS_TOL)`` formula.
    """
    if abs(prev_value) > IMPROVEMENT_TOLERANCE_ABSOLUTE:
        return abs(prev_value)
    return history_scale


def detect_convergence(
    metric_history: list[float],
    window_size: int = CONVERGENCE_WINDOW_SIZE,
    improvement_threshold: float = CONVERGENCE_IMPROVEMENT_THRESHOLD,
    min_observations: int = CONVERGENCE_MIN_OBSERVATIONS,
) -> ConvergenceReport:
    """Detect optimization convergence based on metric history.

    Analyzes the improvement rate over recent iterations to determine
    if optimization has reached a point of diminishing returns.

    **Scale-invariant normalization.** Per-iteration deltas are divided by
    a robust scale derived from the full history (IQR → std → abs latest
    → ``IMPROVEMENT_TOLERANCE_ABSOLUTE``). This makes the same campaign
    rescaled by 1000× (e.g. cost in dollars vs cents) converge at the
    same iteration count — the previous formula divided by
    ``abs(prev)``, which scales with the absolute magnitude of the
    metric and therefore returned different "still improving" verdicts
    for the same underlying optimization run.

    Args:
        metric_history: History of optimization metric (e.g., hypervolume, best value)
                       Higher values should indicate better optimization.
        window_size: Number of iterations to consider for recent improvement
        improvement_threshold: Relative improvement below which convergence is detected
        min_observations: Minimum observations before convergence can be detected

    Returns:
        ConvergenceReport with convergence analysis

    Reference:
        Section 1.4 of Implementation Plan - Early Stopping Detection
    """
    n = len(metric_history)

    # Not enough data
    if n < min_observations:
        return ConvergenceReport(
            converged=False,
            convergence_score=0.0,
            reason=f"Insufficient history ({n}/{min_observations} observations required)",
            avg_improvement=0.0,
            window_size=window_size,
            iterations_without_improvement=0,
            recommendation="Continue optimization to gather more data.",
        )

    # Not enough for window comparison
    if n < window_size + 1:
        return ConvergenceReport(
            converged=False,
            convergence_score=0.0,
            reason=f"Insufficient history for window analysis ({n}/{window_size + 1} needed)",
            avg_improvement=0.0,
            window_size=window_size,
            iterations_without_improvement=0,
            recommendation="Continue optimization to enable convergence detection.",
        )

    # Robust scale derived from the full history — only consulted by
    # :func:`_step_denominator` when the per-step ``|prev|`` is below the
    # absolute floor, so high-magnitude trajectories keep their historical
    # per-step relative semantics.
    scale = _history_scale(metric_history)

    # Compute improvements over recent window using the hybrid scale
    # (per-step ``|prev|`` when meaningful, history scale otherwise).
    recent = metric_history[-window_size:]
    improvements = []
    for i in range(1, len(recent)):
        delta = recent[i] - recent[i - 1]
        improvements.append(delta / _step_denominator(recent[i - 1], scale))

    avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

    # Count iterations without meaningful improvement (same hybrid scale).
    iterations_without_improvement = 0
    for i in range(n - 1, 0, -1):
        delta = metric_history[i] - metric_history[i - 1]
        rel_improvement = delta / _step_denominator(metric_history[i - 1], scale)

        if rel_improvement > improvement_threshold:
            break
        iterations_without_improvement += 1

    # Compute convergence score (0 = not converged, 1 = fully converged)
    # Based on how small the average improvement is
    if avg_improvement <= 0:
        convergence_score = 1.0  # No improvement = converged
    elif avg_improvement < improvement_threshold:
        convergence_score = 1.0 - (avg_improvement / improvement_threshold)
    else:
        convergence_score = 0.0

    # Also factor in consecutive non-improving iterations
    stagnation_score = min(1.0, iterations_without_improvement / (2 * window_size))
    convergence_score = max(convergence_score, stagnation_score)

    converged = avg_improvement < improvement_threshold

    # Generate reason and recommendation
    if converged:
        reason = (
            f"Improvement rate {avg_improvement:.4f} is below threshold {improvement_threshold}"
        )
        recommendation = (
            "Optimization has likely converged. Consider: "
            "(1) accepting current best solution, "
            "(2) reviewing if constraints are too tight, or "
            "(3) expanding the parameter search space."
        )
    else:
        reason = "Still improving"
        recommendation = "Continue optimization to find better solutions."

    if iterations_without_improvement >= window_size:
        reason = f"No improvement in {iterations_without_improvement} consecutive iterations"
        recommendation = (
            f"Optimization stagnant for {iterations_without_improvement} iterations. "
            "Consider stopping or trying different acquisition parameters."
        )

    return ConvergenceReport(
        converged=converged,
        convergence_score=convergence_score,
        reason=reason,
        avg_improvement=avg_improvement,
        window_size=window_size,
        iterations_without_improvement=iterations_without_improvement,
        recommendation=recommendation,
    )


def detect_hypervolume_convergence(
    hypervolume_history: list[float],
    window_size: int = CONVERGENCE_WINDOW_SIZE,
    improvement_threshold: float = CONVERGENCE_IMPROVEMENT_THRESHOLD,
    min_observations: int = CONVERGENCE_MIN_OBSERVATIONS,
) -> ConvergenceReport:
    """Detect convergence for multi-objective optimization using hypervolume.

    Wrapper around detect_convergence specifically for hypervolume metric.

    Args:
        hypervolume_history: History of hypervolume values
        window_size: Window size for analysis
        improvement_threshold: Threshold for convergence detection
        min_observations: Minimum observations required

    Returns:
        ConvergenceReport with hypervolume-specific analysis
    """
    return detect_convergence(
        metric_history=hypervolume_history,
        window_size=window_size,
        improvement_threshold=improvement_threshold,
        min_observations=min_observations,
    )


def detect_single_objective_convergence(
    best_value_history: list[float],
    minimize: bool = True,
    window_size: int = CONVERGENCE_WINDOW_SIZE,
    improvement_threshold: float = CONVERGENCE_IMPROVEMENT_THRESHOLD,
    min_observations: int = CONVERGENCE_MIN_OBSERVATIONS,
) -> ConvergenceReport:
    """Detect convergence for single-objective optimization.

    Args:
        best_value_history: History of running best values
        minimize: Whether objective is being minimized
        window_size: Window size for analysis
        improvement_threshold: Threshold for convergence detection
        min_observations: Minimum observations required

    Returns:
        ConvergenceReport with single-objective specific analysis
    """
    # For minimization, we negate so that improvement is positive
    metric_history = [-v for v in best_value_history] if minimize else best_value_history

    return detect_convergence(
        metric_history=metric_history,
        window_size=window_size,
        improvement_threshold=improvement_threshold,
        min_observations=min_observations,
    )


class StoppingReason(StrEnum):
    """Reason a campaign was instructed to stop generating suggestions."""

    BUDGET_EXCEEDED_ITERATIONS = "budget_exceeded_iterations"
    BUDGET_EXCEEDED_OBSERVATIONS = "budget_exceeded_observations"
    CONVERGED = "converged"


@dataclass(frozen=True)
class StoppingDecision:
    """Outcome of the budget / convergence check.

    Attributes:
        should_stop: True iff the suggestion entry point must short-circuit.
        reason: Discriminator describing why the decision was reached. Only
            meaningful when ``should_stop`` is True.
        message: User-facing message for the response envelope.
        details: Structured payload (iteration/observation counts, threshold,
            etc.) — included verbatim in the ``next_action_recommendation``
            response so agent loops can branch on it.
    """

    should_stop: bool
    reason: StoppingReason | None
    message: str
    details: dict[str, object]


def _best_value_history(
    observations: list[ObservationData], objective: ObjectiveSpec
) -> list[float]:
    """Build the running-best trajectory of a single objective.

    Used only by :func:`evaluate_stopping_decision` to feed
    ``detect_single_objective_convergence``. The full history is returned
    (not just the last window) so the detector can decide its own
    minimum-observation gate.
    """
    history: list[float] = []
    running_best: float | None = None
    for obs in observations:
        if objective.name not in obs.objective_values:
            continue
        value = float(obs.objective_values[objective.name])
        is_better = running_best is None or (
            value < running_best if objective.minimize else value > running_best
        )
        if is_better:
            running_best = value
        # ``running_best`` is set on the first observation, so it is no longer
        # ``None`` here -- but the static type checker cannot see that, hence
        # the explicit fallback.
        history.append(running_best if running_best is not None else value)
    return history


def evaluate_stopping_decision(
    spec: OptimizationSpec,
    observations: list[ObservationData],
    next_iteration: int,
) -> StoppingDecision:
    """Decide whether to stop suggestion generation based on spec budgets.

    The check runs in three deterministic stages so the response envelope
    is predictable for agent loops:

    1. ``max_iterations`` cap — compared against the *next* iteration the
       caller is about to start (``campaign.iteration + 1``). Reaching the
       budget short-circuits suggestion generation before any BO work runs.
    2. ``max_observations`` cap — compared against the count of stored
       observations regardless of iteration grouping.
    3. ``convergence_tolerance`` — forwarded to the existing
       :func:`detect_single_objective_convergence` detector for the first
       objective. Multi-objective campaigns are out of scope here because
       hypervolume tracking lives in the diagnostics layer.

    When no budget field is configured the decision is a no-op
    (``should_stop=False``).
    """
    if spec.max_iterations is not None and next_iteration > int(spec.max_iterations):
        return StoppingDecision(
            should_stop=True,
            reason=StoppingReason.BUDGET_EXCEEDED_ITERATIONS,
            message=(
                f"Reached max_iterations={spec.max_iterations}; "
                "campaign has exhausted its iteration budget."
            ),
            details={
                "next_iteration": next_iteration,
                "max_iterations": int(spec.max_iterations),
                "next_action_recommendation": "terminate_campaign",
            },
        )

    n_obs = len(observations)
    if spec.max_observations is not None and n_obs >= int(spec.max_observations):
        return StoppingDecision(
            should_stop=True,
            reason=StoppingReason.BUDGET_EXCEEDED_OBSERVATIONS,
            message=(
                f"Reached max_observations={spec.max_observations}; "
                "campaign has exhausted its observation budget."
            ),
            details={
                "n_observations": n_obs,
                "max_observations": int(spec.max_observations),
                "next_action_recommendation": "terminate_campaign",
            },
        )

    # Engine-level defense in depth: the server domain rejects
    # ``convergence_tolerance`` on multi-objective specs at create time, but
    # callers using the engine directly might bypass that validation. Treat
    # multi-objective specs as a silent no-op rather than picking objective 0.
    if spec.convergence_tolerance is not None and len(spec.objectives) == 1:
        history = _best_value_history(observations, spec.objectives[0])
        report = detect_single_objective_convergence(
            best_value_history=history,
            minimize=spec.objectives[0].minimize,
            improvement_threshold=float(spec.convergence_tolerance),
        )
        if report.converged:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.CONVERGED,
                message=("Convergence detected: " + report.reason + ". " + report.recommendation),
                details={
                    "avg_improvement": report.avg_improvement,
                    "convergence_score": report.convergence_score,
                    "iterations_without_improvement": (report.iterations_without_improvement),
                    "convergence_tolerance": float(spec.convergence_tolerance),
                    "next_action_recommendation": "terminate_campaign",
                },
            )

    return StoppingDecision(
        should_stop=False,
        reason=None,
        message="",
        details={},
    )


def estimate_remaining_iterations(
    metric_history: list[float],
    target_improvement: float,
    max_iterations: int = ESTIMATE_REMAINING_MAX_ITERATIONS,
) -> int | None:
    """Estimate iterations needed to achieve target improvement.

    Based on current improvement rate, estimates how many more iterations
    might be needed. Returns None if improvement is too slow to estimate.

    The per-step improvement rate uses the same scale-invariant denominator
    as :func:`detect_convergence` (:func:`_step_denominator` over
    :func:`_history_scale`), so a campaign rescaled by a constant factor
    (e.g. dollars vs cents) yields the same estimate. The previous
    ``delta / abs(prev)`` divisor was magnitude-dependent and disagreed with
    the converged/not-converged verdict computed right next to it.

    Args:
        metric_history: History of optimization metric
        target_improvement: Target relative improvement from current best
        max_iterations: Maximum iterations to consider

    Returns:
        Estimated iterations needed, or None if cannot estimate
    """
    if len(metric_history) < CONVERGENCE_WINDOW_SIZE:
        return None

    # Robust history scale, consulted by _step_denominator only when a
    # step's |prev| falls below the absolute floor.
    scale = _history_scale(metric_history)
    recent = metric_history[-CONVERGENCE_WINDOW_SIZE:]
    improvements = [
        (recent[i] - recent[i - 1]) / _step_denominator(recent[i - 1], scale)
        for i in range(1, len(recent))
    ]

    if not improvements:
        return None

    avg_improvement = sum(improvements) / len(improvements)

    if avg_improvement <= 0:
        return None  # Not improving

    # Estimate iterations: target_improvement / avg_improvement
    estimated = int(target_improvement / avg_improvement)
    return min(estimated, max_iterations) if estimated > 0 else None
