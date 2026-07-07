"""Model calibration diagnostics for Bayesian Optimization.

This module provides functions to verify if the GP's uncertainty estimates
are well-calibrated. If the model says "I'm 95% confident the outcome is in
[2, 4]" but outcomes frequently fall outside this range, users can't trust
the uncertainty estimates.

Section 3.3 - Missing Trust-Building Features (CRITICAL FOR TRUST)

References:
    - Kuleshov et al. "Accurate Uncertainties for Deep Learning Using
      Calibrated Regression" ICML 2018
    - Guo et al. "On Calibration of Modern Neural Networks" ICML 2017
    - Rasmussen & Williams "GPML" Ch. 2 (Prediction with GPs)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from botorch.models import ModelListGP, SingleTaskGP
from scipy import stats as scipy_stats
from torch import Tensor

from bo_engine.constants import (
    CALIBRATION_CONFIDENCE_LEVELS,
    CALIBRATION_GOOD_THRESHOLD,
    CALIBRATION_POOR_THRESHOLD,
)
from bo_engine.device import get_device, get_dtype
from bo_engine.models import create_and_fit_single_task_model


@dataclass
class CoverageResult:
    """Coverage result for a single confidence level.

    Attributes:
        confidence_level: The nominal confidence level (e.g., 0.95).
        expected_coverage: Expected fraction of values in interval (equals confidence_level).
        observed_coverage: Actual fraction of values that fell in interval.
        calibration_error: Absolute difference (observed - expected).
        n_in_interval: Count of values in interval.
        n_total: Total count of values.
    """

    confidence_level: float
    expected_coverage: float
    observed_coverage: float
    calibration_error: float
    n_in_interval: int
    n_total: int


@dataclass
class CalibrationReport:
    """Complete calibration assessment for a model.

    Attributes:
        is_well_calibrated: Whether the model passes calibration checks.
        calibration_score: Overall score (1.0 = perfect, 0.0 = worst).
        coverage_results: List of coverage results per confidence level.
        mean_calibration_error: Average absolute calibration error.
        max_calibration_error: Maximum absolute calibration error.
        worst_level: The confidence level with worst calibration.
        recommendation: Actionable recommendation based on assessment.
        warnings: List of calibration-related warnings.
    """

    is_well_calibrated: bool
    calibration_score: float
    coverage_results: list[CoverageResult]
    mean_calibration_error: float
    max_calibration_error: float
    worst_level: float
    recommendation: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class CalibrationCurveData:
    """Data for plotting a calibration curve.

    Attributes:
        expected_coverage: List of expected coverage values.
        observed_coverage: List of observed coverage values.
        confidence_levels: List of confidence levels evaluated.
        area_under_curve: Area between calibration curve and perfect diagonal.
    """

    expected_coverage: list[float]
    observed_coverage: list[float]
    confidence_levels: list[float]
    area_under_curve: float


def compute_calibration_score(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    confidence_levels: list[float] | None = None,
    objective_index: int = 0,
) -> CalibrationReport:
    """Compute model calibration score using coverage analysis.

    Evaluates whether the GP's predicted intervals contain the actual
    values at the expected rates. Well-calibrated models have observed
    coverage matching expected coverage across all confidence levels.

    The compared values are noisy observations, so the intervals are built
    from the posterior predictive (latent function plus observation noise);
    latent-only intervals would systematically under-cover noisy data.

    Args:
        model: Fitted GP model.
        train_x: Training input data (n x d tensor).
        train_y: Training output data (n x 1 or n tensor).
        confidence_levels: List of confidence levels to evaluate.
            Default: [0.5, 0.9, 0.95].
        objective_index: Which objective to evaluate (for ModelListGP).

    Returns:
        CalibrationReport with detailed calibration assessment.

    Example:
        >>> from bo_engine import compute_calibration_score
        >>> report = compute_calibration_score(model, train_x, train_y)
        >>> print(f"Calibration score: {report.calibration_score:.2f}")
        >>> if not report.is_well_calibrated:
        ...     print(f"Warning: {report.recommendation}")

    References:
        - Kuleshov et al. ICML 2018: Introduces coverage-based calibration metrics
    """
    device = get_device()
    dtype = get_dtype()

    train_x = train_x.to(device=device, dtype=dtype)
    train_y = train_y.to(device=device, dtype=dtype)

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    if confidence_levels is None:
        confidence_levels = CALIBRATION_CONFIDENCE_LEVELS

    n_points = train_x.shape[0]

    # Posterior predictive for observations: latent f plus observation noise
    with torch.no_grad():
        if isinstance(model, ModelListGP):
            posterior = model.models[objective_index].posterior(  # ty: ignore[call-non-callable]
                train_x, observation_noise=True
            )
        else:
            posterior = model.posterior(train_x, observation_noise=True)

        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)

    # Compute coverage for each confidence level
    coverage_results: list[CoverageResult] = []

    for level in confidence_levels:
        z = scipy_stats.norm.ppf((1 + level) / 2)
        lower = mean - z * std
        upper = mean + z * std

        # Check how many actual values fall within interval
        actual = train_y.squeeze(-1) if train_y.dim() > 1 else train_y
        in_interval = (actual >= lower) & (actual <= upper)
        n_in = in_interval.sum().item()
        observed = n_in / n_points

        coverage_results.append(
            CoverageResult(
                confidence_level=level,
                expected_coverage=level,
                observed_coverage=observed,
                calibration_error=abs(observed - level),
                n_in_interval=int(n_in),
                n_total=n_points,
            )
        )

    # Compute aggregate metrics
    errors = [cr.calibration_error for cr in coverage_results]
    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    worst_level = coverage_results[errors.index(max_error)].confidence_level

    # Calibration score: 1 - mean_error (higher is better)
    calibration_score = 1.0 - mean_error

    # Determine if well-calibrated
    is_well_calibrated = mean_error < CALIBRATION_GOOD_THRESHOLD

    # Generate warnings
    warnings: list[str] = []
    for cr in coverage_results:
        if cr.calibration_error > CALIBRATION_POOR_THRESHOLD:
            level_pct = int(cr.confidence_level * 100)
            obs_pct = int(cr.observed_coverage * 100)
            if cr.observed_coverage < cr.expected_coverage:
                warnings.append(
                    f"{level_pct}% intervals contain only {obs_pct}% of values "
                    f"(model is overconfident)"
                )
            else:
                warnings.append(
                    f"{level_pct}% intervals contain {obs_pct}% of values (model is underconfident)"
                )

    # Generate recommendation
    recommendation = _generate_calibration_recommendation(mean_error, coverage_results)

    return CalibrationReport(
        is_well_calibrated=is_well_calibrated,
        calibration_score=calibration_score,
        coverage_results=coverage_results,
        mean_calibration_error=mean_error,
        max_calibration_error=max_error,
        worst_level=worst_level,
        recommendation=recommendation,
        warnings=warnings,
    )


def compute_calibration_curve(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    n_levels: int = 20,
    objective_index: int = 0,
) -> CalibrationCurveData:
    """Compute calibration curve data for visualization.

    The calibration curve plots expected vs observed coverage across
    many confidence levels. A perfectly calibrated model produces
    a diagonal line.

    Args:
        model: Fitted GP model.
        train_x: Training input data (n x d tensor).
        train_y: Training output data (n tensor or n x 1 tensor).
        n_levels: Number of confidence levels to evaluate.
        objective_index: Which objective (for ModelListGP).

    Returns:
        CalibrationCurveData for plotting.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> curve = compute_calibration_curve(model, train_x, train_y)
        >>> plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
        >>> plt.plot(curve.expected_coverage, curve.observed_coverage, 'b-', label='Model')
        >>> plt.xlabel('Expected Coverage')
        >>> plt.ylabel('Observed Coverage')
    """
    # Generate confidence levels from 0.05 to 0.95
    confidence_levels = [0.05 + i * 0.90 / (n_levels - 1) for i in range(n_levels)]

    report = compute_calibration_score(
        model=model,
        train_x=train_x,
        train_y=train_y,
        confidence_levels=confidence_levels,
        objective_index=objective_index,
    )

    expected = [cr.expected_coverage for cr in report.coverage_results]
    observed = [cr.observed_coverage for cr in report.coverage_results]

    # Compute area under curve (deviation from diagonal)
    # Using trapezoidal integration of |observed - expected|
    auc = 0.0
    for i in range(len(expected) - 1):
        dx = expected[i + 1] - expected[i]
        dy_avg = (abs(observed[i] - expected[i]) + abs(observed[i + 1] - expected[i + 1])) / 2
        auc += dx * dy_avg

    return CalibrationCurveData(
        expected_coverage=expected,
        observed_coverage=observed,
        confidence_levels=confidence_levels,
        area_under_curve=auc,
    )


def compute_loo_calibration(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    confidence_levels: list[float] | None = None,
    use_input_warping: bool = False,
) -> CalibrationReport:
    """Compute calibration using leave-one-out cross-validation.

    Uses LOO-CV to get out-of-sample predictions, which provides a
    more honest assessment of calibration on truly unseen data. Held-out
    points are noisy observations, so coverage is evaluated with the
    posterior predictive (observation noise included).

    Args:
        train_x: Training input data (n x d tensor).
        train_y: Training output data (n tensor or n x 1 tensor).
        bounds: Parameter bounds (2 x d tensor).
        confidence_levels: List of confidence levels to evaluate.
        use_input_warping: Whether to use input warping in models.

    Returns:
        CalibrationReport based on LOO-CV predictions.

    Note:
        This is more computationally expensive but more honest than
        using training data predictions.
    """
    device = get_device()
    dtype = get_dtype()

    train_x = train_x.to(device=device, dtype=dtype)
    train_y = train_y.to(device=device, dtype=dtype)
    bounds = bounds.to(device=device, dtype=dtype)

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    if confidence_levels is None:
        confidence_levels = CALIBRATION_CONFIDENCE_LEVELS

    n_points = train_x.shape[0]

    # Store LOO predictions
    loo_means = torch.zeros(n_points, device=device, dtype=dtype)
    loo_stds = torch.zeros(n_points, device=device, dtype=dtype)

    for i in range(n_points):
        # Create LOO dataset
        mask = torch.ones(n_points, dtype=torch.bool, device=device)
        mask[i] = False

        loo_x = train_x[mask]
        loo_y = train_y[mask]

        # Fit model on LOO data
        model = create_and_fit_single_task_model(
            train_x=loo_x,
            train_y=loo_y.squeeze(-1),
            bounds=bounds,
            use_input_warping=use_input_warping,
        )

        # Predict held-out observation (predictive, not latent, interval)
        with torch.no_grad():
            posterior = model.posterior(train_x[i : i + 1], observation_noise=True)
            loo_means[i] = posterior.mean.item()
            loo_stds[i] = posterior.variance.sqrt().item()

    # Compute coverage for each confidence level
    coverage_results: list[CoverageResult] = []
    actual = train_y.squeeze(-1)

    for level in confidence_levels:
        z = scipy_stats.norm.ppf((1 + level) / 2)
        lower = loo_means - z * loo_stds
        upper = loo_means + z * loo_stds

        in_interval = (actual >= lower) & (actual <= upper)
        n_in = in_interval.sum().item()
        observed = n_in / n_points

        coverage_results.append(
            CoverageResult(
                confidence_level=level,
                expected_coverage=level,
                observed_coverage=observed,
                calibration_error=abs(observed - level),
                n_in_interval=int(n_in),
                n_total=n_points,
            )
        )

    # Aggregate metrics
    errors = [cr.calibration_error for cr in coverage_results]
    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    worst_level = coverage_results[errors.index(max_error)].confidence_level

    calibration_score = 1.0 - mean_error
    is_well_calibrated = mean_error < CALIBRATION_GOOD_THRESHOLD

    warnings: list[str] = []
    for cr in coverage_results:
        if cr.calibration_error > CALIBRATION_POOR_THRESHOLD:
            level_pct = int(cr.confidence_level * 100)
            obs_pct = int(cr.observed_coverage * 100)
            if cr.observed_coverage < cr.expected_coverage:
                warnings.append(
                    f"LOO: {level_pct}% intervals contain only {obs_pct}% of held-out values"
                )
            else:
                warnings.append(
                    f"LOO: {level_pct}% intervals contain {obs_pct}% of held-out values"
                )

    recommendation = _generate_calibration_recommendation(mean_error, coverage_results, is_loo=True)

    return CalibrationReport(
        is_well_calibrated=is_well_calibrated,
        calibration_score=calibration_score,
        coverage_results=coverage_results,
        mean_calibration_error=mean_error,
        max_calibration_error=max_error,
        worst_level=worst_level,
        recommendation=recommendation,
        warnings=warnings,
    )


def assess_multi_objective_calibration(
    model: ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    objective_names: list[str] | None = None,
    confidence_levels: list[float] | None = None,
) -> dict[str, CalibrationReport]:
    """Assess calibration for each objective in a multi-objective model.

    Args:
        model: Fitted ModelListGP.
        train_x: Training input data (n x d tensor).
        train_y: Training output data (n x n_obj tensor).
        objective_names: Names for each objective.
        confidence_levels: Confidence levels to evaluate.

    Returns:
        Dict mapping objective name to CalibrationReport.
    """
    n_objectives = len(model.models)

    if objective_names is None:
        objective_names = [f"obj_{i}" for i in range(n_objectives)]

    results: dict[str, CalibrationReport] = {}

    for i, name in enumerate(objective_names):
        report = compute_calibration_score(
            model=model,
            train_x=train_x,
            train_y=train_y[:, i : i + 1] if train_y.dim() > 1 else train_y,
            confidence_levels=confidence_levels,
            objective_index=i,
        )
        results[name] = report

    return results


def get_calibration_summary(report: CalibrationReport) -> str:
    """Generate a human-readable summary of calibration results.

    Args:
        report: CalibrationReport to summarize.

    Returns:
        Formatted string summary.
    """
    lines = [
        f"Calibration Score: {report.calibration_score:.2f} (1.0 = perfect)",
        f"Status: {'Well-calibrated' if report.is_well_calibrated else 'Poorly calibrated'}",
        f"Mean Calibration Error: {report.mean_calibration_error:.3f}",
        f"Worst Level: {int(report.worst_level * 100)}% CI "
        f"(error: {report.max_calibration_error:.3f})",
        "",
        "Coverage by Confidence Level:",
    ]

    for cr in report.coverage_results:
        level_pct = int(cr.confidence_level * 100)
        obs_pct = int(cr.observed_coverage * 100)
        status = "OK" if cr.calibration_error < CALIBRATION_GOOD_THRESHOLD else "WARN"
        lines.append(f"  {level_pct}% CI: {obs_pct}% observed [{status}]")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in report.warnings)

    lines.append("")
    lines.append(f"Recommendation: {report.recommendation}")

    return "\n".join(lines)


def _generate_calibration_recommendation(
    mean_error: float,
    coverage_results: list[CoverageResult],
    is_loo: bool = False,
) -> str:
    """Generate actionable recommendation based on calibration assessment."""
    prefix = "LOO-CV calibration: " if is_loo else ""

    if mean_error < CALIBRATION_GOOD_THRESHOLD:
        return (
            f"{prefix}Model uncertainty estimates are well-calibrated. "
            "Prediction intervals can be trusted."
        )

    # Check for systematic bias
    over_count = sum(1 for cr in coverage_results if cr.observed_coverage < cr.expected_coverage)
    under_count = len(coverage_results) - over_count

    if over_count > under_count * 2:
        return (
            f"{prefix}Model is systematically overconfident (intervals too narrow). "
            "Consider using input warping or checking for outliers."
        )
    if under_count > over_count * 2:
        return (
            f"{prefix}Model is systematically underconfident (intervals too wide). "
            "This is safer for decision-making but may indicate poor model fit."
        )
    return (
        f"{prefix}Calibration is inconsistent across confidence levels. "
        "The model may need more training data or a different kernel."
    )
