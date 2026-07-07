"""Posterior predictive checks for GP model validation.

This module provides functions to validate that the GP model's assumptions
are met. GPs assume normally distributed residuals, and violations indicate
model misspecification.

Section 3.4 - Missing Trust-Building Features

References:
    - Gelman et al. "Bayesian Data Analysis" Ch. 6 (Model Checking)
    - Rasmussen & Williams "GPML" Ch. 2.2 (Prediction with GPs)
    - BoTorch Model Diagnostics: https://botorch.org/docs/models/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import torch
from botorch.models import ModelListGP, SingleTaskGP
from scipy import stats as scipy_stats
from torch import Tensor

from bo_engine.constants import (
    POSTERIOR_CHECK_KURTOSIS_THRESHOLD,
    POSTERIOR_CHECK_NORMALITY_ALPHA,
    POSTERIOR_CHECK_SKEWNESS_THRESHOLD,
    SAFE_DIVISION_EPSILON,
)
from bo_engine.cross_validation import compute_exact_loo_moments, matches_model_training_data
from bo_engine.device import get_device, get_dtype

logger = logging.getLogger(__name__)


@dataclass
class NormalityTestResult:
    """Result of a normality test on residuals.

    Attributes:
        test_name: Name of the statistical test used.
        statistic: Test statistic value.
        p_value: P-value for the test.
        is_normal: Whether the null hypothesis of normality is retained.
        interpretation: Human-readable interpretation.
    """

    test_name: str
    statistic: float
    p_value: float
    is_normal: bool
    interpretation: str


@dataclass
class ResidualAnalysis:
    """Comprehensive analysis of model residuals.

    Attributes:
        standardized_residuals: Tensor of leave-one-out standardized
            residuals (see :func:`compute_standardized_residuals`).
        mean: Mean of standardized residuals (should be ~0).
        std: Standard deviation (should be ~1 for well-specified model).
        skewness: Skewness (should be ~0 for normal).
        kurtosis: Excess kurtosis (should be ~0 for normal).
        max_abs_residual: Largest absolute residual.
        outlier_indices: Indices of potential outliers (|z| > 3).
    """

    standardized_residuals: Tensor
    mean: float
    std: float
    skewness: float
    kurtosis: float
    max_abs_residual: float
    outlier_indices: list[int]


@dataclass
class QQPlotData:
    """Data for creating a Q-Q (quantile-quantile) plot.

    Attributes:
        theoretical_quantiles: Expected quantiles from standard normal.
        sample_quantiles: Observed quantiles from standardized residuals.
        reference_line_intercept: Intercept for reference line.
        reference_line_slope: Slope for reference line.
    """

    theoretical_quantiles: list[float]
    sample_quantiles: list[float]
    reference_line_intercept: float
    reference_line_slope: float


@dataclass
class PosteriorCheckReport:
    """Complete posterior predictive check report.

    Attributes:
        assumptions_met: Whether key model assumptions appear to be met.
        residual_analysis: Detailed residual analysis.
        normality_tests: Results of normality tests.
        qq_plot_data: Data for Q-Q plot visualization.
        warnings: List of assumption violations detected.
        recommendations: Actionable recommendations.
    """

    assumptions_met: bool
    residual_analysis: ResidualAnalysis
    normality_tests: list[NormalityTestResult]
    qq_plot_data: QQPlotData
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def compute_standardized_residuals(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    objective_index: int = 0,
) -> Tensor:
    """Compute leave-one-out standardized residuals for model diagnostics.

    Residuals are (y_i - μ_{-i}) / σ_{-i}, where μ_{-i} and σ²_{-i} are the
    exact LOO predictive moments at fixed hyperparameters (GPML §5.4.2,
    Eqs. 5.10-5.12), evaluated in the model's transformed target space —
    the space in which the GP's Gaussian assumption holds. Because σ²_{-i}
    includes observation noise, these residuals are approximately N(0, 1)
    for a well-specified model; in-sample latent residuals carry no such
    guarantee (they collapse toward 0/0 as the posterior mean interpolates).

    When the LOO downdate is unavailable (a batched fully Bayesian model,
    or the passed data is not — by value — the model's own training data),
    falls back to in-sample residuals at the passed points, standardized by
    the predictive (observation-noise-inclusive) std.

    Args:
        model: Fitted GP model. The LOO path applies when the passed data
            matches the model's training data.
        train_x: Training input data (n x d tensor).
        train_y: Training output data (n tensor or n x 1 tensor).
        objective_index: Which objective (for ModelListGP).

    Returns:
        Tensor of standardized residuals (n,).

    References:
        - Gelman BDA Ch. 6: Standardized residuals for model checking
        - Rasmussen & Williams "GPML" §5.4.2: LOO predictive moments
    """
    device = get_device()
    dtype = get_dtype()

    train_x = train_x.to(device=device, dtype=dtype)
    train_y = train_y.to(device=device, dtype=dtype)

    if train_y.dim() > 1:
        train_y = train_y.squeeze(-1)

    if isinstance(model, ModelListGP):
        sub_model = cast("SingleTaskGP", model.models[objective_index])
    else:
        sub_model = model

    if matches_model_training_data(sub_model, train_x, train_y):
        try:
            loo_mean, loo_var = compute_exact_loo_moments(sub_model)
            loo_std = loo_var.sqrt().clamp(min=SAFE_DIVISION_EPSILON)
            return (sub_model.train_targets - loo_mean) / loo_std
        except (RuntimeError, TypeError, ValueError) as e:
            logger.debug(
                "Exact LOO downdate failed, falling back to in-sample predictive residuals: %s: %s",
                type(e).__name__,
                e,
            )

    # Fallback: in-sample residuals with the predictive (noisy) std
    with torch.no_grad():
        posterior = sub_model.posterior(train_x, observation_noise=True)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)

    std = torch.clamp(std, min=SAFE_DIVISION_EPSILON)

    return (train_y - mean) / std


def analyze_residuals(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    objective_index: int = 0,
) -> ResidualAnalysis:
    """Perform comprehensive residual analysis.

    Args:
        model: Fitted GP model.
        train_x: Training input data.
        train_y: Training output data.
        objective_index: Which objective (for ModelListGP).

    Returns:
        ResidualAnalysis with statistics and outlier detection.
    """
    residuals = compute_standardized_residuals(
        model=model,
        train_x=train_x,
        train_y=train_y,
        objective_index=objective_index,
    )

    # Compute statistics
    r_np = residuals.cpu().numpy()

    mean_val = float(r_np.mean())
    std_val = float(r_np.std())
    skewness = float(scipy_stats.skew(r_np))
    kurtosis = float(scipy_stats.kurtosis(r_np))  # Excess kurtosis
    max_abs = float(abs(r_np).max())

    # Find outliers (|z| > 3)
    outlier_mask = abs(residuals) > 3.0
    outlier_indices = torch.where(outlier_mask)[0].tolist()

    return ResidualAnalysis(
        standardized_residuals=residuals,
        mean=mean_val,
        std=std_val,
        skewness=skewness,
        kurtosis=kurtosis,
        max_abs_residual=max_abs,
        outlier_indices=outlier_indices,
    )


def check_residual_normality(
    residuals: Tensor,
    alpha: float = POSTERIOR_CHECK_NORMALITY_ALPHA,
) -> list[NormalityTestResult]:
    """Test whether residuals follow a normal distribution.

    Applies multiple normality tests and returns results. If most tests
    reject normality, the GP assumptions may be violated.

    Args:
        residuals: Standardized residuals tensor.
        alpha: Significance level for hypothesis tests.

    Returns:
        List of NormalityTestResult from different tests.
    """
    r_np = residuals.cpu().numpy().flatten()
    n = len(r_np)
    results: list[NormalityTestResult] = []

    # Shapiro-Wilk test (best for small samples, n < 50)
    # Shapiro-Wilk has both a lower minimum (n>=3) and an upper sample-size limit (n<=5000).
    if 3 <= n <= 5000:
        stat, p_val = scipy_stats.shapiro(r_np)
        is_normal = p_val > alpha
        interpretation = (
            "Residuals appear normally distributed (Shapiro-Wilk)"
            if is_normal
            else "Residuals may not be normally distributed (Shapiro-Wilk)"
        )
        results.append(
            NormalityTestResult(
                test_name="Shapiro-Wilk",
                statistic=float(stat),
                p_value=float(p_val),
                is_normal=is_normal,
                interpretation=interpretation,
            )
        )

    # D'Agostino-Pearson test (requires n >= 20)
    if n >= 20:
        stat, p_val = scipy_stats.normaltest(r_np)
        is_normal = p_val > alpha
        interpretation = (
            "Residuals pass D'Agostino-Pearson test"
            if is_normal
            else "Significant departure from normality (D'Agostino-Pearson)"
        )
        results.append(
            NormalityTestResult(
                test_name="D'Agostino-Pearson",
                statistic=float(stat),
                p_value=float(p_val),
                is_normal=is_normal,
                interpretation=interpretation,
            )
        )

    # Jarque-Bera test (based on skewness and kurtosis)
    if n >= 8:
        jb_result = scipy_stats.jarque_bera(r_np)
        jb_stat = float(jb_result.statistic)
        jb_pval = float(jb_result.pvalue)
        is_normal = jb_pval > alpha
        interpretation = (
            "Skewness and kurtosis consistent with normality"
            if is_normal
            else "Skewness/kurtosis deviate from normal (Jarque-Bera)"
        )
        results.append(
            NormalityTestResult(
                test_name="Jarque-Bera",
                statistic=jb_stat,
                p_value=jb_pval,
                is_normal=is_normal,
                interpretation=interpretation,
            )
        )

    return results


def compute_qq_plot_data(residuals: Tensor) -> QQPlotData:
    """Compute data for a Q-Q plot of standardized residuals.

    A Q-Q plot compares the quantiles of the sample against theoretical
    quantiles from a standard normal distribution. Points should fall
    on a straight line if residuals are normally distributed.

    Args:
        residuals: Standardized residuals tensor.

    Returns:
        QQPlotData for visualization.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> qq_data = compute_qq_plot_data(residuals)
        >>> plt.scatter(qq_data.theoretical_quantiles, qq_data.sample_quantiles)
        >>> plt.plot([x for x in qq_data.theoretical_quantiles],
        ...          [qq_data.reference_line_slope * x + qq_data.reference_line_intercept
        ...           for x in qq_data.theoretical_quantiles], 'r--')
    """
    r_np = residuals.cpu().numpy().flatten()
    r_sorted = sorted(r_np.tolist())
    n = len(r_sorted)

    # Compute theoretical quantiles (standard normal)
    theoretical = [float(scipy_stats.norm.ppf((i + 0.5) / n)) for i in range(n)]

    # Reference line parameters (first and third quartiles)
    q1_idx = n // 4
    q3_idx = 3 * n // 4

    if q3_idx > q1_idx:
        x1, x2 = theoretical[q1_idx], theoretical[q3_idx]
        y1, y2 = r_sorted[q1_idx], r_sorted[q3_idx]

        if abs(x2 - x1) > 1e-8:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
        else:
            slope = 1.0
            intercept = 0.0
    else:
        slope = 1.0
        intercept = 0.0

    return QQPlotData(
        theoretical_quantiles=theoretical,
        sample_quantiles=r_sorted,
        reference_line_intercept=intercept,
        reference_line_slope=slope,
    )


def run_posterior_checks(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    objective_index: int = 0,
    alpha: float = POSTERIOR_CHECK_NORMALITY_ALPHA,
) -> PosteriorCheckReport:
    """Run comprehensive posterior predictive checks.

    Validates GP model assumptions by analyzing residuals, testing
    normality, and identifying potential issues.

    Args:
        model: Fitted GP model.
        train_x: Training input data.
        train_y: Training output data.
        objective_index: Which objective (for ModelListGP).
        alpha: Significance level for normality tests.

    Returns:
        PosteriorCheckReport with full assessment.

    Example:
        >>> from bo_engine import run_posterior_checks
        >>> report = run_posterior_checks(model, train_x, train_y)
        >>> if not report.assumptions_met:
        ...     for warning in report.warnings:
        ...         print(f"Warning: {warning}")
    """
    # Analyze residuals
    residual_analysis = analyze_residuals(
        model=model,
        train_x=train_x,
        train_y=train_y,
        objective_index=objective_index,
    )

    # Test normality
    normality_tests = check_residual_normality(
        residuals=residual_analysis.standardized_residuals,
        alpha=alpha,
    )

    # Compute Q-Q plot data
    qq_data = compute_qq_plot_data(residual_analysis.standardized_residuals)

    # Identify issues
    warnings: list[str] = []
    recommendations: list[str] = []

    # Check mean (should be ~0)
    if abs(residual_analysis.mean) > 0.2:
        warnings.append(
            f"Residual mean is {residual_analysis.mean:.3f} (expected ~0). "
            "Model may have systematic bias."
        )
        recommendations.append(
            "Consider adding a constant mean function or checking for data issues."
        )

    # Check std (should be ~1)
    if residual_analysis.std > 1.5 or residual_analysis.std < 0.5:
        warnings.append(
            f"Residual std is {residual_analysis.std:.3f} (expected ~1). "
            "Variance may be misestimated."
        )
        recommendations.append(
            "The noise variance may be incorrectly estimated. "
            "Consider more training data or different priors."
        )

    # Check skewness
    if abs(residual_analysis.skewness) > POSTERIOR_CHECK_SKEWNESS_THRESHOLD:
        warnings.append(
            f"Residuals are skewed (skewness={residual_analysis.skewness:.3f}). "
            "Normality assumption may be violated."
        )
        recommendations.append("Consider output warping or a more robust likelihood.")

    # Check kurtosis
    if abs(residual_analysis.kurtosis) > POSTERIOR_CHECK_KURTOSIS_THRESHOLD:
        warnings.append(
            f"Residuals have excess kurtosis ({residual_analysis.kurtosis:.3f}). "
            "Tails are heavier/lighter than normal."
        )
        recommendations.append("Heavy tails may indicate outliers. Consider robust GP methods.")

    # Check outliers
    if residual_analysis.outlier_indices:
        warnings.append(
            f"Found {len(residual_analysis.outlier_indices)} potential outliers "
            f"(indices: {residual_analysis.outlier_indices[:5]}...)."
        )
        recommendations.append("Review flagged observations for data quality issues.")

    # Check normality tests
    failed_tests = [t for t in normality_tests if not t.is_normal]
    if len(failed_tests) > len(normality_tests) / 2:
        warnings.append(
            f"{len(failed_tests)} of {len(normality_tests)} normality tests rejected H0. "
            "Residuals may not be normally distributed."
        )
        recommendations.append(
            "Consider input warping, different kernel, or transformation of outputs."
        )

    # Overall assessment
    # Assumptions met if: mean ~0, std ~1, no severe skew/kurtosis, normality not rejected
    assumptions_met = (
        abs(residual_analysis.mean) <= 0.2
        and 0.5 <= residual_analysis.std <= 1.5
        and abs(residual_analysis.skewness) <= POSTERIOR_CHECK_SKEWNESS_THRESHOLD
        and abs(residual_analysis.kurtosis) <= POSTERIOR_CHECK_KURTOSIS_THRESHOLD
        and len(failed_tests) <= len(normality_tests) / 2
    )

    if assumptions_met and not recommendations:
        recommendations.append(
            "Model assumptions appear to be met. Predictions should be reliable."
        )

    return PosteriorCheckReport(
        assumptions_met=assumptions_met,
        residual_analysis=residual_analysis,
        normality_tests=normality_tests,
        qq_plot_data=qq_data,
        warnings=warnings,
        recommendations=recommendations,
    )


def check_multi_objective_posteriors(
    model: ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    objective_names: list[str] | None = None,
    alpha: float = POSTERIOR_CHECK_NORMALITY_ALPHA,
) -> dict[str, PosteriorCheckReport]:
    """Run posterior checks for each objective in multi-objective model.

    Args:
        model: Fitted ModelListGP.
        train_x: Training input data.
        train_y: Training output data (n x n_obj).
        objective_names: Names for objectives.
        alpha: Significance level for tests.

    Returns:
        Dict mapping objective name to PosteriorCheckReport.
    """
    n_objectives = len(model.models)

    if objective_names is None:
        objective_names = [f"obj_{i}" for i in range(n_objectives)]

    results: dict[str, PosteriorCheckReport] = {}

    for i, name in enumerate(objective_names):
        y_i = train_y[:, i] if train_y.dim() > 1 else train_y
        report = run_posterior_checks(
            model=model,
            train_x=train_x,
            train_y=y_i,
            objective_index=i,
            alpha=alpha,
        )
        results[name] = report

    return results


def get_posterior_check_summary(report: PosteriorCheckReport) -> str:
    """Generate human-readable summary of posterior checks.

    Args:
        report: PosteriorCheckReport to summarize.

    Returns:
        Formatted string summary.
    """
    ra = report.residual_analysis

    lines = [
        f"Assumptions Met: {'Yes' if report.assumptions_met else 'No'}",
        "",
        "Residual Statistics:",
        f"  Mean: {ra.mean:.4f} (expected: 0)",
        f"  Std: {ra.std:.4f} (expected: 1)",
        f"  Skewness: {ra.skewness:.4f}",
        f"  Excess Kurtosis: {ra.kurtosis:.4f}",
        f"  Max |residual|: {ra.max_abs_residual:.4f}",
        f"  Potential outliers: {len(ra.outlier_indices)}",
        "",
        "Normality Tests:",
    ]

    for test in report.normality_tests:
        status = "PASS" if test.is_normal else "FAIL"
        lines.append(f"  {test.test_name}: p={test.p_value:.4f} [{status}]")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in report.warnings)

    if report.recommendations:
        lines.append("")
        lines.append("Recommendations:")
        lines.extend(f"  - {r}" for r in report.recommendations)

    return "\n".join(lines)
