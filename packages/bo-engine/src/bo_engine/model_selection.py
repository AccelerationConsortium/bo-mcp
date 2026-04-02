"""Model Selection and Comparison for Gaussian Processes.

This module provides automatic model comparison using cross-validation scores
to select the best GP configuration for a given dataset.

v2.6: Initial implementation with model comparison framework (Section 2.5)

The Problem:
    The system always uses SingleTaskGP or ModelListGP without considering
    alternatives. Standard GPs may not be optimal for all problem types:
    - Non-stationary objectives → input warping helps
    - Heteroscedastic noise → standard GP assumes homoscedastic noise
    - High noise → might benefit from different kernels

The Solution:
    1. Fit multiple model configurations (standard, warped, different kernels)
    2. Compare using LOO-CV scores
    3. Select the best model and expose results in diagnostics

References:
    - Rasmussen & Williams "GPML" Ch. 5 (Model Selection)
    - Snoek et al. "Input Warping for Bayesian Optimization of
      Non-Stationary Functions" ICML 2014
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize, Warp
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from bo_engine.cross_validation import CVConfig, CVMetrics, compute_loo_cv_optimized
from bo_engine.device import ensure_device

logger = logging.getLogger(__name__)


class KernelType(Enum):
    """Supported kernel types for GP models."""

    MATERN_52 = "matern_52"  # Default, good balance
    MATERN_32 = "matern_32"  # Slightly rougher functions
    MATERN_12 = "matern_12"  # Exponential kernel, rough functions
    RBF = "rbf"  # Infinitely differentiable, very smooth functions


class ModelConfiguration(Enum):
    """Model configuration options."""

    STANDARD = "standard"  # SingleTaskGP with default settings
    WARPED = "warped"  # With Kumaraswamy input warping
    HETEROSCEDASTIC = "heteroscedastic"  # With learned noise (future)


@dataclass
class ModelCandidate:
    """A candidate model configuration.

    Attributes:
        name: Human-readable name
        kernel: Kernel type
        use_warping: Whether to use input warping
        configuration: Model configuration type
        description: Description of when to use this model
    """

    name: str
    kernel: KernelType
    use_warping: bool
    configuration: ModelConfiguration
    description: str


@dataclass
class ModelComparisonResult:
    """Result of model comparison.

    Attributes:
        candidate: The model candidate configuration
        model: Fitted model (if kept)
        cv_metrics: Cross-validation metrics
        log_marginal_likelihood: Log marginal likelihood of the fitted model
        bic: Bayesian Information Criterion
        rank: Rank among compared models (1 = best)
    """

    candidate: ModelCandidate
    model: SingleTaskGP | None
    cv_metrics: CVMetrics
    log_marginal_likelihood: float
    bic: float
    rank: int = 0


@dataclass
class ModelSelectionResult:
    """Overall result of model selection.

    Attributes:
        best_model: The selected best model
        best_candidate: Configuration of the best model
        all_results: Results for all compared models
        recommendation: Human-readable recommendation
        confidence: Confidence in the selection ("high", "medium", "low")
    """

    best_model: SingleTaskGP
    best_candidate: ModelCandidate
    all_results: list[ModelComparisonResult]
    recommendation: str
    confidence: str


@dataclass
class ModelSelectionConfig:
    """Configuration for model selection.

    Attributes:
        candidates: List of model configurations to compare
        cv_config: Cross-validation configuration
        selection_criterion: Criterion for selection ("cv_rmse", "cv_r2", "lml", "bic")
        keep_all_models: Whether to keep fitted models in memory
        min_improvement: Minimum relative improvement to prefer complex model
    """

    candidates: list[ModelCandidate] = field(default_factory=list)
    cv_config: CVConfig | None = None
    selection_criterion: str = "cv_rmse"
    keep_all_models: bool = False
    min_improvement: float = 0.05  # 5% improvement required for complex model


# Default candidate configurations
DEFAULT_CANDIDATES = [
    ModelCandidate(
        name="Standard GP (Matern 5/2)",
        kernel=KernelType.MATERN_52,
        use_warping=False,
        configuration=ModelConfiguration.STANDARD,
        description="Default choice for most problems. Assumes smooth, stationary functions.",
    ),
    ModelCandidate(
        name="Warped GP (Matern 5/2)",
        kernel=KernelType.MATERN_52,
        use_warping=True,
        configuration=ModelConfiguration.WARPED,
        description="Better for non-stationary functions where behavior varies across the space.",
    ),
    ModelCandidate(
        name="Standard GP (RBF)",
        kernel=KernelType.RBF,
        use_warping=False,
        configuration=ModelConfiguration.STANDARD,
        description="For very smooth, infinitely differentiable functions.",
    ),
    ModelCandidate(
        name="Standard GP (Matern 3/2)",
        kernel=KernelType.MATERN_32,
        use_warping=False,
        configuration=ModelConfiguration.STANDARD,
        description="For rougher functions with less smoothness than Matern 5/2.",
    ),
]


def compare_models(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    config: ModelSelectionConfig | None = None,
) -> ModelSelectionResult:
    """Compare multiple GP model configurations and select the best.

    This function fits multiple GP models with different configurations
    and compares them using cross-validation. The best model is selected
    based on the specified criterion.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1) or (n_samples,)
        bounds: Parameter bounds of shape (2, n_dims)
        config: Model selection configuration

    Returns:
        ModelSelectionResult with the best model and comparison results

    Example:
        >>> result = compare_models(train_x, train_y, bounds)
        >>> best_model = result.best_model
        >>> print(result.recommendation)
    """
    if config is None:
        config = ModelSelectionConfig(candidates=DEFAULT_CANDIDATES)

    if not config.candidates:
        config.candidates = DEFAULT_CANDIDATES

    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    n_samples = train_x.shape[0]

    # Compare each candidate
    results: list[ModelComparisonResult] = []

    for candidate in config.candidates:
        try:
            # Build and fit model
            model = _build_model(train_x, train_y, bounds, candidate)

            # Compute log marginal likelihood
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            model.train()
            output = model(train_x)
            log_mll = mll(output, train_y.squeeze()).item()  # ty: ignore[unresolved-attribute]
            model.eval()

            # Compute BIC
            n_params = _count_model_parameters(model)
            bic = -2 * log_mll + n_params * torch.log(torch.tensor(n_samples)).item()

            # Compute CV metrics
            cv_metrics = compute_loo_cv_optimized(train_x, train_y, bounds, config.cv_config)

            result = ModelComparisonResult(
                candidate=candidate,
                model=model if config.keep_all_models else None,
                cv_metrics=cv_metrics,
                log_marginal_likelihood=log_mll,
                bic=bic,
            )
            results.append(result)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to fit model '{candidate.name}': {e}")
            # Create failed result
            results.append(
                ModelComparisonResult(
                    candidate=candidate,
                    model=None,
                    cv_metrics=CVMetrics(
                        rmse=float("inf"),
                        mae=float("inf"),
                        r_squared=float("-inf"),
                        mean_standardized_error=float("inf"),
                        coverage_95=0.0,
                        per_fold_errors=[],
                        computation_time=0.0,
                        method="failed",
                    ),
                    log_marginal_likelihood=float("-inf"),
                    bic=float("inf"),
                )
            )

    # Rank models based on criterion
    results = _rank_models(results, config.selection_criterion)

    # Select best model
    best_result = results[0]

    # Refit best model if we didn't keep it
    if best_result.model is None:
        best_model = _build_model(train_x, train_y, bounds, best_result.candidate)
    else:
        best_model = best_result.model

    # Generate recommendation
    recommendation, confidence = _generate_recommendation(results)

    return ModelSelectionResult(
        best_model=best_model,
        best_candidate=best_result.candidate,
        all_results=results,
        recommendation=recommendation,
        confidence=confidence,
    )


def _build_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    candidate: ModelCandidate,
) -> SingleTaskGP:
    """Build and fit a GP model for the given candidate configuration.

    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Parameter bounds
        candidate: Model configuration

    Returns:
        Fitted SingleTaskGP
    """
    n_dims = train_x.shape[-1]

    # Build input transform
    if candidate.use_warping:
        input_transform = Warp(
            d=n_dims,
            indices=list(range(n_dims)),
            concentration1_prior=None,
            concentration0_prior=None,
        )
    else:
        input_transform = Normalize(d=n_dims, bounds=bounds)

    # Build kernel
    kernel = _build_kernel(candidate.kernel, n_dims)

    # Create model
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=kernel,
        input_transform=input_transform,
        outcome_transform=Standardize(m=1),
    )

    # Fit model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


def _build_kernel(
    kernel_type: KernelType,
    n_dims: int,
) -> ScaleKernel:
    """Build a kernel for the specified type.

    Args:
        kernel_type: Type of kernel to build
        n_dims: Number of input dimensions

    Returns:
        ScaleKernel wrapping the base kernel
    """
    if kernel_type == KernelType.RBF:
        base_kernel = RBFKernel(ard_num_dims=n_dims)
    elif kernel_type == KernelType.MATERN_32:
        base_kernel = MaternKernel(nu=1.5, ard_num_dims=n_dims)
    elif kernel_type == KernelType.MATERN_12:
        base_kernel = MaternKernel(nu=0.5, ard_num_dims=n_dims)
    else:  # Default: Matern 5/2
        base_kernel = MaternKernel(nu=2.5, ard_num_dims=n_dims)

    return ScaleKernel(base_kernel)


def _count_model_parameters(model: SingleTaskGP) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: GP model

    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_cv_rmse(r: ModelComparisonResult) -> float:
    """Get CV RMSE from result (key function for sorting)."""
    return r.cv_metrics.rmse


def _get_cv_r2(r: ModelComparisonResult) -> float:
    """Get CV R² from result (key function for sorting)."""
    return r.cv_metrics.r_squared


def _get_lml(r: ModelComparisonResult) -> float:
    """Get log marginal likelihood from result (key function for sorting)."""
    return r.log_marginal_likelihood


def _get_bic(r: ModelComparisonResult) -> float:
    """Get BIC from result (key function for sorting)."""
    return r.bic


def _rank_models(
    results: list[ModelComparisonResult],
    criterion: str,
) -> list[ModelComparisonResult]:
    """Rank models based on the selection criterion.

    Args:
        results: List of comparison results
        criterion: Selection criterion

    Returns:
        Results sorted by rank (best first)
    """
    # Default to RMSE for unknown criteria
    key = _get_cv_rmse
    reverse = False

    if criterion == "cv_r2":
        key = _get_cv_r2
        reverse = True  # Higher is better
    elif criterion == "lml":
        key = _get_lml
        reverse = True  # Higher is better
    elif criterion == "bic":
        key = _get_bic
        reverse = False  # Lower is better

    # Sort results
    sorted_results = sorted(results, key=key, reverse=reverse)

    # Assign ranks
    for i, result in enumerate(sorted_results):
        result.rank = i + 1

    return sorted_results


def _generate_recommendation(
    results: list[ModelComparisonResult],
) -> tuple[str, str]:
    """Generate a human-readable recommendation based on results.

    Args:
        results: Ranked comparison results

    Returns:
        Tuple of (recommendation, confidence)
    """
    if not results or results[0].cv_metrics.rmse == float("inf"):
        return "Unable to fit any models successfully.", "low"

    best = results[0]
    second_best = results[1] if len(results) > 1 else None

    parts = [f"Selected '{best.candidate.name}' with CV R² = {best.cv_metrics.r_squared:.3f}."]

    # Confidence assessment
    if best.cv_metrics.r_squared > 0.9:
        confidence = "high"
        parts.append("Model fits the data very well.")
    elif best.cv_metrics.r_squared > 0.7:
        confidence = "medium"
        parts.append("Model fits reasonably well.")
    elif best.cv_metrics.r_squared > 0.5:
        confidence = "medium"
        parts.append("Model fit is moderate; predictions may have high uncertainty.")
    else:
        confidence = "low"
        parts.append("Model fit is poor. Consider collecting more data or checking for outliers.")

    # Compare to second best
    if second_best and second_best.cv_metrics.r_squared > 0:
        improvement = best.cv_metrics.r_squared - second_best.cv_metrics.r_squared
        second_r2 = second_best.cv_metrics.r_squared
        second_name = second_best.candidate.name
        if abs(improvement) < 0.01:
            parts.append(f"'{second_name}' performs similarly (R² = {second_r2:.3f}).")
            confidence = "medium"  # Close models reduce confidence
        elif improvement > 0.1:
            parts.append(f"Substantially better than '{second_name}' (R² = {second_r2:.3f}).")

    # Add description of selected model
    parts.append(best.candidate.description)

    return " ".join(parts), confidence


def select_best_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    include_warping: bool = True,
    include_rbf: bool = True,
) -> tuple[SingleTaskGP, dict[str, Any]]:
    """Simplified interface for model selection.

    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Parameter bounds
        include_warping: Whether to include warped GP in comparison
        include_rbf: Whether to include RBF kernel in comparison

    Returns:
        Tuple of (best_model, selection_info)

    Example:
        >>> model, info = select_best_model(train_x, train_y, bounds)
        >>> print(info["selected_model"])
        >>> print(info["cv_r_squared"])
    """
    candidates = [DEFAULT_CANDIDATES[0]]  # Standard Matern 5/2

    if include_warping:
        candidates.append(DEFAULT_CANDIDATES[1])  # Warped Matern 5/2

    if include_rbf:
        candidates.append(DEFAULT_CANDIDATES[2])  # Standard RBF

    config = ModelSelectionConfig(candidates=candidates)
    result = compare_models(train_x, train_y, bounds, config)

    info = {
        "selected_model": result.best_candidate.name,
        "cv_r_squared": result.best_candidate and result.all_results[0].cv_metrics.r_squared,
        "cv_rmse": result.all_results[0].cv_metrics.rmse,
        "recommendation": result.recommendation,
        "confidence": result.confidence,
        "all_models_compared": [r.candidate.name for r in result.all_results],
        "all_r_squared": {r.candidate.name: r.cv_metrics.r_squared for r in result.all_results},
    }

    return result.best_model, info


def get_model_selection_summary(result: ModelSelectionResult) -> dict[str, Any]:
    """Get a summary of model selection results for diagnostics.

    Args:
        result: Model selection result

    Returns:
        Dictionary suitable for inclusion in diagnostics output
    """
    comparison_table = []
    for r in result.all_results:
        comparison_table.append(
            {
                "model": r.candidate.name,
                "rank": r.rank,
                "cv_rmse": round(r.cv_metrics.rmse, 4)
                if r.cv_metrics.rmse != float("inf")
                else None,
                "cv_r_squared": round(r.cv_metrics.r_squared, 4)
                if r.cv_metrics.r_squared != float("-inf")
                else None,
                "bic": round(r.bic, 2) if r.bic != float("inf") else None,
                "log_mll": round(r.log_marginal_likelihood, 2)
                if r.log_marginal_likelihood != float("-inf")
                else None,
            }
        )

    return {
        "selected_model": result.best_candidate.name,
        "selected_kernel": result.best_candidate.kernel.value,
        "uses_warping": result.best_candidate.use_warping,
        "recommendation": result.recommendation,
        "confidence": result.confidence,
        "comparison_table": comparison_table,
        "n_models_compared": len(result.all_results),
    }
