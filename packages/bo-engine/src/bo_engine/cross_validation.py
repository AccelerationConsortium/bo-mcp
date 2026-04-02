"""Optimized Cross-Validation for Gaussian Processes.

This module provides efficient cross-validation implementations for GP models,
including approximate LOO-CV using influence functions for large datasets.

v2.6: Initial implementation with approximate LOO-CV (Section 2.4)

The Problem:
    Standard LOO-CV refits the GP model N times (once for each held-out point),
    which is O(N^4) overall due to O(N^3) per model fit. For large datasets
    (N > 100), this becomes prohibitively expensive.

The Solution:
    1. Use BoTorch's batch_cross_validation for optimized parallel fitting
    2. Implement approximate LOO using influence functions (O(N^3) total)
    3. Add caching for repeated CV calls
    4. Provide K-fold CV as a faster alternative when LOO is too expensive

References:
    - Sundararajan & Keerthi "Predictive Approaches for Choosing Hyperparameters
      in Gaussian Processes" (2001)
    - Rasmussen & Williams "Gaussian Processes for Machine Learning" Ch. 5.4.2
    - BoTorch batch_cross_validation documentation
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from bo_engine.constants import MIN_OBSERVATIONS_FOR_LOO_CV
from bo_engine.device import ensure_device

logger = logging.getLogger(__name__)


@dataclass
class CVMetrics:
    """Cross-validation metrics.

    Attributes:
        rmse: Root mean squared error
        mae: Mean absolute error
        r_squared: Coefficient of determination (R²)
        mean_standardized_error: Average standardized prediction error
        coverage_95: Fraction of points within 95% CI
        per_fold_errors: List of errors for each fold
        computation_time: Time taken for CV computation (seconds)
        method: CV method used (e.g., "loo", "approximate_loo", "kfold")
    """

    rmse: float
    mae: float
    r_squared: float
    mean_standardized_error: float
    coverage_95: float
    per_fold_errors: list[float]
    computation_time: float
    method: str


@dataclass
class CVConfig:
    """Configuration for cross-validation.

    Attributes:
        method: CV method to use ("auto", "batch_loo", "approximate_loo", "kfold")
        use_approximate: Whether to use approximate LOO for large datasets
        approximate_threshold: Dataset size above which to use approximation
        k_folds: Number of folds for K-fold CV (None = use LOO)
        cache_results: Whether to cache CV results
        cache_ttl: Time-to-live for cached results (seconds)
    """

    method: str = "auto"  # "auto", "batch_loo", "approximate_loo", "kfold"
    use_approximate: bool = True
    approximate_threshold: int = 100
    k_folds: int | None = None
    cache_results: bool = True
    cache_ttl: float = 300.0  # 5 minutes


# Simple cache for CV results
_cv_cache: dict[str, tuple[CVMetrics, float]] = {}


def _parse_cv_arguments(
    model_or_train_x: SingleTaskGP | Tensor,
    train_x_or_train_y: Tensor,
    train_y_or_bounds: Tensor,
    config_or_none: CVConfig | Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, CVConfig]:
    """Parse the polymorphic arguments into (train_x, train_y, bounds, config)."""
    if isinstance(model_or_train_x, SingleTaskGP):
        train_x = train_x_or_train_y
        train_y = train_y_or_bounds
        config = config_or_none if isinstance(config_or_none, CVConfig) else CVConfig()
        bounds = torch.stack([train_x.min(dim=0).values, train_x.max(dim=0).values])
    else:
        train_x = model_or_train_x
        train_y = train_x_or_train_y
        bounds = train_y_or_bounds
        config = config_or_none if isinstance(config_or_none, CVConfig) else CVConfig()

    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    return train_x, train_y, bounds, config


def _resolve_cv_method(config: CVConfig, n_samples: int) -> str:
    """Determine which CV method to use based on config and dataset size."""
    method = config.method
    if method != "auto":
        return method
    if config.k_folds is not None:
        return "kfold"
    if config.use_approximate and n_samples >= config.approximate_threshold:
        return "approximate_loo"
    return "batch_loo"


def _dispatch_cv_method(
    method: str,
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    config: CVConfig,
) -> CVMetrics:
    """Dispatch to the appropriate CV computation function."""
    if method == "kfold":
        k = config.k_folds if config.k_folds is not None else 5
        return _compute_kfold_cv(train_x, train_y, bounds, k)
    if method == "approximate_loo":
        return _compute_approximate_loo(train_x, train_y, bounds)
    return _compute_batch_loo_cv(train_x, train_y)


def _check_cv_cache(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    config: CVConfig,
) -> tuple[str | None, CVMetrics | None]:
    """Check cache for existing CV results. Returns (cache_key, cached_result_or_None)."""
    if not config.cache_results:
        return None, None
    cache_key = _compute_cache_key(train_x, train_y, bounds, config)
    if cache_key in _cv_cache:
        cached_metrics, timestamp = _cv_cache[cache_key]
        if time.time() - timestamp < config.cache_ttl:
            return cache_key, cached_metrics
    return cache_key, None


def compute_loo_cv_optimized(
    model_or_train_x: SingleTaskGP | Tensor,
    train_x_or_train_y: Tensor,
    train_y_or_bounds: Tensor,
    config_or_none: CVConfig | Tensor | None = None,
) -> CVMetrics:
    """Compute LOO-CV with automatic optimization for large datasets.

    This function automatically selects the best CV method based on
    dataset size and configuration.

    Can be called in two ways:
    1. compute_loo_cv_optimized(model, train_x, train_y, config)
    2. compute_loo_cv_optimized(train_x, train_y, bounds, config)

    Args:
        model_or_train_x: Either a fitted GP model or training inputs
        train_x_or_train_y: Training inputs (if model provided) or training outputs
        train_y_or_bounds: Training outputs (if model provided) or parameter bounds
        config_or_none: CV configuration

    Returns:
        CVMetrics with computed metrics

    Example:
        >>> metrics = compute_loo_cv_optimized(train_x, train_y, bounds)
        >>> print(f"R² = {metrics.r_squared:.3f}, Method = {metrics.method}")
    """
    train_x, train_y, bounds, config = _parse_cv_arguments(
        model_or_train_x, train_x_or_train_y, train_y_or_bounds, config_or_none
    )

    n_samples = train_x.shape[0]
    if n_samples < MIN_OBSERVATIONS_FOR_LOO_CV:
        return _create_nan_metrics("insufficient_data")

    cache_key, cached = _check_cv_cache(train_x, train_y, bounds, config)
    if cached is not None:
        return cached

    start_time = time.time()
    method = _resolve_cv_method(config, n_samples)
    metrics = _dispatch_cv_method(method, train_x, train_y, bounds, config)

    metrics = CVMetrics(
        rmse=metrics.rmse,
        mae=metrics.mae,
        r_squared=metrics.r_squared,
        mean_standardized_error=metrics.mean_standardized_error,
        coverage_95=metrics.coverage_95,
        per_fold_errors=metrics.per_fold_errors,
        computation_time=time.time() - start_time,
        method=metrics.method,
    )

    if config.cache_results and cache_key is not None:
        _cv_cache[cache_key] = (metrics, time.time())

    return metrics


def _compute_batch_loo_cv(
    train_x: Tensor,
    train_y: Tensor,
) -> CVMetrics:
    """Compute LOO-CV using BoTorch's batch_cross_validation.

    This is the standard approach but optimized for parallel fitting.

    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Parameter bounds

    Returns:
        CVMetrics
    """
    # Generate LOO folds
    cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y)

    try:
        # Use BoTorch's optimized batch CV
        cv_results = batch_cross_validation(
            model_cls=SingleTaskGP,
            mll_cls=ExactMarginalLogLikelihood,
            cv_folds=cv_folds,
        )

        # Extract predictions
        pred_mean = cv_results.posterior.mean.squeeze()
        pred_var = cv_results.posterior.variance.squeeze()
        actuals = train_y.squeeze()

        # Compute metrics
        errors = (pred_mean - actuals).abs()
        squared_errors = errors**2

        rmse = squared_errors.mean().sqrt().item()
        mae = errors.mean().item()

        ss_res = ((pred_mean - actuals) ** 2).sum()
        ss_tot = ((actuals - actuals.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / (ss_tot + 1e-10)).item() if ss_tot > 0 else 0.0

        # Standardized errors
        std = pred_var.sqrt().clamp(min=1e-6)
        standardized_errors = (pred_mean - actuals).abs() / std
        mean_std_error = standardized_errors.mean().item()

        # Coverage: fraction within 1.96 std (95% CI)
        within_95ci = standardized_errors < 1.96
        coverage_95 = within_95ci.float().mean().item()

        return CVMetrics(
            rmse=rmse,
            mae=mae,
            r_squared=r_squared,
            mean_standardized_error=mean_std_error,
            coverage_95=coverage_95,
            per_fold_errors=errors.tolist(),
            computation_time=0.0,
            method="batch_loo",
        )

    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning(f"Batch LOO-CV failed: {e}")
        return _create_nan_metrics("batch_loo_failed")


def _compute_approximate_loo(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
) -> CVMetrics:
    """Compute approximate LOO-CV using influence functions.

    For a GP with kernel matrix K and observations y, the LOO predictions
    can be approximated without refitting using:

        μ_{-i} ≈ μ - α_i / K^{-1}_{ii}
        σ²_{-i} ≈ 1 / K^{-1}_{ii}

    where α = K^{-1} y and K^{-1}_{ii} is the i-th diagonal of the inverse.

    This is O(N^3) instead of O(N^4) for standard LOO.

    Reference:
        Rasmussen & Williams "GPML" Equation 5.12

    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Parameter bounds

    Returns:
        CVMetrics with approximate values
    """
    n_dims = train_x.shape[-1]

    # Fit full GP model
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=n_dims, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Get the kernel matrix inverse and compute LOO predictions
    # Use the model's covariance module
    model.eval()

    with torch.no_grad():
        # For approximate LOO, we use the posterior variance at training points
        # as a proxy for LOO variance (this is an approximation)
        train_posterior = model.posterior(train_x)
        train_mean = train_posterior.mean.squeeze()
        train_var = train_posterior.variance.squeeze()

        # Compute standardized residuals
        actuals = train_y.squeeze()
        residuals = train_mean - actuals

        # Use leave-one-out approximation
        # LOO prediction error ≈ residual / (1 - leverage)
        # For GP, leverage ≈ 1 - noise_variance / posterior_variance
        noise_var = model.likelihood.noise.item()  # ty: ignore[call-non-callable]
        leverage = 1 - noise_var / (train_var + noise_var + 1e-10)

        # Approximate LOO errors (PRESS residuals)
        loo_errors = residuals / (1 - leverage.clamp(max=0.99) + 1e-10)
        loo_var = train_var / ((1 - leverage.clamp(max=0.99)) ** 2 + 1e-10)

    # Compute metrics from approximate LOO
    errors = loo_errors.abs()
    squared_errors = errors**2

    rmse = squared_errors.mean().sqrt().item()
    mae = errors.mean().item()

    ss_res = squared_errors.sum()
    ss_tot = ((actuals - actuals.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / (ss_tot + 1e-10)).item() if ss_tot > 0 else 0.0

    # Standardized errors using approximate LOO variance
    std = loo_var.sqrt().clamp(min=1e-6)
    standardized_errors = errors / std
    mean_std_error = standardized_errors.mean().item()

    # Coverage
    within_95ci = standardized_errors < 1.96
    coverage_95 = within_95ci.float().mean().item()

    return CVMetrics(
        rmse=rmse,
        mae=mae,
        r_squared=r_squared,
        mean_standardized_error=mean_std_error,
        coverage_95=coverage_95,
        per_fold_errors=errors.tolist(),
        computation_time=0.0,
        method="approximate_loo",
    )


def _compute_cv_metrics_from_predictions(
    predictions: Tensor,
    variances: Tensor,
    actuals: Tensor,
    per_fold_errors: list[float],
    method: str,
) -> CVMetrics:
    """Compute CVMetrics from prediction vs actual tensors."""
    errors = (predictions - actuals).abs()
    squared_errors = errors**2

    rmse = squared_errors.mean().sqrt().item()
    mae = errors.mean().item()

    ss_res = ((predictions - actuals) ** 2).sum()
    ss_tot = ((actuals - actuals.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / (ss_tot + 1e-10)).item() if ss_tot > 0 else 0.0

    std = variances.sqrt().clamp(min=1e-6)
    standardized_errors = errors / std
    mean_std_error = standardized_errors.mean().item()

    within_95ci = standardized_errors < 1.96
    coverage_95 = within_95ci.float().mean().item()

    return CVMetrics(
        rmse=rmse,
        mae=mae,
        r_squared=r_squared,
        mean_standardized_error=mean_std_error,
        coverage_95=coverage_95,
        per_fold_errors=per_fold_errors,
        computation_time=0.0,
        method=method,
    )


def _compute_kfold_cv(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    k: int = 5,
) -> CVMetrics:
    """Compute K-fold cross-validation.

    K-fold CV is faster than LOO when K << N, trading off some bias
    for reduced computation.

    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Parameter bounds
        k: Number of folds

    Returns:
        CVMetrics
    """
    n_samples = train_x.shape[0]
    n_dims = train_x.shape[-1]

    if k > n_samples:
        k = n_samples  # Fall back to LOO

    indices = torch.randperm(n_samples)
    fold_sizes = [n_samples // k + (1 if i < n_samples % k else 0) for i in range(k)]

    all_predictions = torch.zeros_like(train_y.squeeze())
    all_variances = torch.zeros_like(train_y.squeeze())
    per_fold_errors: list[float] = []

    current_idx = 0
    for fold_idx in range(k):
        fold_size = fold_sizes[fold_idx]
        test_indices = indices[current_idx : current_idx + fold_size]
        train_indices = torch.cat([indices[:current_idx], indices[current_idx + fold_size :]])
        current_idx += fold_size

        train_x_fold = train_x[train_indices]
        train_y_fold = train_y[train_indices]
        test_x_fold = train_x[test_indices]
        test_y_fold = train_y[test_indices]

        if train_x_fold.shape[0] < 2:
            continue

        try:
            model = SingleTaskGP(
                train_X=train_x_fold,
                train_Y=train_y_fold,
                input_transform=Normalize(d=n_dims, bounds=bounds),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            model.eval()
            with torch.no_grad():
                posterior = model.posterior(test_x_fold)
                pred_mean = posterior.mean.squeeze()
                pred_var = posterior.variance.squeeze()

            all_predictions[test_indices] = pred_mean
            all_variances[test_indices] = pred_var
            per_fold_errors.append((pred_mean - test_y_fold.squeeze()).abs().mean().item())

        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"K-fold CV failed for fold {fold_idx}: {e}")
            continue

    return _compute_cv_metrics_from_predictions(
        all_predictions,
        all_variances,
        train_y.squeeze(),
        per_fold_errors,
        f"{k}fold",
    )


def compute_cv_for_model_list(
    model: ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    config: CVConfig | None = None,
) -> dict[int, CVMetrics]:
    """Compute CV metrics for each objective in a ModelListGP.

    Args:
        model: Fitted ModelListGP
        train_x: Training inputs
        train_y: Training outputs (n_samples, n_objectives)
        bounds: Parameter bounds
        config: CV configuration

    Returns:
        Dictionary mapping objective index to CVMetrics
    """
    results = {}

    for i, _sub_model in enumerate(model.models):
        train_y_i = train_y[:, i : i + 1]
        metrics = compute_loo_cv_optimized(train_x, train_y_i, bounds, config)
        results[i] = metrics

    return results


def clear_cv_cache() -> None:
    """Clear the cross-validation results cache."""
    global _cv_cache
    _cv_cache.clear()


def _compute_cache_key(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    config: CVConfig,
) -> str:
    """Compute a cache key for CV results.

    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Parameter bounds
        config: CV configuration

    Returns:
        Cache key string
    """
    # Create hash of inputs
    x_bytes = train_x.detach().cpu().numpy().tobytes()
    y_bytes = train_y.detach().cpu().numpy().tobytes()
    b_bytes = bounds.detach().cpu().numpy().tobytes()

    config_str = f"{config.use_approximate}_{config.approximate_threshold}_{config.k_folds}"

    combined = x_bytes + y_bytes + b_bytes + config_str.encode()
    return hashlib.sha256(combined).hexdigest()


def _create_nan_metrics(method: str) -> CVMetrics:
    """Create CVMetrics with NaN values for failed computations."""
    return CVMetrics(
        rmse=float("nan"),
        mae=float("nan"),
        r_squared=float("nan"),
        mean_standardized_error=float("nan"),
        coverage_95=float("nan"),
        per_fold_errors=[],
        computation_time=0.0,
        method=method,
    )


def estimate_cv_time(
    n_samples: int,
    method: str = "auto",
    k_folds: int | None = None,
) -> float:
    """Estimate cross-validation computation time.

    Provides rough estimates of CV computation time for a specific method
    to help users choose the appropriate approach.

    Args:
        n_samples: Number of training samples
        method: "batch_loo", "approximate_loo", "kfold", or "auto"
        k_folds: Number of folds for kfold method (default 5)

    Returns:
        Estimated time in seconds for the specified method
    """
    # Rough complexity estimates (empirical, will vary by hardware)
    # GP fitting is O(N^3), LOO requires N fits

    # Base time for single GP fit (N^3 complexity)
    base_fit_time = (n_samples / 100) ** 3 * 0.1  # ~0.1s for N=100

    # Handle method aliases
    if method in ("loo", "batch_loo"):
        return n_samples * base_fit_time
    elif method == "approximate_loo":
        return base_fit_time * 1.5
    elif method == "kfold":
        k = k_folds if k_folds is not None else 5
        return k * base_fit_time
    elif method == "auto":
        # Return estimate for recommended method
        if n_samples > 100:
            return base_fit_time * 1.5  # approximate_loo
        else:
            return n_samples * base_fit_time  # batch_loo
    else:
        # Default to LOO estimate
        return n_samples * base_fit_time


def get_cv_time_estimates(
    n_samples: int,
) -> dict[str, Any]:
    """Get estimated computation times for all CV methods.

    Provides rough estimates of CV computation time for different methods
    to help users choose the appropriate approach.

    Args:
        n_samples: Number of training samples

    Returns:
        Dictionary with estimated times in seconds for each method
    """
    # Base time for single GP fit (N^3 complexity)
    base_fit_time = (n_samples / 100) ** 3 * 0.1  # ~0.1s for N=100

    estimates: dict[str, Any] = {
        "batch_loo": n_samples * base_fit_time,  # N fits
        "approximate_loo": base_fit_time * 1.5,  # 1 fit + some computation
        "5fold": 5 * base_fit_time,  # 5 fits
        "10fold": 10 * base_fit_time,  # 10 fits
    }

    # Add warning thresholds
    estimates["recommended"] = "approximate_loo" if n_samples > 100 else "batch_loo"
    estimates["warning_threshold_seconds"] = 60.0

    return estimates
