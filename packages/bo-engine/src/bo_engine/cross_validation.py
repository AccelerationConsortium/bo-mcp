"""Optimized Cross-Validation for Gaussian Processes.

This module provides efficient cross-validation implementations for GP models,
including the exact leave-one-out posterior downdate for large datasets.

The Problem:
    Standard LOO-CV refits the GP model N times (once for each held-out point),
    which is O(N^4) overall due to O(N^3) per model fit. For large datasets
    (N > 100), this becomes prohibitively expensive.

The Solution:
    1. Use BoTorch's batch_cross_validation for optimized parallel fitting
    2. For large N, fit once and compute the exact LOO predictive moments from
       the inverse train covariance (O(N^3) total); the only approximation is
       that the hyperparameters are not re-estimated per fold
    3. Add caching for repeated CV calls
    4. Provide K-fold CV as a faster alternative when LOO is too expensive

All predictive moments are for *observations* (latent f plus observation
noise), so standardized errors and coverage refer to held-out measurements.

References:
    - Sundararajan & Keerthi "Predictive Approaches for Choosing Hyperparameters
      in Gaussian Processes" (2001)
    - Rasmussen & Williams "Gaussian Processes for Machine Learning" §5.4.2,
      Eqs. 5.10-5.12 (exact LOO posterior via the inverse covariance diagonal)
    - BoTorch batch_cross_validation documentation
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from bo_engine.constants import (
    CI_95_Z_SCORE,
    MIN_OBSERVATIONS_FOR_LOO_CV,
    NUMERICAL_EPSILON,
    SAFE_DIVISION_EPSILON,
)
from bo_engine.device import ensure_device

logger = logging.getLogger(__name__)

# Builds *and fits* a single-output GP on the given (train_x, train_y); any
# further context (bounds, kernel choice, warping) is baked into the closure.
ModelFactory = Callable[[Tensor, Tensor], SingleTaskGP]


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
) -> tuple[SingleTaskGP | None, Tensor, Tensor, Tensor, CVConfig]:
    """Parse the polymorphic arguments into (model, train_x, train_y, bounds, config).

    ``model`` is None for the tensor-form call; for the model-form call it is
    the fitted model the caller asked to validate.
    """
    if isinstance(model_or_train_x, SingleTaskGP):
        model = model_or_train_x
        train_x = train_x_or_train_y
        train_y = train_y_or_bounds
        config = config_or_none if isinstance(config_or_none, CVConfig) else CVConfig()
        bounds = torch.stack([train_x.min(dim=0).values, train_x.max(dim=0).values])
    else:
        model = None
        train_x = model_or_train_x
        train_y = train_x_or_train_y
        bounds = train_y_or_bounds
        config = config_or_none if isinstance(config_or_none, CVConfig) else CVConfig()

    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    return model, train_x, train_y, bounds, config


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
    model_factory: ModelFactory | None = None,
) -> CVMetrics:
    """Dispatch to the appropriate CV computation function."""
    if method == "kfold":
        k = config.k_folds if config.k_folds is not None else 5
        return _compute_kfold_cv(train_x, train_y, bounds, k, model_factory)
    if method == "approximate_loo":
        return _compute_approximate_loo(train_x, train_y, bounds, model_factory)
    return _compute_batch_loo_cv(train_x, train_y, model_factory=model_factory)


def _check_cv_cache(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    config: CVConfig,
    model_factory_key: str | None = None,
) -> tuple[str | None, CVMetrics | None]:
    """Check cache for existing CV results. Returns (cache_key, cached_result_or_None)."""
    if not config.cache_results:
        return None, None
    cache_key = _compute_cache_key(train_x, train_y, bounds, config, model_factory_key)
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
    *,
    model_factory: ModelFactory | None = None,
    model_factory_key: str | None = None,
) -> CVMetrics:
    """Compute LOO-CV with automatic optimization for large datasets.

    This function automatically selects the best CV method based on
    dataset size and configuration.

    Can be called in two ways:

    1. ``compute_loo_cv_optimized(model, train_x, train_y, config)`` —
       validates the **passed fitted model** via the exact LOO downdate at
       its fitted hyperparameters (reported as method ``"model_loo"``).
       The tensors must be the model's own training data, and only the
       ``"auto"``/``"approximate_loo"`` methods apply: refit-based methods
       (``"batch_loo"``, ``"kfold"``, including a K-fold request made via
       ``CVConfig(k_folds=...)``) cannot rebuild an arbitrary fitted model
       and raise ``ValueError`` unless a ``model_factory`` is given.
       Results are never cached (the cache key cannot capture model
       identity).
    2. ``compute_loo_cv_optimized(train_x, train_y, bounds, config)`` —
       builds the models itself, via ``model_factory`` or the default
       construction.

    Args:
        model_or_train_x: Either a fitted GP model or training inputs
        train_x_or_train_y: Training inputs (if model provided) or training outputs
        train_y_or_bounds: Training outputs (if model provided) or parameter bounds
        config_or_none: CV configuration
        model_factory: Optional callable that builds and fits the GP evaluated
            by CV. When omitted, a default SingleTaskGP (Normalize +
            Standardize, Matern 5/2) is used. This is how model-selection
            candidates are scored on their own fits rather than the default's.
            Takes precedence over a passed fitted model.
        model_factory_key: Stable identifier for ``model_factory``, included
            in the cache key so different candidates never share cached
            metrics. When a factory is given without a key, caching is
            bypassed entirely for the call.

    Returns:
        CVMetrics with computed metrics

    Raises:
        ValueError: For the model-form call when a refit-based CV method is
            requested without a ``model_factory``, or when the passed
            tensors are not the model's training data.

    Example:
        >>> metrics = compute_loo_cv_optimized(train_x, train_y, bounds)
        >>> print(f"R² = {metrics.r_squared:.3f}, Method = {metrics.method}")
    """
    provided_model, train_x, train_y, bounds, config = _parse_cv_arguments(
        model_or_train_x, train_x_or_train_y, train_y_or_bounds, config_or_none
    )

    n_samples = train_x.shape[0]
    if n_samples < MIN_OBSERVATIONS_FOR_LOO_CV:
        return _create_nan_metrics("insufficient_data")

    use_provided_model = provided_model is not None and model_factory is None

    # An anonymous factory cannot be keyed, and the cache key cannot capture
    # a fitted model's identity, so neither may share cached results.
    cache_usable = not use_provided_model and (
        model_factory is None or model_factory_key is not None
    )
    cache_key: str | None = None
    if cache_usable:
        cache_key, cached = _check_cv_cache(train_x, train_y, bounds, config, model_factory_key)
        if cached is not None:
            return cached

    start_time = time.time()
    if provided_model is not None and model_factory is None:
        metrics = _compute_provided_model_loo(provided_model, train_x, train_y, config)
    else:
        method = _resolve_cv_method(config, n_samples)
        metrics = _dispatch_cv_method(method, train_x, train_y, bounds, config, model_factory)

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


def _compute_provided_model_loo(
    model: SingleTaskGP,
    train_x: Tensor,
    train_y: Tensor,
    config: CVConfig,
) -> CVMetrics:
    """Validate a caller-provided fitted model via the exact LOO downdate.

    A fitted model carries no recipe for rebuilding itself on fold subsets,
    so refit-based methods cannot honor "validate THIS model" semantics —
    silently scoring a freshly built default model instead would misreport
    a custom (warped/RBF/non-default) model's quality. The downdate at the
    model's own fitted hyperparameters is the one honest option.
    """
    # Mirror _resolve_cv_method's precedence: an explicit method wins,
    # otherwise a k_folds setting is this API's K-fold request — it must be
    # rejected here, not silently downgraded to the downdate.
    requested_method = config.method
    if requested_method == "auto" and config.k_folds is not None:
        requested_method = "kfold"

    if requested_method not in ("auto", "approximate_loo"):
        message = (
            f"CV method '{requested_method}' refits models per fold and cannot rebuild "
            "an arbitrary fitted model. Pass model_factory=... or use the tensor "
            "form compute_loo_cv_optimized(train_x, train_y, bounds, config)."
        )
        raise ValueError(message)

    if not matches_model_training_data(model, train_x, train_y):
        message = (
            "The passed tensors are not the fitted model's training data, so its "
            "LOO downdate would describe a different dataset. Pass the model's own "
            "training data, or use the tensor form to cross-validate fresh fits."
        )
        raise ValueError(message)

    return _loo_metrics_from_fitted_model(model, train_y, "model_loo")


def _make_default_model_factory(bounds: Tensor) -> ModelFactory:
    """Return the default CV model factory: Normalize + Standardize SingleTaskGP."""

    def build_and_fit(train_x: Tensor, train_y: Tensor) -> SingleTaskGP:
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    return build_and_fit


def _compute_batch_loo_cv(
    train_x: Tensor,
    train_y: Tensor,
    model_factory: ModelFactory | None = None,
) -> CVMetrics:
    """Compute LOO-CV using BoTorch's batch_cross_validation.

    This is the standard approach but optimized for parallel fitting. When a
    ``model_factory`` is given, the batched path cannot apply it (BoTorch
    builds the fold models itself), so each fold is refit sequentially with
    the factory instead.

    Predictive moments include observation noise: the held-out target is a
    noisy measurement, not the latent function value.

    Args:
        train_x: Training inputs
        train_y: Training outputs
        model_factory: Optional custom model builder, applied per fold (any
            context such as bounds arrives via the factory's closure)

    Returns:
        CVMetrics
    """
    if model_factory is not None:
        return _compute_refit_loo_cv(train_x, train_y, model_factory)

    # Generate LOO folds
    cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y)

    try:
        # Use BoTorch's optimized batch CV
        cv_results = batch_cross_validation(
            model_cls=SingleTaskGP,
            mll_cls=ExactMarginalLogLikelihood,
            cv_folds=cv_folds,
            observation_noise=True,
        )

        # Extract predictions
        pred_mean = cv_results.posterior.mean.squeeze()
        pred_var = cv_results.posterior.variance.squeeze()
        actuals = train_y.squeeze()

        errors = (pred_mean - actuals).abs()

        return _compute_cv_metrics_from_predictions(
            pred_mean,
            pred_var,
            actuals,
            errors.tolist(),
            "batch_loo",
        )

    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("Batch LOO-CV failed: %s", e)
        return _create_nan_metrics("batch_loo_failed")


def _compute_refit_loo_cv(
    train_x: Tensor,
    train_y: Tensor,
    model_factory: ModelFactory,
) -> CVMetrics:
    """Exact LOO-CV with a per-fold refit through ``model_factory``.

    O(n) model fits — used when CV must score a custom model configuration
    that BoTorch's batched CV cannot construct.
    """
    n_samples = train_x.shape[0]
    pred_means = torch.zeros_like(train_y.squeeze(-1))
    pred_vars = torch.ones_like(train_y.squeeze(-1))
    fitted = torch.zeros(n_samples, dtype=torch.bool, device=train_x.device)

    for i in range(n_samples):
        mask = torch.ones(n_samples, dtype=torch.bool, device=train_x.device)
        mask[i] = False
        try:
            model = model_factory(train_x[mask], train_y[mask])
            model.eval()
            with torch.no_grad():
                posterior = model.posterior(train_x[i : i + 1], observation_noise=True)
                pred_means[i] = posterior.mean.reshape(-1)[0]
                pred_vars[i] = posterior.variance.reshape(-1)[0]
            fitted[i] = True
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Refit LOO fold %d failed: %s: %s", i, type(e).__name__, e)
            continue

    if int(fitted.sum().item()) < 2:
        return _create_nan_metrics("loo_refit_failed")

    actuals = train_y.squeeze(-1)[fitted]
    errors = (pred_means[fitted] - actuals).abs()

    return _compute_cv_metrics_from_predictions(
        pred_means[fitted],
        pred_vars[fitted],
        actuals,
        errors.tolist(),
        "batch_loo",
    )


def compute_exact_loo_moments(model: SingleTaskGP) -> tuple[Tensor, Tensor]:
    """Exact LOO predictive moments of a fitted GP at fixed hyperparameters.

    Implements Rasmussen & Williams "GPML" Eqs. 5.10-5.12: with the noisy
    train covariance ``K = K_f + σ_n²·I`` and targets ``t`` (both in the
    model's transformed target space),

        μ_{-i} = t_i - [K⁻¹(t - m)]_i / [K⁻¹]_{ii}
        σ²_{-i} = 1 / [K⁻¹]_{ii}

    where ``m`` is the prior mean. Because ``K`` includes the noise term,
    ``σ²_{-i}`` is the predictive variance of the held-out *observation*.
    The downdate is exact for fixed hyperparameters; only their
    re-estimation per fold is skipped relative to a full refit-LOO.

    Args:
        model: Fitted single-output, non-batched GP. Moments are returned in
            the model's internal target space (i.e. after any outcome
            transform applied at construction); untransform via
            ``model.outcome_transform.untransform`` for original units.

    Returns:
        Tuple ``(loo_mean, loo_var)`` of shape ``(n,)`` tensors.

    Raises:
        ValueError: If the model is batched or multi-output.
        RuntimeError: If the train covariance is not positive definite.
    """
    targets = model.train_targets
    if targets.dim() != 1:
        message = (
            "Exact LOO downdate requires a single-output, non-batched GP; "
            f"got train_targets of shape {tuple(targets.shape)}."
        )
        raise ValueError(message)

    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            # In train mode the forward pass applies the input transform and
            # returns the prior, exactly as during marginal-likelihood fitting.
            prior = model(*model.train_inputs)
            if prior.batch_shape:
                message = (
                    "Exact LOO downdate requires a non-batched GP; got a prior "
                    f"with batch shape {tuple(prior.batch_shape)} (e.g. a fully "
                    "Bayesian model)."
                )
                raise ValueError(message)
            # A Gaussian(-family) likelihood marginalizes the prior into a
            # MultivariateNormal with the noise added to the covariance.
            mvn = model.likelihood(prior)
            covar = mvn.covariance_matrix  # ty: ignore[unresolved-attribute]
            chol = torch.linalg.cholesky(covar)
            covar_inv = torch.cholesky_inverse(chol)
            covar_inv_diag = covar_inv.diagonal()
            alpha = covar_inv @ (targets - mvn.mean)
            loo_mean = targets - alpha / covar_inv_diag
            loo_var = 1.0 / covar_inv_diag
    finally:
        if not was_training:
            model.eval()

    return loo_mean, loo_var


def _compute_approximate_loo(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    model_factory: ModelFactory | None = None,
) -> CVMetrics:
    """Compute LOO-CV from a single fit via the exact posterior downdate.

    Fits one GP on the full data and evaluates the exact LOO predictive
    moments (GPML Eqs. 5.10-5.12) in the model's standardized target space,
    then untransforms them to original units through the fitted outcome
    transform. This is O(N^3) instead of O(N^4) for refit-LOO; the only
    approximation is that hyperparameters are not re-estimated per fold.

    Reference:
        Rasmussen & Williams "GPML" §5.4.2, Eqs. 5.10-5.12

    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Parameter bounds
        model_factory: Optional custom model builder for the full-data fit

    Returns:
        CVMetrics in original target units
    """
    factory = model_factory if model_factory is not None else _make_default_model_factory(bounds)
    model = factory(train_x, train_y)

    return _loo_metrics_from_fitted_model(model, train_y, "approximate_loo")


def _loo_metrics_from_fitted_model(
    model: SingleTaskGP,
    train_y: Tensor,
    method: str,
) -> CVMetrics:
    """CV metrics from a fitted model's exact LOO downdate, in original units.

    The downdate moments live in the model's transformed target space; they
    are untransformed through the fitted outcome transform before being
    scored against ``train_y``.
    """
    loo_mean, loo_var = compute_exact_loo_moments(model)

    outcome_transform = getattr(model, "outcome_transform", None)
    if outcome_transform is not None:
        loo_mean, loo_var = outcome_transform.untransform(
            loo_mean.unsqueeze(-1), loo_var.unsqueeze(-1)
        )
        loo_mean = loo_mean.squeeze(-1)
        loo_var = loo_var.squeeze(-1)

    actuals = train_y.squeeze(-1)
    errors = (loo_mean - actuals).abs()

    return _compute_cv_metrics_from_predictions(
        loo_mean,
        loo_var,
        actuals,
        errors.tolist(),
        method,
    )


def matches_model_training_data(model: SingleTaskGP, train_x: Tensor, train_y: Tensor) -> bool:
    """Whether the passed data is, by value, the model's own training data.

    The LOO downdate produces results for the model's stored training set,
    so it may only stand in for the caller's data when that data *is* the
    stored set — same points, same order. Inputs are compared in raw
    (untransformed) space; targets are compared by untransforming the
    model's stored (transformed) targets back to the caller's units. Any
    failure to compare (shape mismatch, batched targets, a transform that
    cannot untransform) counts as a mismatch.
    """
    if train_y.dim() > 1:
        train_y = train_y.squeeze(-1)

    inputs = model.train_inputs[0]
    targets = model.train_targets
    if targets.dim() != 1 or inputs.shape != train_x.shape or targets.shape != train_y.shape:
        return False

    if not torch.allclose(inputs, train_x.to(inputs), rtol=1e-6, atol=1e-8):
        return False

    outcome_transform = getattr(model, "outcome_transform", None)
    if outcome_transform is None:
        raw_targets = targets
    else:
        try:
            # NotImplementedError (e.g. Log transforms) is a RuntimeError subclass.
            raw_targets, _ = outcome_transform.untransform(targets.unsqueeze(-1))
        except (RuntimeError, ValueError):
            return False
        raw_targets = raw_targets.squeeze(-1)

    return torch.allclose(raw_targets, train_y.to(raw_targets), rtol=1e-6, atol=1e-8)


@dataclass(frozen=True)
class CVScoreFields:
    """Scalar cross-validation scores shared by every CV/LOO surface.

    Attributes mirror the regression-quality block that was historically
    copy-pasted across the CV and diagnostics LOO modules; this is now the
    single computation site so the metric definitions cannot drift.
    """

    rmse: float
    mae: float
    r_squared: float
    mean_standardized_error: float
    coverage_95: float


def compute_cv_score_fields(
    predictions: Tensor,
    variances: Tensor,
    actuals: Tensor,
) -> CVScoreFields:
    """Single source of truth for the RMSE/MAE/R²/std-error/coverage block.

    Both the CV surface (``cross_validation``) and the diagnostics LOO
    surface (``diagnostics_loo``) route through this helper so the metric
    definitions — and the named constants they depend on
    (:data:`CI_95_Z_SCORE`, :data:`SAFE_DIVISION_EPSILON`,
    :data:`NUMERICAL_EPSILON`) — stay identical.

    Predictive ``variances`` include observation noise, so the standardized
    errors and 95% coverage describe held-out *measurements*. The variance is
    floored at :data:`SAFE_DIVISION_EPSILON` before the square root to keep the
    standardized-error division well-defined, and R² falls back to ``0.0`` when
    the total sum of squares is below :data:`NUMERICAL_EPSILON` (degenerate,
    near-constant targets).

    Args:
        predictions: Predicted held-out means, shape ``(n,)``.
        variances: Predictive (observation) variances, shape ``(n,)``.
        actuals: Observed held-out targets, shape ``(n,)``.

    Returns:
        CVScoreFields with the scalar metrics.
    """
    errors = (predictions - actuals).abs()

    rmse = (errors**2).mean().sqrt().item()
    mae = errors.mean().item()

    ss_res = ((predictions - actuals) ** 2).sum()
    ss_tot = ((actuals - actuals.mean()) ** 2).sum()
    r_squared = (
        1 - (ss_res / (ss_tot + NUMERICAL_EPSILON)).item() if ss_tot > NUMERICAL_EPSILON else 0.0
    )

    std = variances.sqrt().clamp(min=SAFE_DIVISION_EPSILON)
    standardized_errors = errors / std
    mean_std_error = standardized_errors.mean().item()

    coverage_95 = (standardized_errors < CI_95_Z_SCORE).float().mean().item()

    return CVScoreFields(
        rmse=rmse,
        mae=mae,
        r_squared=r_squared,
        mean_standardized_error=mean_std_error,
        coverage_95=coverage_95,
    )


def _compute_cv_metrics_from_predictions(
    predictions: Tensor,
    variances: Tensor,
    actuals: Tensor,
    per_fold_errors: list[float],
    method: str,
) -> CVMetrics:
    """Compute CVMetrics from prediction vs actual tensors."""
    scores = compute_cv_score_fields(predictions, variances, actuals)

    return CVMetrics(
        rmse=scores.rmse,
        mae=scores.mae,
        r_squared=scores.r_squared,
        mean_standardized_error=scores.mean_standardized_error,
        coverage_95=scores.coverage_95,
        per_fold_errors=per_fold_errors,
        computation_time=0.0,
        method=method,
    )


def _compute_kfold_cv(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    k: int = 5,
    model_factory: ModelFactory | None = None,
) -> CVMetrics:
    """Compute K-fold cross-validation.

    K-fold CV is faster than LOO when K << N, trading off some bias
    for reduced computation. Predictive moments include observation noise.

    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Parameter bounds
        k: Number of folds
        model_factory: Optional custom model builder, applied per fold

    Returns:
        CVMetrics
    """
    n_samples = train_x.shape[0]

    if k > n_samples:
        k = n_samples  # Fall back to LOO

    factory = model_factory if model_factory is not None else _make_default_model_factory(bounds)

    indices = _fold_permutation(train_x, train_y)
    fold_sizes = [n_samples // k + (1 if i < n_samples % k else 0) for i in range(k)]

    all_predictions = torch.zeros_like(train_y.squeeze())
    all_variances = torch.zeros_like(train_y.squeeze())
    predicted = torch.zeros(n_samples, dtype=torch.bool)
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
            model = factory(train_x_fold, train_y_fold)

            model.eval()
            with torch.no_grad():
                posterior = model.posterior(test_x_fold, observation_noise=True)
                pred_mean = posterior.mean.squeeze()
                pred_var = posterior.variance.squeeze()

            all_predictions[test_indices] = pred_mean
            all_variances[test_indices] = pred_var
            predicted[test_indices] = True
            per_fold_errors.append((pred_mean - test_y_fold.squeeze()).abs().mean().item())

        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning("K-fold CV failed for fold %s: %s", fold_idx, e)
            continue

    # Score only rows a fold actually predicted — zero-filled placeholders
    # from failed folds must never enter the metrics as real predictions.
    if int(predicted.sum().item()) < 2:
        return _create_nan_metrics(f"{k}fold_failed")

    return _compute_cv_metrics_from_predictions(
        all_predictions[predicted],
        all_variances[predicted],
        train_y.squeeze()[predicted],
        per_fold_errors,
        f"{k}fold",
    )


def _fold_permutation(train_x: Tensor, train_y: Tensor) -> Tensor:
    """Deterministic, data-derived sample permutation for K-fold splits.

    Deriving the fold assignment from the data (instead of the global RNG)
    makes K-fold comparisons paired — every model configuration evaluated
    on the same dataset is scored on identical train/test splits, so metric
    differences reflect the models rather than fold-sampling noise — and
    keeps CV from consuming or depending on global torch RNG state.
    """
    digest = hashlib.sha256(
        train_x.detach().cpu().numpy().tobytes() + train_y.detach().cpu().numpy().tobytes()
    ).digest()
    seed = int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(train_x.shape[0], generator=generator)


def compute_cv_for_model_list(
    model: ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    config: CVConfig | None = None,
) -> dict[int, CVMetrics]:
    """Compute CV metrics for each objective's fitted sub-model.

    Each sub-model is validated via the exact LOO downdate at its own
    fitted hyperparameters (the model-form contract of
    :func:`compute_loo_cv_optimized`); refit-based CV methods require the
    tensor form instead.

    Args:
        model: Fitted ModelListGP
        train_x: Training inputs
        train_y: Training outputs (n_samples, n_objectives)
        bounds: Parameter bounds (unused for the downdate; kept for API
            compatibility)
        config: CV configuration

    Returns:
        Dictionary mapping objective index to CVMetrics
    """
    del bounds  # the fitted sub-models carry their own input transforms
    results = {}

    for i, sub_model in enumerate(model.models):
        train_y_i = train_y[:, i : i + 1]
        metrics = compute_loo_cv_optimized(
            cast("SingleTaskGP", sub_model), train_x, train_y_i, config
        )
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
    model_factory_key: str | None = None,
) -> str:
    """Compute a cache key for CV results.

    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Parameter bounds
        config: CV configuration
        model_factory_key: Identifier of the model configuration under CV, so
            metrics computed for one candidate are never served for another

    Returns:
        Cache key string
    """
    # Create hash of inputs
    x_bytes = train_x.detach().cpu().numpy().tobytes()
    y_bytes = train_y.detach().cpu().numpy().tobytes()
    b_bytes = bounds.detach().cpu().numpy().tobytes()

    # Every config field that changes which computation runs must be part of
    # the key, or one method's cached metrics get served for another.
    config_str = (
        f"{config.method}_{config.use_approximate}_{config.approximate_threshold}"
        f"_{config.k_folds}_{model_factory_key}"
    )

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
    if method == "approximate_loo":
        return base_fit_time * 1.5
    if method == "kfold":
        k = k_folds if k_folds is not None else 5
        return k * base_fit_time
    if method == "auto":
        # Return estimate for recommended method
        if n_samples > 100:
            return base_fit_time * 1.5  # approximate_loo
        return n_samples * base_fit_time  # batch_loo
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
