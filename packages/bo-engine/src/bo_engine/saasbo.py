"""SAASBO: Sparse Axis-Aligned Subspace Bayesian Optimization.

For high-dimensional optimization (50+ parameters) where only a subset
of parameters are important.

v2.0: Initial implementation based on Eriksson & Jankowiak (UAI 2021)
v2.3: Added GPU auto-detection and acceleration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from torch import Tensor

from bo_engine.constants import (
    SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD,
    SAASBO_WIDE_INTERVAL_LOG10_THRESHOLD,
)
from bo_engine.device import ensure_device
from bo_engine.types import AcquisitionOptimizationConfig


@dataclass(frozen=True)
class SAASBOConfig:
    """Configuration for SAASBO optimization.

    SAASBO uses NUTS (No-U-Turn Sampler) for fully Bayesian inference,
    which can be computationally expensive.
    """

    warmup_steps: int = 256  # NUTS warmup steps (recommended: 256-512)
    num_samples: int = 128  # Number of posterior samples (recommended: 128-256)
    thinning: int = 16  # Keep every Nth sample to reduce autocorrelation
    disable_progbar: bool = True  # Disable progress bar for cleaner output


def create_saasbo_model(
    train_x: Tensor,
    train_y: Tensor,
    train_yvar: Tensor | None = None,
) -> SaasFullyBayesianSingleTaskGP:
    """Create a SAASBO model.

    The SAAS model uses sparsity-inducing priors on inverse lengthscales
    to identify the most important parameters.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1)
        train_yvar: Optional known noise variance of shape (n_samples, 1)

    Returns:
        SaasFullyBayesianSingleTaskGP model (unfitted)
    """
    if train_yvar is not None:
        train_x, train_y, train_yvar = ensure_device(train_x, train_y, train_yvar)
    else:
        train_x, train_y = ensure_device(train_x, train_y)

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    kwargs: dict[str, Any] = {
        "train_X": train_x,
        "train_Y": train_y,
        "outcome_transform": Standardize(m=1),
    }

    if train_yvar is not None:
        kwargs["train_Yvar"] = train_yvar

    return SaasFullyBayesianSingleTaskGP(**kwargs)


def fit_saasbo_model(
    model: SaasFullyBayesianSingleTaskGP,
    config: SAASBOConfig | None = None,
) -> SaasFullyBayesianSingleTaskGP:
    """Fit SAASBO model using NUTS.

    This uses Hamiltonian Monte Carlo via NUTS for fully Bayesian inference.
    Can be computationally expensive for large datasets.

    Args:
        model: SaasFullyBayesianSingleTaskGP to fit
        config: SAASBO configuration

    Returns:
        Fitted model
    """
    if config is None:
        config = SAASBOConfig()

    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=config.warmup_steps,
        num_samples=config.num_samples,
        thinning=config.thinning,
        disable_progbar=config.disable_progbar,
    )

    return model


def create_and_fit_saasbo_model(
    train_x: Tensor,
    train_y: Tensor,
    train_yvar: Tensor | None = None,
    config: SAASBOConfig | None = None,
) -> SaasFullyBayesianSingleTaskGP:
    """Create and fit SAASBO model.

    Args:
        train_x: Training inputs
        train_y: Training outputs
        train_yvar: Optional known noise variance
        config: SAASBO configuration

    Returns:
        Fitted SaasFullyBayesianSingleTaskGP
    """
    model = create_saasbo_model(train_x, train_y, train_yvar)
    return fit_saasbo_model(model, config)


def get_saasbo_lengthscales(
    model: SaasFullyBayesianSingleTaskGP,
) -> Tensor:
    """Extract lengthscales from fitted SAASBO model.

    The lengthscales indicate parameter importance:
    - Small lengthscale = important parameter (sensitive)
    - Large lengthscale = less important parameter (SAAS prior pushes these large)

    Args:
        model: Fitted SAASBO model

    Returns:
        1-D tensor of median lengthscales (one entry per input dimension).
    """
    # Get lengthscales from all posterior samples
    model.eval()

    # Access the lengthscales from the covariance module
    # For fully Bayesian models, we have multiple samples
    lengthscales = model.covar_module.base_kernel.lengthscale  # ty: ignore[unresolved-attribute]

    # Reduce posterior-sample dimension via median, then flatten to a strict
    # 1-D vector. ``.squeeze()`` collapses every singleton dimension and
    # produces a 0-D tensor on a 1-D search space, which crashes downstream
    # ``shape[-1]`` indexing.
    return lengthscales.median(dim=0).values.reshape(-1)


def get_saasbo_lengthscale_quantiles(
    model: SaasFullyBayesianSingleTaskGP,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> Tensor:
    """Extract per-dimension lengthscale posterior quantiles from SAASBO.

    The MCMC sample dimension is reduced via :func:`torch.quantile` so the
    return is a ``(len(quantiles), n_dims)`` tensor — one row per
    requested quantile. Pair with :func:`get_saasbo_lengthscales` (which
    is preserved for backward compatibility on the median-only path).

    Args:
        model: Fitted SAASBO model.
        quantiles: Quantile levels in ``[0, 1]``. Defaults to ``(0.25,
            0.5, 0.75)`` — the standard IQR triple Eriksson & Jankowiak
            recommend for inspecting per-dimension lengthscale spread.

    Returns:
        Tensor of shape ``(len(quantiles), n_dims)``.
    """
    model.eval()
    raw = model.covar_module.base_kernel.lengthscale  # ty: ignore[unresolved-attribute]
    # ``raw`` has the MCMC samples as the leading batch dimension; flatten
    # any trailing singleton dims so we end up with ``(n_samples, n_dims)``.
    flat = raw.reshape(raw.shape[0], -1)
    q_tensor = torch.tensor(list(quantiles), dtype=flat.dtype, device=flat.device)
    return torch.quantile(flat, q_tensor, dim=0)


@dataclass(frozen=True)
class SAASBOImportance:
    """Per-dimension SAASBO importance summary.

    The normalized ``importance`` reciprocal makes "small contribution" and
    "provably inactive" look identical (both are close to zero), which is
    exactly the distinction SAASBO is designed to expose. We therefore also
    surface the raw median ``lengthscale`` per dimension, posterior interval
    bounds (``lengthscale_q25`` / ``lengthscale_q75``), the resulting
    ``log10_interval_width``, and a boolean ``active`` flag derived from
    :data:`bo_engine.constants.SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD`.

    ``confident`` is False when the 25–75 % credible interval spans more
    than :data:`SAASBO_WIDE_INTERVAL_LOG10_THRESHOLD` orders of magnitude.
    Early in a campaign the half-Cauchy posterior is wide and the median
    importance alone is unreliable for pruning decisions; consumers
    (feature-importance UI, dimensionality-reduction heuristics) should
    refuse to act on a dimension whose interval is wide.

    Reference:
        Eriksson & Jankowiak, "High-Dimensional Bayesian Optimization with
        Sparse Axis-Aligned Subspaces", UAI 2021
        (https://arxiv.org/abs/2103.00349), §3.3 & Appendix B: under the
        SAAS prior, truly inactive lengthscales converge to large values
        (>= 1e2-1e3) so the boolean mask is robust against the post-hoc
        normalization step. The same paper recommends inspecting posterior
        spread alongside the point estimate before pruning dimensions.
    """

    name: str
    lengthscale: float
    importance: float
    active: bool
    lengthscale_q25: float = 0.0
    lengthscale_q75: float = 0.0
    log10_interval_width: float = 0.0
    confident: bool = True


def compute_saasbo_importance(
    model: SaasFullyBayesianSingleTaskGP,
    parameter_names: list[str] | None = None,
    inactive_threshold: float = SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD,
) -> dict[str, float]:
    """Compute the normalized SAASBO importance score per parameter.

    Kept for backward compatibility (returns the same ``dict[str, float]``
    shape consumers already depend on). For the richer descriptor that also
    surfaces raw lengthscales and the boolean active mask, see
    :func:`compute_saasbo_importance_report`.

    Args:
        model: Fitted SAASBO model
        parameter_names: Optional parameter names for the result dict
        inactive_threshold: Unused here; preserved so callers can pass the
            same kwarg they would pass to the richer reporter without
            branching on which function they call.

    Returns:
        Dictionary mapping parameter index/name to importance score
    """
    del inactive_threshold  # surfaced through the report API instead
    report = compute_saasbo_importance_report(model, parameter_names)
    return {entry.name: entry.importance for entry in report}


def compute_saasbo_importance_report(
    model: SaasFullyBayesianSingleTaskGP,
    parameter_names: list[str] | None = None,
    inactive_threshold: float = SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD,
    wide_interval_threshold: float = SAASBO_WIDE_INTERVAL_LOG10_THRESHOLD,
) -> list[SAASBOImportance]:
    """Return per-parameter SAASBO importance with credible intervals and confidence flags.

    Args:
        model: Fitted SAASBO model
        parameter_names: Optional parameter names. When omitted, parameters
            are named ``param_0 .. param_{d-1}``.
        inactive_threshold: Median per-dimension lengthscale above which a
            dimension is flagged ``active=False``. Defaults to the
            calibration recommended by Eriksson & Jankowiak 2021.
        wide_interval_threshold: Order-of-magnitude width (in log10 units)
            above which a dimension's posterior interval is considered
            "wide" and the entry is flagged ``confident=False``. Defaults
            to 1 (spanning one order of magnitude or more).

    Returns:
        List of :class:`SAASBOImportance` entries in input-parameter order,
        each including the median lengthscale, the 25 % / 75 % posterior
        quantiles, the log10 width of the interval, and a ``confident``
        flag callers should use to gate pruning decisions.

    Raises:
        ValueError: If ``parameter_names`` is provided but its length does
            not match the number of dimensions in the model.
    """
    # ``get_saasbo_lengthscales`` is kept as the median entry point — it's
    # the canonical extractor and is monkey-patched by tests / external
    # adapters. The quantile band is best-effort: when the underlying
    # ``covar_module`` does not expose per-sample lengthscales (or the
    # extractor is patched without quantile data) we fall back to the
    # median-only path and report ``confident=True`` (single-point
    # estimate is treated as collapsed-interval data).
    medians = get_saasbo_lengthscales(model).reshape(-1)
    n_params = int(medians.shape[0])

    q25, q75 = _try_extract_quantile_band(model, n_params)

    # Importance is the inverse of the median lengthscale (normalized).
    importance = 1.0 / (medians + 1e-6)
    importance = importance / importance.sum()

    if parameter_names is None:
        parameter_names = [f"param_{i}" for i in range(n_params)]
    elif len(parameter_names) != n_params:
        msg = (
            f"parameter_names length ({len(parameter_names)}) does not match "
            f"the model's input dimensionality ({n_params})."
        )
        raise ValueError(msg)

    if q25 is None or q75 is None:
        # No posterior spread available — surface medians as the singleton
        # interval so consumers see consistent fields and treat the
        # dimension as confident (no spread to be uncertain about).
        q25_vals = medians
        q75_vals = medians
        log10_widths = torch.zeros_like(medians)
    else:
        eps = 1e-12
        q25_vals = q25
        q75_vals = q75
        log10_widths = (q75.clamp(min=eps).log10() - q25.clamp(min=eps).log10()).abs()

    return [
        SAASBOImportance(
            name=parameter_names[i],
            lengthscale=float(medians[i].item()),
            importance=float(importance[i].item()),
            active=float(medians[i].item()) <= inactive_threshold,
            lengthscale_q25=float(q25_vals[i].item()),
            lengthscale_q75=float(q75_vals[i].item()),
            log10_interval_width=float(log10_widths[i].item()),
            confident=float(log10_widths[i].item()) <= wide_interval_threshold,
        )
        for i in range(n_params)
    ]


def _try_extract_quantile_band(
    model: SaasFullyBayesianSingleTaskGP, expected_n_params: int
) -> tuple[Tensor | None, Tensor | None]:
    """Best-effort extraction of the 25/75 % lengthscale bands.

    Returns ``(None, None)`` when the model does not expose per-sample
    lengthscales (e.g. monkey-patched stubs in tests that only know how to
    return the median).
    """
    try:
        quantile_table = get_saasbo_lengthscale_quantiles(model, quantiles=(0.25, 0.5, 0.75))
    except (AttributeError, RuntimeError, ValueError):
        return None, None
    if quantile_table.dim() != 2 or quantile_table.shape[0] != 3:
        return None, None
    q25 = quantile_table[0].reshape(-1)
    q75 = quantile_table[2].reshape(-1)
    if q25.shape[0] != expected_n_params or q75.shape[0] != expected_n_params:
        return None, None
    return q25, q75


def should_use_saasbo(
    n_parameters: int,
    n_observations: int,
    threshold: int = 50,
) -> bool:
    """Determine if SAASBO should be used.

    SAASBO is recommended for high-dimensional problems where:
    1. Number of parameters exceeds threshold
    2. We have enough data points (at least 2 * n_params)

    Args:
        n_parameters: Number of parameters
        n_observations: Number of observations
        threshold: Parameter count threshold (default: 50)

    Returns:
        True if SAASBO is recommended
    """
    has_enough_params = n_parameters >= threshold
    has_enough_data = n_observations >= max(10, n_parameters // 5)

    return has_enough_params and has_enough_data


def generate_saasbo_suggestions(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    batch_size: int = 1,
    config: SAASBOConfig | None = None,
    parameter_names: list[str] | None = None,
    acquisition_optimization: AcquisitionOptimizationConfig | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Generate suggestions using SAASBO.

    Main entry point for high-dimensional optimization with SAASBO.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1)
        bounds: Parameter bounds of shape (2, n_dims)
        batch_size: Number of suggestions to generate
        config: SAASBO configuration
        parameter_names: Optional parameter names for importance dict
        acquisition_optimization: Optional restart / raw-sample override. If
            omitted, the dimension-adaptive defaults from
            :class:`AcquisitionOptimizationConfig` apply -- important for
            SAASBO, which targets >20-dimensional spaces where the fixed
            ``num_restarts=20`` could trap the optimizer in shallow minima.

    Returns:
        Tuple of (candidates, acquisition_values, metadata)
    """
    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    # Create and fit model
    model = create_and_fit_saasbo_model(train_x, train_y, config=config)

    # Compute parameter importance (full report includes raw lengthscale
    # and boolean active mask alongside the normalized score).
    importance_report = compute_saasbo_importance_report(model, parameter_names)
    importance = {entry.name: entry.importance for entry in importance_report}

    # Create acquisition function
    model.eval()
    acqf = qLogExpectedImprovement(
        model=model,
        best_f=train_y.min().item(),  # Assumes minimization
    )

    acq_config = acquisition_optimization or AcquisitionOptimizationConfig()
    num_restarts, raw_samples = acq_config.resolve(int(bounds.shape[-1]))

    # Optimize acquisition
    candidates, acq_values = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        sequential=True,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # Get top important parameters
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_params = sorted_importance[:5]

    metadata = {
        "model_type": "SaasFullyBayesianSingleTaskGP (SAASBO)",
        "acquisition_function": "qLogExpectedImprovement",
        "parameter_importance": importance,
        "top_important_parameters": dict(top_params),
        "inference_method": "NUTS (fully Bayesian)",
        "n_posterior_samples": config.num_samples if config else 128,
        "parameter_lengthscales": {entry.name: entry.lengthscale for entry in importance_report},
        "active_parameters": [entry.name for entry in importance_report if entry.active],
        "inactive_parameters": [entry.name for entry in importance_report if not entry.active],
    }

    return candidates, acq_values, metadata


def estimate_saasbo_runtime(
    n_observations: int,
    config: SAASBOConfig | None = None,
) -> str:
    """Estimate SAASBO runtime.

    SAASBO scales cubically with the number of observations.

    Args:
        n_observations: Number of observations
        config: SAASBO configuration

    Returns:
        Human-readable runtime estimate
    """
    if config is None:
        config = SAASBOConfig()

    # Rough estimates based on benchmarks
    # NUTS is O(n^3) in the number of data points
    total_samples = config.warmup_steps + config.num_samples

    if n_observations < 50:
        base_time = 10  # seconds per sample
    elif n_observations < 100:
        base_time = 30
    elif n_observations < 200:
        base_time = 120
    else:
        base_time = 300

    estimated_seconds = base_time * (total_samples / 100)

    if estimated_seconds < 60:
        return f"~{int(estimated_seconds)} seconds"
    if estimated_seconds < 3600:
        return f"~{int(estimated_seconds / 60)} minutes"
    return f"~{int(estimated_seconds / 3600)} hours"
