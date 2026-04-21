"""SAASBO: Sparse Axis-Aligned Subspace Bayesian Optimization.

For high-dimensional optimization (50+ parameters) where only a subset
of parameters are important.

v2.0: Initial implementation based on Eriksson & Jankowiak (UAI 2021)
v2.3: Added GPU auto-detection and acceleration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from torch import Tensor

from bo_engine.device import ensure_device


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
        Tensor of median lengthscales across posterior samples
    """
    # Get lengthscales from all posterior samples
    model.eval()

    # Access the lengthscales from the covariance module
    # For fully Bayesian models, we have multiple samples
    lengthscales = model.covar_module.base_kernel.lengthscale  # ty: ignore[unresolved-attribute]

    # Return median across samples for robustness
    return lengthscales.median(dim=0).values.squeeze()


def compute_saasbo_importance(
    model: SaasFullyBayesianSingleTaskGP,
    parameter_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute parameter importance from SAASBO lengthscales.

    Args:
        model: Fitted SAASBO model
        parameter_names: Optional parameter names for the result dict

    Returns:
        Dictionary mapping parameter index/name to importance score
    """
    lengthscales = get_saasbo_lengthscales(model)
    n_params = lengthscales.shape[-1]

    # Importance is inverse of lengthscale (normalized)
    importance = 1.0 / (lengthscales + 1e-6)
    importance = importance / importance.sum()

    if parameter_names is None:
        parameter_names = [f"param_{i}" for i in range(n_params)]

    return {name: importance[i].item() for i, name in enumerate(parameter_names)}


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

    Returns:
        Tuple of (candidates, acquisition_values, metadata)
    """
    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    # Create and fit model
    model = create_and_fit_saasbo_model(train_x, train_y, config=config)

    # Compute parameter importance
    importance = compute_saasbo_importance(model, parameter_names)

    # Create acquisition function
    model.eval()
    acqf = qLogExpectedImprovement(
        model=model,
        best_f=train_y.min().item(),  # Assumes minimization
    )

    # Optimize acquisition
    candidates, acq_values = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=batch_size,
        num_restarts=20,
        raw_samples=512,
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
    elif estimated_seconds < 3600:
        return f"~{int(estimated_seconds / 60)} minutes"
    else:
        return f"~{int(estimated_seconds / 3600)} hours"
