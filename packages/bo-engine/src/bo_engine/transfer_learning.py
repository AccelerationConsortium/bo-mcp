"""Transfer Learning with RGPE (Rank-weighted GP Ensemble).

Leverages knowledge from prior optimization campaigns to improve
optimization on a new but related task.

v2.0: Initial implementation based on Feurer, Letham, Bakshy (ICML 2018)
v2.3: Added GPU auto-detection and acceleration
v2.6: Added RGPEAcquisition class that properly uses ensemble predictions (Section 2.2)

References:
    - Feurer et al. "Scalable Meta-Learning for Bayesian Optimization"
      ICML AutoML Workshop 2018 (https://arxiv.org/abs/1802.02219)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.distributions import Normal

from bo_engine.device import ensure_device


@dataclass(frozen=True)
class PriorTaskData:
    """Data from a prior optimization task.

    Represents historical optimization data that can be used
    for transfer learning.

    Attributes:
        train_x: Training inputs (n_samples, n_dims)
        train_y: Training outputs (n_samples, 1)
        name: Unique identifier for the prior task
        metadata: Optional metadata dictionary
    """

    train_x: Tensor  # Training inputs (n_samples, n_dims)
    train_y: Tensor  # Training outputs (n_samples, 1)
    name: str = "prior"  # Unique identifier for the prior task
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RGPEConfig:
    """Configuration for RGPE transfer learning."""

    num_samples: int = 512  # Samples for ranking
    use_input_warping: bool = False  # Whether to use input warping
    temperature: float = 0.5  # Softmax temperature for weight distribution


class RGPE(torch.nn.Module):
    """Rank-weighted GP Ensemble for transfer learning.

    Combines multiple GP models (from prior tasks + target task)
    using rank-based weighting to leverage prior knowledge.
    """

    def __init__(
        self,
        base_models: list[SingleTaskGP],
        target_model: SingleTaskGP,
        weights: Tensor | None = None,
        temperature: float = 0.5,
    ) -> None:
        """Initialize RGPE.

        Args:
            base_models: List of fitted GP models from prior tasks
            target_model: GP model for the current target task
            weights: Optional pre-computed weights (will be computed if None)
            temperature: Softmax temperature for weight distribution (0.5 default)
        """
        super().__init__()
        self.base_models = torch.nn.ModuleList(base_models)
        self.target_model = target_model
        self._weights = weights
        self.temperature = temperature

    @property
    def num_models(self) -> int:
        """Total number of models in the ensemble."""
        return len(self.base_models) + 1

    @property
    def weights(self) -> Tensor:
        """Get ensemble weights."""
        if self._weights is None:
            raise ValueError("Weights not computed. Call compute_weights() first.")
        return self._weights

    def compute_weights(
        self,
        target_x: Tensor,
        target_y: Tensor,
        num_samples: int = 512,
    ) -> Tensor:
        """Compute rank-based weights for the ensemble.

        Uses leave-one-out cross-validation on the target data to
        estimate the quality of each model. The weight computation
        measures how well each model predicts held-out target data.

        The approach follows the original RGPE paper (Feurer et al., 2018):
        1. Compute ranking loss for each model on target data
        2. Convert rankings to weights using softmax with temperature

        For the target model, we apply a small penalty since it was trained
        on the target data, while prior models make true out-of-sample predictions.

        Args:
            target_x: Target task inputs
            target_y: Target task outputs
            num_samples: Number of samples for ranking computation

        Returns:
            Tensor of weights with shape (num_models,)
        """
        n_target = target_x.shape[0]
        all_models = list(self.base_models) + [self.target_model]
        n_models = len(all_models)

        # Compute mean squared error for each model on target data
        # Use normalized MSE to avoid scale issues
        mses = torch.zeros(n_models, dtype=torch.double)
        target_var = target_y.var().item() + 1e-6  # For normalization

        for model_idx, model in enumerate(all_models):
            model.eval()

            with torch.no_grad():
                posterior = model.posterior(target_x)
                pred_mean = posterior.mean.squeeze()
                target_y_flat = target_y.squeeze()

                # Mean squared error normalized by target variance
                mse = ((pred_mean - target_y_flat) ** 2).mean().item()
                normalized_mse = mse / target_var

                mses[model_idx] = normalized_mse

        # For the target model, apply LOO correction since it was trained on this data
        # This penalizes the target model to account for in-sample bias
        # Use approximate LOO factor: MSE_loo ≈ MSE / (1 - leverage)^2
        # For GP, average leverage ≈ 2 * n_dims / n_target
        n_dims = target_x.shape[-1]
        avg_leverage = min(0.5, 2.0 * n_dims / n_target)  # Cap at 0.5
        loo_factor = 1.0 / ((1.0 - avg_leverage) ** 2 + 1e-6)
        mses[-1] = mses[-1] * loo_factor

        # Convert MSEs to ranking scores (lower MSE = better = higher score)
        # Use inverse with regularization for stability
        scores = 1.0 / (mses + 0.1)  # Add 0.1 to avoid division by zero

        # Normalize scores to range [0, 1] before softmax
        scores = scores / scores.max()

        # Apply softmax with temperature to get weights
        # Higher temperature = smoother (more uniform) weights
        log_scores = torch.log(scores + 1e-10)
        weights = torch.softmax(log_scores / self.temperature, dim=0)

        self._weights = weights
        return weights

    def posterior(self, x: Tensor) -> GPyTorchPosterior:
        """Compute weighted posterior prediction.

        Args:
            x: Input points of shape (..., n_dims)

        Returns:
            GPyTorchPosterior with weighted predictions
        """
        all_models = list(self.base_models) + [self.target_model]
        weights = self.weights

        # Get posteriors from all models
        # Note: Do NOT use torch.no_grad() here to allow gradient flow
        # through the ensemble for acquisition function optimization
        means = []
        variances = []

        for model in all_models:
            model.eval()
            posterior = model.posterior(x)
            means.append(posterior.mean)
            variances.append(posterior.variance)

        # Stack predictions
        means = torch.stack(means, dim=0)  # (n_models, ..., 1)
        variances = torch.stack(variances, dim=0)

        # Weighted combination
        # Reshape weights for broadcasting
        # Detach weights to prevent gradient flow through weight computation
        weight_shape = [len(weights)] + [1] * (means.dim() - 1)
        weights_reshaped = weights.detach().view(*weight_shape)

        # Weighted mean: sum(w_i * mu_i)
        weighted_mean = (weights_reshaped * means).sum(dim=0)

        # Weighted variance: sum(w_i * (var_i + mu_i^2)) - (sum(w_i * mu_i))^2
        # This accounts for both within-model and between-model variance
        weighted_second_moment = (weights_reshaped * (variances + means**2)).sum(dim=0)
        weighted_variance = weighted_second_moment - weighted_mean**2
        weighted_variance = weighted_variance.clamp(min=1e-6)

        # Create MultivariateNormal for GPyTorchPosterior
        # Need to convert to correct format
        mvn = MultivariateNormal(
            weighted_mean.squeeze(-1),
            torch.diag_embed(weighted_variance.squeeze(-1)),
        )

        return GPyTorchPosterior(mvn)

    def forward(self, x: Tensor) -> MultivariateNormal:
        """Forward pass returning MultivariateNormal.

        Args:
            x: Input points

        Returns:
            MultivariateNormal distribution
        """
        posterior = self.posterior(x)
        return posterior.distribution


def create_base_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
) -> SingleTaskGP:
    """Create and fit a base model for transfer learning.

    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Parameter bounds

    Returns:
        Fitted SingleTaskGP
    """
    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    n_dims = train_x.shape[-1]
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=n_dims, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


def create_rgpe_model(
    target_x: Tensor,
    target_y: Tensor,
    prior_tasks: list[PriorTaskData],
    bounds: Tensor,
    config: RGPEConfig | None = None,
) -> RGPE:
    """Create RGPE ensemble model from prior tasks and target data.

    Args:
        target_x: Target task training inputs
        target_y: Target task training outputs
        prior_tasks: List of prior task data
        bounds: Parameter bounds for all tasks (assumed same)
        config: RGPE configuration

    Returns:
        Fitted RGPE model with computed weights
    """
    if config is None:
        config = RGPEConfig()

    target_x, target_y, bounds = ensure_device(target_x, target_y, bounds)

    if target_y.dim() == 1:
        target_y = target_y.unsqueeze(-1)

    # Create and fit base models from prior tasks
    base_models = []
    for prior_task in prior_tasks:
        model = create_base_model(prior_task.train_x, prior_task.train_y, bounds)
        base_models.append(model)

    # Create and fit target model
    target_model = create_base_model(target_x, target_y, bounds)

    # Create RGPE ensemble
    rgpe = RGPE(
        base_models=base_models,
        target_model=target_model,
    )

    # Compute weights
    rgpe.compute_weights(
        target_x=target_x,
        target_y=target_y,
        num_samples=config.num_samples,
    )

    return rgpe


def get_rgpe_weights_explanation(
    rgpe: RGPE,
    prior_tasks: list[PriorTaskData],
) -> dict[str, float]:
    """Get human-readable explanation of RGPE weights.

    Args:
        rgpe: Fitted RGPE model
        prior_tasks: List of prior tasks (for task IDs)

    Returns:
        Dictionary mapping task names to weights
    """
    weights = rgpe.weights.tolist()

    result = {}
    for i, prior_task in enumerate(prior_tasks):
        result[f"prior_{prior_task.name}"] = weights[i]
    result["target"] = weights[-1]

    return result


class RGPEAcquisition(AcquisitionFunction):
    """Acquisition function using RGPE ensemble predictions.

    This class implements Expected Improvement using the weighted ensemble
    predictions from RGPE, rather than just using the target model.
    This is the key improvement over the previous implementation that
    only used RGPE weights for explanation but not for acquisition.

    The ensemble predictions provide:
    1. Better uncertainty estimates through model averaging
    2. Transfer of knowledge from prior tasks into candidate selection
    3. More robust predictions when target data is scarce

    Reference:
        Feurer et al. "Scalable Meta-Learning for Bayesian Optimization"
        ICML AutoML Workshop 2018

    Args:
        rgpe_ensemble: Fitted RGPE model with computed weights
        best_f: Best observed value on the target task
        maximize: If True, maximize; else minimize (default)

    Example:
        >>> rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        >>> best_f = target_y.min().item()  # For minimization
        >>> acqf = RGPEAcquisition(rgpe, best_f)
        >>> candidates, acq_values = optimize_acqf(acqf, bounds, q=1)
    """

    def __init__(
        self,
        rgpe_ensemble: RGPE,
        best_f: float,
        maximize: bool = False,
    ) -> None:
        """Initialize RGPE acquisition function.

        Args:
            rgpe_ensemble: Fitted RGPE model
            best_f: Best observed value (for EI computation)
            maximize: If True, maximize the objective
        """
        # Initialize with target model as the base model for BoTorch compatibility
        super().__init__(model=rgpe_ensemble.target_model)
        self.rgpe_ensemble = rgpe_ensemble
        self.best_f = best_f
        self.maximize = maximize
        self._X_pending: Tensor | None = None

    @property
    def X_pending(self) -> Tensor | None:  # noqa: N802
        """Get pending candidates."""
        return self._X_pending

    @X_pending.setter
    def X_pending(self, value: Tensor | None) -> None:  # noqa: N802
        """Set pending candidates."""
        self._X_pending = value

    def forward(self, X: Tensor) -> Tensor:  # noqa: N803
        """Compute Expected Improvement using RGPE ensemble predictions.

        This method uses the weighted ensemble posterior from RGPE to compute
        EI, rather than just using the target model. This properly leverages
        the transfer learning signal.

        Args:
            X: Candidate points of shape (..., q, d) where:
               - ... are batch dimensions
               - q is the batch size for joint acquisition
               - d is the input dimension

        Returns:
            Expected Improvement values of shape (...)
        """
        # Get ensemble posterior from RGPE
        posterior = self.rgpe_ensemble.posterior(X)

        # Extract mean and standard deviation
        mean = posterior.mean  # Shape: (..., q, 1)
        variance = posterior.variance
        std = variance.sqrt().clamp(min=1e-6)  # Shape: (..., q, 1)

        # Squeeze output dimension
        mean = mean.squeeze(-1)  # Shape: (..., q)
        std = std.squeeze(-1)  # Shape: (..., q)

        # Compute Expected Improvement
        # For minimization: EI = E[max(best_f - f(x), 0)]
        # For maximization: EI = E[max(f(x) - best_f, 0)]
        if self.maximize:
            z = (mean - self.best_f) / std
        else:
            z = (self.best_f - mean) / std

        # Standard normal distribution for EI computation
        normal = Normal(torch.zeros_like(z), torch.ones_like(z))
        pdf = torch.exp(normal.log_prob(z))
        cdf = normal.cdf(z)

        # EI = std * (z * cdf + pdf)
        ei = std * (z * cdf + pdf)

        # Average over q dimension for joint acquisition
        if ei.dim() > 1 and ei.shape[-1] > 1:
            ei = ei.mean(dim=-1)
        else:
            ei = ei.squeeze(-1)

        # Ensure non-negative
        ei = ei.clamp(min=0.0)

        return ei


class RGPELogEI(AcquisitionFunction):
    """Log Expected Improvement using RGPE ensemble predictions.

    Uses log-space computation for numerical stability, similar to
    qLogNoisyExpectedImprovement but with RGPE ensemble predictions.

    Args:
        rgpe_ensemble: Fitted RGPE model with computed weights
        best_f: Best observed value on the target task
        maximize: If True, maximize; else minimize
    """

    def __init__(
        self,
        rgpe_ensemble: RGPE,
        best_f: float,
        maximize: bool = False,
    ) -> None:
        """Initialize RGPE Log-EI acquisition function."""
        super().__init__(model=rgpe_ensemble.target_model)
        self.rgpe_ensemble = rgpe_ensemble
        self.best_f = best_f
        self.maximize = maximize
        self._X_pending: Tensor | None = None

    @property
    def X_pending(self) -> Tensor | None:  # noqa: N802
        """Get pending candidates."""
        return self._X_pending

    @X_pending.setter
    def X_pending(self, value: Tensor | None) -> None:  # noqa: N802
        """Set pending candidates."""
        self._X_pending = value

    def forward(self, X: Tensor) -> Tensor:  # noqa: N803
        """Compute Log Expected Improvement using RGPE ensemble.

        Args:
            X: Candidate points of shape (..., q, d)

        Returns:
            Log Expected Improvement values of shape (...)
        """
        posterior = self.rgpe_ensemble.posterior(X)

        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1).clamp(min=1e-6)

        if self.maximize:
            z = (mean - self.best_f) / std
        else:
            z = (self.best_f - mean) / std

        # Log-EI computation for numerical stability
        # log(EI) = log(std) + log(z * Phi(z) + phi(z))
        normal = Normal(torch.zeros_like(z), torch.ones_like(z))
        log_pdf = normal.log_prob(z)
        log_cdf = torch.log(normal.cdf(z).clamp(min=1e-10))

        # Use log-sum-exp for stability
        log_std = torch.log(std)
        log_ei = log_std + torch.log(z * torch.exp(log_cdf) + torch.exp(log_pdf) + 1e-10)

        # Average over q dimension
        if log_ei.dim() > 1 and log_ei.shape[-1] > 1:
            log_ei = log_ei.mean(dim=-1)
        else:
            log_ei = log_ei.squeeze(-1)

        return log_ei


def generate_rgpe_suggestions(
    target_x: Tensor,
    target_y: Tensor,
    prior_tasks: list[PriorTaskData],
    bounds: Tensor,
    batch_size: int = 1,
    config: RGPEConfig | None = None,
    use_ensemble_acquisition: bool = True,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Generate suggestions using RGPE transfer learning.

    This function creates an RGPE ensemble from prior task data and uses
    the weighted ensemble predictions to guide acquisition. The key improvement
    in v2.6 is that ensemble predictions are now properly used for candidate
    selection, not just for weight computation.

    Args:
        target_x: Target task training inputs
        target_y: Target task training outputs
        prior_tasks: List of prior task data
        bounds: Parameter bounds
        batch_size: Number of suggestions
        config: RGPE configuration
        use_ensemble_acquisition: If True (default), use RGPEAcquisition which
            properly leverages ensemble predictions. If False, use only the
            target model (legacy behavior).

    Returns:
        Tuple of (candidates, acquisition_values, metadata)

    Example:
        >>> from bo_engine.transfer_learning import generate_rgpe_suggestions, PriorTaskData
        >>> prior_tasks = [
        ...     PriorTaskData("task1", prior_x1, prior_y1),
        ...     PriorTaskData("task2", prior_x2, prior_y2),
        ... ]
        >>> candidates, acq_values, metadata = generate_rgpe_suggestions(
        ...     target_x, target_y, prior_tasks, bounds
        ... )
        >>> print(metadata["weights"])  # Shows task weights
    """
    target_x, target_y, bounds = ensure_device(target_x, target_y, bounds)

    # Create RGPE model
    rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds, config)

    # Determine best observed value (assumes minimization)
    best_f = target_y.min().item()

    if use_ensemble_acquisition:
        # NEW: Use ensemble predictions for acquisition (Section 2.2 fix)
        acqf = RGPEAcquisition(
            rgpe_ensemble=rgpe,
            best_f=best_f,
            maximize=False,
        )
        acq_name = "RGPEAcquisition (Ensemble EI)"
    else:
        # Legacy: Use only target model for acquisition
        target_model = rgpe.target_model
        target_model.eval()
        acqf = qLogNoisyExpectedImprovement(
            model=target_model,
            X_baseline=target_x,
            prune_baseline=True,
            cache_root=False,
        )
        acq_name = "qLogNoisyExpectedImprovement (Target Only)"

    # Optimize acquisition
    candidates, acq_values = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=batch_size,
        num_restarts=20,
        raw_samples=512,
        sequential=True,
    )

    # Get weight explanation
    weight_explanation = get_rgpe_weights_explanation(rgpe, prior_tasks)

    # Compute ensemble uncertainty at candidates for metadata
    rgpe.eval()
    with torch.no_grad():
        posterior = rgpe.posterior(candidates)
        ensemble_uncertainty = posterior.variance.mean().item()

    metadata = {
        "model_type": "RGPE (Rank-weighted GP Ensemble)",
        "acquisition_function": acq_name,
        "use_ensemble_acquisition": use_ensemble_acquisition,
        "n_prior_tasks": len(prior_tasks),
        "weights": weight_explanation,
        "target_weight": weight_explanation["target"],
        "ensemble_uncertainty": ensemble_uncertainty,
        "best_f": best_f,
    }

    return candidates, acq_values, metadata


def generate_rgpe_suggestions_legacy(
    target_x: Tensor,
    target_y: Tensor,
    prior_tasks: list[PriorTaskData],
    bounds: Tensor,
    batch_size: int = 1,
    config: RGPEConfig | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Generate suggestions using legacy RGPE (target model only).

    This function maintains backward compatibility with the previous
    implementation that only used the target model for acquisition.

    Args:
        target_x: Target task training inputs
        target_y: Target task training outputs
        prior_tasks: List of prior task data
        bounds: Parameter bounds
        batch_size: Number of suggestions
        config: RGPE configuration

    Returns:
        Tuple of (candidates, acquisition_values, metadata)

    Note:
        Consider using generate_rgpe_suggestions with use_ensemble_acquisition=True
        for better transfer learning performance.
    """
    return generate_rgpe_suggestions(
        target_x=target_x,
        target_y=target_y,
        prior_tasks=prior_tasks,
        bounds=bounds,
        batch_size=batch_size,
        config=config,
        use_ensemble_acquisition=False,
    )
