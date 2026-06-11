"""Transfer Learning with RGPE (Rank-weighted GP Ensemble).

Leverages knowledge from prior optimization campaigns to improve
optimization on a new but related task.

Ensemble weights follow the paper's ranking loss: each model is scored by
how often its posterior samples misrank pairs of target observations, the
target model is scored out-of-sample via exact leave-one-out moments, and
a model's weight is the fraction of samples in which it achieves the
lowest loss (ties split equally). Because the loss counts pair orderings
only, weights are invariant to affine rescaling of any task's outputs —
a prior on a different y-scale but with the same landscape ranks highly,
which is exactly the property MSE-style weighting lacks.

v2.0: Initial implementation based on Feurer, Letham, Bakshy (ICML 2018)
v2.3: Added GPU auto-detection and acceleration
v2.6: Added RGPEAcquisition class that properly uses ensemble predictions (Section 2.2)

References:
    - Feurer, Letham, Bakshy "Scalable Meta-Learning for Bayesian
      Optimization" ICML AutoML Workshop 2018; extended as "Practical
      Transfer Learning for Bayesian Optimization"
      (https://arxiv.org/abs/1802.02219)
    - BoTorch RGPE tutorial:
      https://botorch.org/docs/tutorials/meta_learning_with_rgpe/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import _log_ei_helper
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
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

from bo_engine.constants import (
    NUMERICAL_EPSILON,
    RGPE_DILUTION_BASE_QUANTILE,
    RGPE_DILUTION_TARGET_QUANTILE,
    RGPE_MIN_TARGET_OBSERVATIONS,
    RGPE_NUM_SAMPLES,
    RGPE_PENDING_PENALTY_LENGTHSCALE,
)
from bo_engine.cross_validation import compute_exact_loo_moments
from bo_engine.device import ensure_device
from bo_engine.types import AcquisitionOptimizationConfig


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

    num_samples: int = RGPE_NUM_SAMPLES  # Posterior samples for ranking-loss estimation
    use_input_warping: bool = False  # Whether to use input warping


def _ranking_loss(pred_left: Tensor, pred_right: Tensor, observed: Tensor) -> Tensor:
    """Count ordered pairs whose predicted ordering disagrees with the observed one.

    Implements the pair-counting statistic of Feurer et al. (Eq. 1):
    ``Σ_j Σ_k 1[(left_j < right_k) ⊕ (y_j < y_k)]`` per posterior sample,
    with strict ``<`` on both sides. The diagonal contributes nothing for
    the base-model variant (``left is right``); the count depends only on
    orderings, never on magnitudes, so it is invariant to any strictly
    increasing transform of either side.

    Args:
        pred_left: ``(S, n)`` predictions indexed by the first pair member.
        pred_right: ``(S, n)`` predictions indexed by the second pair
            member — the same tensor as ``pred_left`` for base models
            (model-vs-model orderings, Eq. 1) or the broadcast observed
            targets for the LOO target variant (prediction-vs-observation
            orderings, Eq. 4 of the extended paper).
        observed: ``(n,)`` observed target values.

    Returns:
        ``(S,)`` integer tensor of discordant ordered-pair counts.
    """
    pred_less = pred_left.unsqueeze(-1) < pred_right.unsqueeze(-2)  # (S, n, n)
    observed_less = (observed.unsqueeze(-1) < observed.unsqueeze(-2)).unsqueeze(0)
    return (pred_less ^ observed_less).sum(dim=(-2, -1))


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
        num_samples: int = RGPE_NUM_SAMPLES,
    ) -> None:
        """Initialize RGPE.

        Args:
            base_models: List of fitted GP models from prior tasks
            target_model: GP model for the current target task
            weights: Optional pre-computed weights (will be computed if None)
            num_samples: Posterior samples drawn per model when estimating
                the ranking-loss distribution in :meth:`compute_weights`
        """
        super().__init__()
        self.base_models = torch.nn.ModuleList(base_models)
        self.target_model = target_model
        self._weights = weights
        self.num_samples = num_samples

    @property
    def num_models(self) -> int:
        """Total number of models in the ensemble."""
        return len(self.base_models) + 1

    @property
    def weights(self) -> Tensor:
        """Get ensemble weights."""
        if self._weights is None:
            msg = "Weights not computed. Call compute_weights() first."
            raise ValueError(msg)
        return self._weights

    def compute_weights(
        self,
        target_x: Tensor,
        target_y: Tensor,
    ) -> Tensor:
        """Compute ranking-loss weights per Feurer et al. (2018).

        Each model's quality is the distribution of its ranking loss —
        the number of target-observation pairs whose ordering the model's
        posterior samples get wrong:

        1. For every base model, draw ``num_samples`` joint posterior
           samples at ``target_x`` and count discordant pairs against
           ``target_y`` (paper Eq. 1).
        2. For the target model, draw the samples from its exact
           leave-one-out posterior (GPML Eqs. 5.10-5.12 via
           :func:`bo_engine.cross_validation.compute_exact_loo_moments` —
           hyperparameters fixed, no refit, exactly the paper's LOO
           construction) so it is scored out-of-sample like the priors.
        3. Weight dilution prevention (paper v1 percentile rule): a base
           model whose median loss is >= the target model's 95th
           percentile loss is discarded before weighting.
        4. ``w_i`` is the fraction of samples in which model ``i``
           achieves the minimal loss, with ties split equally among the
           tied models (extended paper, Eq. 5).

        Because the loss counts pair orderings only, the weights are
        invariant to affine rescaling of any task's outputs — a prior
        task observed on a different y-scale but with the same landscape
        still ranks highly.

        With fewer than ``RGPE_MIN_TARGET_OBSERVATIONS`` target points the
        LOO models are too small to rank anything and all models receive
        uniform weight, as prescribed by the paper.

        Args:
            target_x: Target task inputs — must be the data the target
                model was trained on (``create_rgpe_model`` guarantees
                this), since the LOO downdate scores the model's own
                training points.
            target_y: Target task outputs in the same row order.

        Returns:
            Tensor of weights with shape (num_models,), summing to 1.
        """
        n_target = target_x.shape[0]
        n_models = self.num_models

        if n_target < RGPE_MIN_TARGET_OBSERVATIONS:
            weights = torch.full((n_models,), 1.0 / n_models, dtype=torch.double)
            self._weights = weights
            return weights

        observed = target_y.reshape(-1).to(torch.double)

        # ModuleList iteration is typed as bare Module; the list is
        # populated exclusively with fitted SingleTaskGPs in __init__.
        losses = [
            self._base_model_ranking_losses(cast("SingleTaskGP", model), target_x, observed)
            for model in self.base_models
        ]
        losses.append(self._target_model_ranking_losses())

        loss_tensor = torch.stack(losses).to(torch.double)  # (n_models, S)
        loss_tensor = self._apply_weight_dilution(loss_tensor)

        # w_i = mean_s [ 1(i in argmin losses_s) / |argmin losses_s| ]
        min_per_sample = loss_tensor.min(dim=0).values
        is_best = loss_tensor == min_per_sample.unsqueeze(0)
        weights = (is_best.to(torch.double) / is_best.sum(dim=0, keepdim=True)).mean(dim=1)

        self._weights = weights
        return weights

    def _base_model_ranking_losses(
        self,
        model: SingleTaskGP,
        target_x: Tensor,
        observed: Tensor,
    ) -> Tensor:
        """Ranking-loss samples of one base model on the target data (Eq. 1)."""
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(target_x)
            samples = posterior.rsample(torch.Size([self.num_samples]))
        pred = samples.reshape(self.num_samples, -1).to(torch.double)
        return _ranking_loss(pred, pred, observed)

    def _target_model_ranking_losses(self) -> Tensor:
        """Ranking-loss samples of the target model via exact LOO (Eq. 4).

        The LOO moments live in the model's transformed target space; the
        observed side therefore uses ``train_targets`` from the same
        space. The transform is affine and strictly increasing, so the
        pair orderings — the only thing the loss consumes — are identical
        to raw-space orderings.
        """
        loo_mean, loo_var = compute_exact_loo_moments(self.target_model)
        observed = self.target_model.train_targets.reshape(-1).to(torch.double)
        noise = torch.randn(
            self.num_samples, loo_mean.shape[0], dtype=loo_mean.dtype, device=loo_mean.device
        )
        loo_samples = (loo_mean.unsqueeze(0) + loo_var.clamp(min=0).sqrt().unsqueeze(0) * noise).to(
            torch.double
        )
        # Out-of-sample prediction at point k is compared against the
        # observed value at every other point l (extended paper, Eq. 4).
        return _ranking_loss(loo_samples, observed.expand_as(loo_samples), observed)

    def _apply_weight_dilution(self, loss_tensor: Tensor) -> Tensor:
        """Discard base models per the paper's v1 percentile dilution rule.

        A base model is removed from the argmin competition (weight 0)
        when the ``RGPE_DILUTION_BASE_QUANTILE`` of its loss samples is
        >= the ``RGPE_DILUTION_TARGET_QUANTILE`` of the target model's —
        without this, many weak priors would each win a few samples and
        collectively dilute the target model's weight.
        """
        target_threshold = torch.quantile(loss_tensor[-1], RGPE_DILUTION_TARGET_QUANTILE)
        diluted = loss_tensor.clone()
        for i in range(loss_tensor.shape[0] - 1):
            base_quantile = torch.quantile(loss_tensor[i], RGPE_DILUTION_BASE_QUANTILE)
            if base_quantile >= target_threshold:
                diluted[i] = torch.inf
        return diluted

    def posterior(self, x: Tensor) -> GPyTorchPosterior:
        """Compute weighted posterior prediction.

        Args:
            x: Input points of shape (..., n_dims)

        Returns:
            GPyTorchPosterior with weighted predictions
        """
        all_models = [*list(self.base_models), self.target_model]
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

    # Validate parameter space compatibility between target and prior tasks
    target_dim = target_x.shape[-1]
    for prior_task in prior_tasks:
        prior_dim = prior_task.train_x.shape[-1]
        if prior_dim != target_dim:
            msg = (
                f"Prior task '{prior_task.name}' has {prior_dim} dimensions "
                f"but target task has {target_dim}. All tasks must share the "
                f"same parameter space dimensionality for transfer learning."
            )
            raise ValueError(msg)

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
        num_samples=config.num_samples,
    )

    # Compute weights
    rgpe.compute_weights(
        target_x=target_x,
        target_y=target_y,
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


def _pending_penalty_factor(
    candidates: Tensor,
    x_pending: Tensor | None,
    bounds: Tensor | None,
    lengthscale: float = RGPE_PENDING_PENALTY_LENGTHSCALE,
) -> Tensor | None:
    """Multiplicative local-penalization factor in [0, 1) per candidate point.

    Gaussian-kernel penalizer centered on the pending points (Gonzalez et
    al. 2016, "Batch Bayesian Optimization via Local Penalization"): a
    candidate coincident with a pending point gets a factor near 0, one
    far from every pending point gets a factor near 1. ``optimize_acqf``
    with ``sequential=True`` feeds already-selected batch members back
    through ``set_X_pending``, so an acquisition that folds this factor
    in stops re-proposing the same maximizer for every batch slot.

    Distances are measured in the normalized [0, 1] input cube when
    ``bounds`` is provided so the lengthscale is unit-free.

    Args:
        candidates: Candidate points of shape (..., q, d).
        x_pending: Pending points of shape (p, d), or None.
        bounds: Optional (2, d) bounds for input normalization.
        lengthscale: Penalization kernel lengthscale in normalized space.

    Returns:
        Factor tensor of shape (..., q), or None when nothing is pending.
    """
    if x_pending is None or x_pending.numel() == 0:
        return None

    pending = x_pending
    if bounds is not None:
        ranges = (bounds[1] - bounds[0]).clamp(min=NUMERICAL_EPSILON)
        candidates = (candidates - bounds[0]) / ranges
        pending = (x_pending - bounds[0]) / ranges

    distances = torch.cdist(candidates, pending.to(candidates))  # (..., q, p)
    min_distances = distances.min(dim=-1).values  # (..., q)
    return 1.0 - torch.exp(-(min_distances**2) / (2 * lengthscale**2))


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

    Pending points set via ``X_pending`` (e.g. by ``optimize_acqf``'s
    sequential batch loop) locally penalize the EI surface so batch
    members spread out instead of collapsing onto one maximizer.

    Reference:
        Feurer et al. "Scalable Meta-Learning for Bayesian Optimization"
        ICML AutoML Workshop 2018

    Args:
        rgpe_ensemble: Fitted RGPE model with computed weights
        best_f: Best observed value on the target task
        maximize: If True, maximize; else minimize (default)
        bounds: Optional (2, d) parameter bounds used to normalize
            pending-point distances for the local penalizer

    Example:
        >>> rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        >>> best_f = target_y.min().item()  # For minimization
        >>> acqf = RGPEAcquisition(rgpe, best_f, bounds=bounds)
        >>> candidates, acq_values = optimize_acqf(acqf, bounds, q=1)
    """

    def __init__(
        self,
        rgpe_ensemble: RGPE,
        best_f: float,
        maximize: bool = False,
        bounds: Tensor | None = None,
    ) -> None:
        """Initialize RGPE acquisition function.

        Args:
            rgpe_ensemble: Fitted RGPE model
            best_f: Best observed value (for EI computation)
            maximize: If True, maximize the objective
            bounds: Optional (2, d) bounds for pending-point normalization
        """
        # Initialize with target model as the base model for BoTorch compatibility
        super().__init__(model=rgpe_ensemble.target_model)
        self.rgpe_ensemble = rgpe_ensemble
        self.best_f = best_f
        self.maximize = maximize
        self.bounds = bounds
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
        z = (mean - self.best_f) / std if self.maximize else (self.best_f - mean) / std

        # Standard normal distribution for EI computation
        normal = Normal(torch.zeros_like(z), torch.ones_like(z))
        pdf = torch.exp(normal.log_prob(z))
        cdf = normal.cdf(z)

        # Closed-form Expected Improvement
        ei = std * (z * cdf + pdf)

        # Condition on pending batch members: EI is non-negative, so the
        # multiplicative local penalizer is sign-safe here (contrast the
        # log-space variant below, which adds the log factor instead).
        penalty = _pending_penalty_factor(X, self._X_pending, self.bounds)
        if penalty is not None:
            ei = ei * penalty

        # Average over q dimension for joint acquisition
        ei = ei.mean(dim=-1) if ei.dim() > 1 and ei.shape[-1] > 1 else ei.squeeze(-1)

        # Ensure non-negative
        return ei.clamp(min=0.0)


class RGPELogEI(AcquisitionFunction):
    """Log Expected Improvement using RGPE ensemble predictions.

    Uses log-space computation for numerical stability, similar to
    qLogNoisyExpectedImprovement but with RGPE ensemble predictions.
    Pending points set via ``X_pending`` penalize the surface additively
    in log space (``log EI + log(1 - k(x, pending))``), which keeps the
    penalty sign-safe for the negative values log-EI takes.

    Args:
        rgpe_ensemble: Fitted RGPE model with computed weights
        best_f: Best observed value on the target task
        maximize: If True, maximize; else minimize
        bounds: Optional (2, d) parameter bounds used to normalize
            pending-point distances for the local penalizer
    """

    def __init__(
        self,
        rgpe_ensemble: RGPE,
        best_f: float,
        maximize: bool = False,
        bounds: Tensor | None = None,
    ) -> None:
        """Initialize RGPE Log-EI acquisition function."""
        super().__init__(model=rgpe_ensemble.target_model)
        self.rgpe_ensemble = rgpe_ensemble
        self.best_f = best_f
        self.maximize = maximize
        self.bounds = bounds
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

        z = (mean - self.best_f) / std if self.maximize else (self.best_f - mean) / std

        # log(EI) = log(std) + log(z * Phi(z) + phi(z)). The bracket
        # ``h(z) = z*Phi(z) + phi(z)`` is strictly positive, but the naive
        # ``log(z * exp(log_cdf) + ...)`` underflows the cdf clamp into a
        # negative argument for large negative z (reachable because ``std``
        # clamps to 1e-6 near well-sampled points), yielding ``log(negative)``
        # = NaN. BoTorch's ``_log_ei_helper`` evaluates ``log h(z)`` in a
        # branch-stable, differentiable way for |z| up to 1e100 (Ament et al.
        # 2023, "Unexpected Improvements to Expected Improvement").
        log_std = torch.log(std)
        log_ei = _log_ei_helper(z) + log_std

        # Condition on pending batch members. log-EI can be negative, so
        # the multiplicative penalizer is applied additively in log space
        # (log of a factor in [0, 1) is <= 0 — always a penalty).
        penalty = _pending_penalty_factor(X, self._X_pending, self.bounds)
        if penalty is not None:
            log_ei = log_ei + torch.log(penalty.clamp(min=NUMERICAL_EPSILON))

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
    acquisition_optimization: AcquisitionOptimizationConfig | None = None,
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
        acquisition_optimization: Optional configuration controlling
            acquisition-function optimization (number of restarts, raw
            samples, etc.). Defaults are used when ``None``.

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
        # NEW: Use ensemble predictions for acquisition (Section 2.2 fix).
        # bounds enable the pending-point penalizer so the sequential batch
        # loop below produces spread-out candidates instead of duplicates.
        acqf = RGPEAcquisition(
            rgpe_ensemble=rgpe,
            best_f=best_f,
            maximize=False,
            bounds=bounds,
        )
        acq_name = "RGPEAcquisition (Ensemble EI)"
    else:
        # Legacy: Use only target model for acquisition. The target model is
        # fit on raw minimization-form targets (this module's documented
        # contract), but BoTorch's qLogNEI always maximizes — wire a
        # negating objective so the acquisition minimizes the raw objective
        # instead of chasing its maximum.
        target_model = rgpe.target_model
        target_model.eval()
        acqf = qLogNoisyExpectedImprovement(
            model=target_model,
            X_baseline=target_x,
            prune_baseline=True,
            cache_root=False,
            # ``GenericMCObjective`` calls ``objective(samples, X=X)`` — the
            # parameter must be named ``X`` even though it is unused.
            objective=GenericMCObjective(lambda samples, X=None: -samples[..., 0]),  # noqa: N803, ARG005
        )
        acq_name = "qLogNoisyExpectedImprovement (Target Only)"

    # Optimize acquisition
    acq_config = acquisition_optimization or AcquisitionOptimizationConfig()
    num_restarts, raw_samples = acq_config.resolve(int(bounds.shape[-1]))
    candidates, acq_values = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
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
