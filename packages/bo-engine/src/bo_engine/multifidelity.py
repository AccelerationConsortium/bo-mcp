"""Multi-Fidelity Bayesian Optimization using qMFKG.

Supports cheap/expensive evaluation strategies where lower-fidelity
evaluations are cheaper but provide useful information about the
high-fidelity function.

v2.0: Initial implementation
v2.3: Added GPU auto-detection and acceleration
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.fit import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from bo_engine.constants import (
    MAX_RANDOM_SEED,
    MF_ACQF_BATCH_LIMIT,
    MF_ACQF_MAXITER,
    MF_DEFAULT_COST_WEIGHT,
    MF_DEFAULT_FIXED_COST,
)
from bo_engine.device import ensure_device, fork_rng_devices, to_device
from bo_engine.reproducibility import GLOBAL_RNG_LOCK, derive_seed


@dataclass(frozen=True)
class FidelitySpec:
    """Specification for a fidelity parameter.

    Fidelity parameters control the approximation level of evaluations.
    Lower fidelity = cheaper but less accurate.
    Higher fidelity = more expensive but more accurate.

    Attributes:
        fidelity_dim: Index of the fidelity dimension in input tensor
        target_fidelity: Target fidelity for final optimization (usually 1.0)
        fixed_cost: Fixed base cost for all evaluations
        cost_weight: Weight for fidelity in cost computation
        name: Optional name for the fidelity parameter
        bounds: Optional bounds tuple (min_fidelity, max_fidelity)
    """

    fidelity_dim: int  # Index of the fidelity dimension
    target_fidelity: float = 1.0  # Target fidelity for final optimization
    fixed_cost: float = MF_DEFAULT_FIXED_COST  # Fixed base cost
    cost_weight: float = MF_DEFAULT_COST_WEIGHT  # Cost scaling factor for fidelity
    name: str = "fidelity"  # Name of the fidelity parameter
    bounds: tuple[float, float] = (0.0, 1.0)  # (min_fidelity, max_fidelity)

    @property
    def target(self) -> float:
        """Alias for target_fidelity for backward compatibility."""
        return self.target_fidelity


@dataclass(frozen=True)
class MultiFidelityConfig:
    """Configuration for multi-fidelity optimization.

    Attributes:
        fidelity_spec: Fidelity-parameter specification.
        num_fantasies: Number of fantasy samples for qMFKG.
        num_restarts: Optimization restarts for the inner / outer problems.
        raw_samples: Raw samples for multi-start initialization.
        random_seed: Optional master seed. When set, the model fit and the
            MFKG candidate optimization run inside a forked torch RNG scope
            seeded via :func:`bo_engine.reproducibility.derive_seed`, so two
            independent calls on the same campaign state reproduce identical
            candidates without leaking the seed to concurrent callers.
    """

    fidelity_spec: FidelitySpec
    num_fantasies: int = 64  # Number of fantasies for qMFKG
    num_restarts: int = 10  # Optimization restarts
    raw_samples: int = 512  # Raw samples for initialization
    random_seed: int | None = None  # Master seed for reproducible candidates


def create_multifidelity_model(
    train_x: Tensor,
    train_y: Tensor,
    fidelity_dim: int,
    bounds: Tensor | None = None,
) -> SingleTaskMultiFidelityGP:
    """Create a multi-fidelity GP model.

    Inputs are normalized to the unit cube before the kernel sees them, matching
    every other model path in the package (transfer learning, calibration, CV).
    Without normalization the GP is fit on raw inputs, so parameters on
    heterogeneous scales (e.g. temperature 20-80 vs. ratio 0-1 vs. fidelity
    [0, 1]) get a single ill-conditioned lengthscale and the fit degrades.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
            where the last column (or fidelity_dim) is the fidelity parameter.
        train_y: Training outputs of shape (n_samples, 1)
        fidelity_dim: Index of the fidelity dimension in train_x
        bounds: Parameter bounds of shape (2, n_dims) used to build the
            ``Normalize`` input transform. When ``None`` the bounds are learned
            from ``train_x`` (min/max per column), which is adequate for data
            already on the unit cube but less robust than explicit bounds.

    Returns:
        SingleTaskMultiFidelityGP model (unfitted)
    """
    train_x, train_y = ensure_device(train_x, train_y)

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    n_dims = train_x.shape[-1]
    if bounds is not None:
        (bounds,) = ensure_device(bounds)
        input_transform = Normalize(d=n_dims, bounds=bounds)
    else:
        input_transform = Normalize(d=n_dims)

    return SingleTaskMultiFidelityGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=input_transform,
        outcome_transform=Standardize(m=1),
        data_fidelities=[fidelity_dim],
    )


def fit_multifidelity_model(
    model: SingleTaskMultiFidelityGP,
) -> SingleTaskMultiFidelityGP:
    """Fit a multi-fidelity GP model.

    Args:
        model: SingleTaskMultiFidelityGP to fit

    Returns:
        Fitted model
    """
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def create_and_fit_multifidelity_model(
    train_x: Tensor,
    train_y: Tensor,
    fidelity_dim: int,
    bounds: Tensor | None = None,
) -> SingleTaskMultiFidelityGP:
    """Create and fit a multi-fidelity GP model.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1)
        fidelity_dim: Index of the fidelity dimension
        bounds: Parameter bounds of shape (2, n_dims) for the ``Normalize``
            input transform; learned from the data when ``None``.

    Returns:
        Fitted SingleTaskMultiFidelityGP
    """
    model = create_multifidelity_model(train_x, train_y, fidelity_dim, bounds)
    return fit_multifidelity_model(model)


def create_cost_model(
    fidelity_dim: FidelitySpec | int,
    cost_weight: float = MF_DEFAULT_COST_WEIGHT,
    fixed_cost: float = MF_DEFAULT_FIXED_COST,
) -> AffineFidelityCostModel:
    """Create a cost model for multi-fidelity optimization.

    The cost model defines how evaluation cost scales with fidelity.
    Higher fidelity = higher cost.

    Args:
        fidelity_dim: Either a FidelitySpec object or the index of
            the fidelity dimension
        cost_weight: Weight for fidelity in cost computation (only used if
            fidelity_dim is an int)
        fixed_cost: Fixed base cost for all evaluations (only used if
            fidelity_dim is an int)

    Returns:
        AffineFidelityCostModel
    """
    if isinstance(fidelity_dim, FidelitySpec):
        fidelity_spec = fidelity_dim
        fidelity_dim = fidelity_spec.fidelity_dim
        cost_weight = fidelity_spec.cost_weight
        fixed_cost = fidelity_spec.fixed_cost

    return AffineFidelityCostModel(
        fidelity_weights={fidelity_dim: cost_weight},
        fixed_cost=fixed_cost,
    )


def create_mfkg_acquisition(
    model: SingleTaskMultiFidelityGP,
    bounds: Tensor,
    fidelity_dim: int,
    target_fidelity: float,
    cost_model: AffineFidelityCostModel | None = None,
    num_fantasies: int = 64,
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> qMultiFidelityKnowledgeGradient:
    """Create qMFKG acquisition function.

    qMFKG (multi-fidelity Knowledge Gradient) optimizes the ratio
    of information gain to cost. Both qMFKG and the ``PosteriorMean``
    current-value computation maximize, so the model must be fit on
    maximization-form targets (negate minimize objectives before the
    fit — :func:`generate_multifidelity_suggestions` does this).

    Args:
        model: Fitted multi-fidelity GP model fit on maximization-form
            targets
        bounds: Parameter bounds of shape (2, n_dims)
        fidelity_dim: Index of the fidelity dimension
        target_fidelity: Target fidelity value
        cost_model: Optional cost model for cost-aware utility
        num_fantasies: Number of fantasy samples for KG
        num_restarts: Optimization restarts for current value
        raw_samples: Raw samples for initialization

    Returns:
        qMultiFidelityKnowledgeGradient acquisition function
    """
    bounds = to_device(bounds)
    n_dims = bounds.shape[-1]
    target_fidelities = {fidelity_dim: target_fidelity}

    # Create projection function for target fidelity
    def project(x: Tensor) -> Tensor:
        return project_to_target_fidelity(X=x, target_fidelities=target_fidelities, d=n_dims)

    # Compute current best value at target fidelity
    # Use FixedFeatureAcquisition to optimize over non-fidelity dimensions
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=n_dims,
        columns=[fidelity_dim],
        values=[target_fidelity],
    )

    # Bounds for non-fidelity dimensions
    non_fidelity_bounds = torch.cat(
        [bounds[:, :fidelity_dim], bounds[:, fidelity_dim + 1 :]],
        dim=-1,
    )

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=non_fidelity_bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    # Create cost-aware utility if cost model provided
    cost_aware_utility = None
    if cost_model is not None:
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=num_fantasies,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


def optimize_mfkg(
    acqf: qMultiFidelityKnowledgeGradient,
    bounds: Tensor,
    batch_size: int = 1,
    num_restarts: int = 20,
    raw_samples: int = 512,
) -> tuple[Tensor, Tensor]:
    """Optimize qMFKG acquisition function.

    Args:
        acqf: qMFKG acquisition function
        bounds: Parameter bounds of shape (2, n_dims)
        batch_size: Number of candidates to generate
        num_restarts: Number of optimization restarts
        raw_samples: Number of raw samples for initialization

    Returns:
        Tuple of (candidates, acquisition_values)
    """
    bounds = to_device(bounds)

    candidates, acq_values = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={
            "batch_limit": MF_ACQF_BATCH_LIMIT,
            "maxiter": MF_ACQF_MAXITER,
        },
    )

    return candidates, acq_values


def generate_multifidelity_suggestions(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    fidelity_config: MultiFidelityConfig,
    batch_size: int = 1,
    *,
    minimize: bool = True,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Generate suggestions using multi-fidelity BO.

    Main entry point for multi-fidelity optimization.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1), on the raw
            user scale (no pre-negation)
        bounds: Parameter bounds of shape (2, n_dims)
        fidelity_config: Multi-fidelity configuration
        batch_size: Number of suggestions to generate
        minimize: Direction of the raw objective.
            ``qMultiFidelityKnowledgeGradient`` and the ``PosteriorMean``
            current-value computation both maximize and expose no
            direction flag, so minimize objectives are negated into the
            engine's maximization form at the data boundary (see
            :mod:`bo_engine.types`); the KG machinery stays
            maximization-internal.

    Returns:
        Tuple of (candidates, acquisition_values, metadata)
        - candidates: Tensor of shape (batch_size, n_dims)
        - acquisition_values: Tensor of shape (batch_size,)
        - metadata: Dictionary with model and acquisition info
    """
    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    fidelity_spec = fidelity_config.fidelity_spec
    fidelity_dim = fidelity_spec.fidelity_dim

    # Negate minimize objectives into maximization form at the data
    # boundary -- BoTorch's KG stack has no direction flag.
    train_y_bo = -train_y if minimize else train_y

    # Resolve a per-call torch seed for the stochastic fit + MFKG
    # optimization. A configured master seed is derived deterministically
    # (role-tagged so it can't collide with another phase); otherwise fresh
    # stdlib entropy is drawn. The fallback is essential: fork_rng restores
    # the global RNG on exit, so without installing a per-call seed two
    # consecutive *unseeded* calls would replay the identical candidates
    # (mirrors thompson_sampling._resolve_call_seed).
    if fidelity_config.random_seed is not None:
        random_seed = derive_seed(fidelity_config.random_seed, "multifidelity:mfkg")
    else:
        # Deliberately non-reproducible — no master seed was supplied.
        random_seed = random.randint(0, MAX_RANDOM_SEED)  # noqa: S311

    # fork_rng isolates the global torch RNG mutation so concurrent callers
    # cannot race on the seed (state is restored on exit); fork_rng_devices()
    # adds the active CUDA device(s) so manual_seed's CUDA-generator mutation
    # is restored too. GLOBAL_RNG_LOCK serializes the snapshot/restore against
    # every other consumer of the process-global RNG.
    with GLOBAL_RNG_LOCK, torch.random.fork_rng(devices=fork_rng_devices()):
        torch.manual_seed(random_seed)

        # Create and fit model on normalized inputs (bounds threaded through).
        model = create_and_fit_multifidelity_model(train_x, train_y_bo, fidelity_dim, bounds)

        # Create cost model
        cost_model = create_cost_model(fidelity_spec)

        # Create acquisition function
        acqf = create_mfkg_acquisition(
            model=model,
            bounds=bounds,
            fidelity_dim=fidelity_dim,
            target_fidelity=fidelity_spec.target_fidelity,
            cost_model=cost_model,
            num_fantasies=fidelity_config.num_fantasies,
            num_restarts=fidelity_config.num_restarts,
            raw_samples=fidelity_config.raw_samples,
        )

        # Optimize
        candidates, acq_values = optimize_mfkg(
            acqf=acqf,
            bounds=bounds,
            batch_size=batch_size,
            num_restarts=fidelity_config.num_restarts,
            raw_samples=fidelity_config.raw_samples,
        )

    metadata = {
        "model_type": "SingleTaskMultiFidelityGP",
        "acquisition_function": "qMultiFidelityKnowledgeGradient",
        "minimize": minimize,
        "fidelity_dim": fidelity_dim,
        "target_fidelity": fidelity_spec.target_fidelity,
        "num_fantasies": fidelity_config.num_fantasies,
        "random_seed": random_seed,
    }

    return candidates, acq_values, metadata
