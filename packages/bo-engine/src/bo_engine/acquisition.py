"""Acquisition function creation and optimization.

Supports both single-objective (noisy EI, EI) and multi-objective
(hypervolume improvement, scalarized multi-objective) acquisition
functions.

All public factories in this module require an explicit ``minimize`` (or
``minimize_mask``) argument and expect ``train_y`` / ``best_f`` /
``ref_point`` to be in *minimization form* (lower = better).  See the
canonical sign-convention documentation in :mod:`bo_engine.types` for
details and for caller responsibilities.

v1.0.1: Added single-objective support via qLogNoisyExpectedImprovement
v1.1: Added qLogNParEGO alternative for multi-objective
v1.3: Added cost-aware (EIpu) and outcome constraint support
v2.3: Added GPU auto-detection and acceleration
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.optim.optimize import optimize_acqf_discrete, optimize_acqf_mixed
from torch import Tensor

from bo_engine.constants import MIXED_CATEGORICAL_COMBO_THRESHOLD
from bo_engine.device import ensure_device, to_device
from bo_engine.transforms import (
    SearchSpaceType,
    build_fixed_features_list,
    classify_search_space,
    count_categorical_combinations,
    enumerate_discrete_choices,
)
from bo_engine.types import AcquisitionConfig, AcquisitionMethod, OptimizationSpec

logger = logging.getLogger(__name__)


def _assert_minimization_form(minimize: bool) -> None:
    """Guard against silently passing maximization-form data into a factory.

    The canonical internal convention is minimization form (lower is better);
    callers must negate any maximization objectives before invoking a
    factory.  See :mod:`bo_engine.types` for the convention.
    """
    if not minimize:
        raise ValueError(
            "bo_engine acquisition factories operate in minimization form "
            "(lower = better).  Negate maximization objectives at the call "
            "site and pass ``minimize=True``.  See the sign-convention note "
            "in bo_engine.types for details."
        )


def _assert_minimization_mask(minimize_mask: Tensor) -> None:
    """Multi-objective counterpart of :func:`_assert_minimization_form`."""
    if not bool(minimize_mask.all()):
        raise ValueError(
            "bo_engine multi-objective factories operate in minimization "
            "form (every objective treated as lower = better).  Negate any "
            "maximization columns at the call site before building the "
            "acquisition function.  See bo_engine.types for details."
        )


def create_single_objective_acquisition(
    model: SingleTaskGP,
    train_x: Tensor,
    train_y: Tensor,
    *,
    minimize: bool,
    best_f: float | None = None,
    use_noisy: bool = True,
    constraints: list | None = None,
) -> AcquisitionFunction:
    """Create acquisition function for single-objective optimization.

    ``train_y`` and ``best_f`` must be supplied in the canonical
    minimization form (lower = better).  See :mod:`bo_engine.types` for the
    sign convention and caller responsibilities.

    Args:
        model: Fitted SingleTaskGP model (trained on minimization-form y)
        train_x: Training inputs for baseline sampling
        train_y: Training outputs in minimization form
        minimize: Direction of the user-facing objective.  Required keyword
            so the caller cannot silently pass the wrong sign convention.
            Currently only ``True`` is supported at the construction
            boundary because the internal engine convention is
            minimization form — pass ``minimize=True`` after negating any
            maximization objectives at the call site.
        best_f: Best observed minimization-form value.  If ``None`` and
            ``use_noisy=False``, computed as ``train_y.min()``.
        use_noisy: If True, use noisy EI (handles noise), else EI
        constraints: Optional list of constraint callables

    Returns:
        Noisy EI or EI acquisition function
    """
    _assert_minimization_form(minimize)
    train_x, train_y = ensure_device(train_x, train_y)

    if use_noisy:
        # Noisy EI handles noisy observations - recommended default
        acqf_kwargs: dict[str, Any] = {
            "model": model,
            "X_baseline": train_x,
            "prune_baseline": True,
            "cache_root": False,
        }
        if constraints is not None and len(constraints) > 0:
            acqf_kwargs["constraints"] = constraints
        return qLogNoisyExpectedImprovement(**acqf_kwargs)
    else:
        # EI for noiseless observations - requires best_f
        if best_f is None:
            # train_y is in minimization form (lower = better); min() is best.
            best_f = train_y.min().item()
        return qLogExpectedImprovement(model=model, best_f=best_f)


def create_multi_objective_acquisition(
    model: ModelListGP,
    ref_point: Tensor,
    train_x: Tensor,
    train_y: Tensor,
    *,
    minimize_mask: Tensor,
    method: AcquisitionMethod = AcquisitionMethod.HYPERVOLUME_IMPROVEMENT,
    constraints: list | None = None,
) -> AcquisitionFunction:
    """Create acquisition function for multi-objective optimization.

    ``train_y`` and ``ref_point`` must be supplied in the canonical
    minimization form — every objective treated as lower = better — with
    maximization columns pre-negated at the call site.  See
    :mod:`bo_engine.types` for the convention.

    Args:
        model: Fitted ModelListGP (trained on minimization-form y)
        ref_point: Reference point in minimization form
        train_x: Training inputs for sampling baseline
        train_y: Training outputs in minimization form
        minimize_mask: Boolean tensor of shape ``(n_objectives,)`` recording
            the user-facing direction of each objective.  Required keyword
            so the call site cannot silently pass mismatched signs; must be
            all-True because the engine operates in minimization form.
        method: Acquisition method (HYPERVOLUME_IMPROVEMENT or SCALARIZED_MULTI_OBJ)
        constraints: Optional list of constraint callables

    Returns:
        Multi-objective acquisition function
    """
    _assert_minimization_mask(minimize_mask)
    train_x, train_y, ref_point = ensure_device(train_x, train_y, ref_point)

    if method == AcquisitionMethod.SCALARIZED_MULTI_OBJ:
        # qLogNParEGO: Random scalarization weights for Pareto exploration
        # When weights=None, qLogNParEGO uses random weights internally
        acqf_kwargs: dict = {
            "model": model,
            "X_baseline": train_x,
            "scalarization_weights": None,  # Use random weights
            "prune_baseline": True,
            "cache_root": False,
        }
        if constraints is not None and len(constraints) > 0:
            acqf_kwargs["constraints"] = constraints

        return qLogNParEGO(**acqf_kwargs)

    else:
        # Default: hypervolume improvement (qLogNEHVI)
        acqf_kwargs = {
            "model": model,
            "ref_point": ref_point.tolist(),
            "X_baseline": train_x,
            "prune_baseline": True,
            "cache_root": False,
        }
        if constraints is not None and len(constraints) > 0:
            acqf_kwargs["constraints"] = constraints

        return qLogNoisyExpectedHypervolumeImprovement(**acqf_kwargs)  # ty: ignore[invalid-argument-type]


def create_acquisition_from_config(config: AcquisitionConfig) -> AcquisitionFunction:
    """Create acquisition function from configuration object.

    This is the preferred way to create acquisition functions as it reduces
    parameter count and improves code clarity.

    Args:
        config: Acquisition configuration containing all parameters.  The
            caller must populate ``config.minimize`` /
            ``config.minimize_mask`` consistent with the canonical
            minimization-form convention (see :mod:`bo_engine.types`).

    Returns:
        Acquisition function appropriate for the problem
    """
    return create_acquisition(
        model=config.model,
        ref_point=config.ref_point,
        train_x=config.train_x,
        train_y=config.train_y,
        n_objectives=config.n_objectives,
        minimize=config.minimize,
        minimize_mask=config.minimize_mask,
        method=config.method,
        constraints=config.constraints,
        outcome_constraint_models=config.outcome_constraint_models,
        cost_model=config.cost_model,
    )


def create_acquisition(
    model: ModelListGP | SingleTaskGP,
    ref_point: Tensor | None,
    train_x: Tensor,
    train_y: Tensor,
    n_objectives: int = 2,
    *,
    minimize: bool | None = None,
    minimize_mask: Tensor | None = None,
    method: AcquisitionMethod = AcquisitionMethod.AUTO,
    constraints: list | None = None,
    outcome_constraint_models: list[tuple[SingleTaskGP, float]] | None = None,
    cost_model: SingleTaskGP | None = None,
) -> AcquisitionFunction:
    """Create appropriate acquisition function based on problem type.

    Automatically selects between single-objective and multi-objective
    acquisition functions based on n_objectives.

    ``train_y`` (and ``ref_point`` for the multi-objective path) must be in
    the canonical minimization form — see :mod:`bo_engine.types`.  Callers
    **must** supply ``minimize`` for single-objective problems and
    ``minimize_mask`` for multi-objective problems so the direction is
    explicit at the construction boundary.

    Note: Consider using create_acquisition_from_config() with an
    AcquisitionConfig object for cleaner code with fewer parameters.

    Args:
        model: Fitted GP model (SingleTaskGP or ModelListGP)
        ref_point: Reference point for hypervolume (multi-objective only)
        train_x: Training inputs for sampling baseline
        train_y: Training outputs in minimization form
        n_objectives: Number of objectives (1 = single, 2+ = multi)
        minimize: User-facing objective direction for single-objective
            problems.  Required when ``n_objectives == 1``.
        minimize_mask: Boolean tensor recording user-facing direction per
            objective.  Required when ``n_objectives >= 2``.
        method: Acquisition method (AUTO selects automatically)
        constraints: Optional list of constraint callables
        outcome_constraint_models: Optional list of (model, threshold) tuples
            for outcome constraints
        cost_model: Optional cost model for EIpu acquisition

    Returns:
        Acquisition function appropriate for the problem
    """
    # Determine method if AUTO
    if method == AcquisitionMethod.AUTO:
        if n_objectives == 1:
            method = AcquisitionMethod.NOISY_EI
        else:
            method = AcquisitionMethod.HYPERVOLUME_IMPROVEMENT

    if n_objectives == 1:
        if minimize is None:
            raise ValueError(
                "create_acquisition requires ``minimize`` for "
                "single-objective problems; see bo_engine.types for the "
                "sign convention."
            )
        return _create_single_objective_dispatch(
            model=model,
            train_x=train_x,
            train_y=train_y,
            minimize=minimize,
            method=method,
            constraints=constraints,
            outcome_constraint_models=outcome_constraint_models,
            cost_model=cost_model,
        )

    if minimize_mask is None:
        raise ValueError(
            "create_acquisition requires ``minimize_mask`` for "
            "multi-objective problems; see bo_engine.types for the sign "
            "convention."
        )
    return _create_multi_objective_dispatch(
        model=model,
        ref_point=ref_point,
        train_x=train_x,
        train_y=train_y,
        minimize_mask=minimize_mask,
        method=method,
        constraints=constraints,
    )


def _create_single_objective_dispatch(
    model: ModelListGP | SingleTaskGP,
    train_x: Tensor,
    train_y: Tensor,
    *,
    minimize: bool,
    method: AcquisitionMethod,
    constraints: list | None,
    outcome_constraint_models: list[tuple[SingleTaskGP, float]] | None,
    cost_model: SingleTaskGP | None,
) -> AcquisitionFunction:
    """Dispatch single-objective acquisition function creation.

    Handles model extraction from ModelListGP, cost-aware EIpu, and
    standard EI/noisy-EI with optional outcome constraints.

    Args:
        model: Fitted GP model (SingleTaskGP or single-model ModelListGP)
        train_x: Training inputs
        train_y: Training outputs in minimization form
        minimize: User-facing objective direction (see :mod:`bo_engine.types`)
        method: Acquisition method
        constraints: Optional constraint callables
        outcome_constraint_models: Optional (model, threshold) pairs
        cost_model: Optional cost model for EIpu

    Returns:
        Single-objective acquisition function
    """
    if not isinstance(model, SingleTaskGP):
        if isinstance(model, ModelListGP) and len(model.models) == 1:
            model = cast(SingleTaskGP, model.models[0])
        else:
            raise ValueError("Single-objective requires SingleTaskGP model")

    if method == AcquisitionMethod.COST_WEIGHTED_EI and cost_model is not None:
        return create_cost_aware_acquisition(
            model=model,  # type: ignore[arg-type]
            train_y=train_y,
            minimize=minimize,
            cost_model=cost_model,
            outcome_constraint_models=outcome_constraint_models,
        )

    if method == AcquisitionMethod.COST_WEIGHTED_EI and cost_model is None:
        logger.warning(
            "COST_WEIGHTED_EI requested but no cost model available. "
            "Falling back to standard Noisy Expected Improvement."
        )

    all_constraints = list(constraints) if constraints else []
    if outcome_constraint_models:
        for constraint_model, threshold in outcome_constraint_models:
            all_constraints.append(_make_outcome_constraint_callable(constraint_model, threshold))

    use_noisy = method != AcquisitionMethod.EXPECTED_IMPROVEMENT
    return create_single_objective_acquisition(
        model=model,  # type: ignore[arg-type]
        train_x=train_x,
        train_y=train_y,
        minimize=minimize,
        use_noisy=use_noisy,
        constraints=all_constraints if all_constraints else None,
    )


def _create_multi_objective_dispatch(
    model: ModelListGP | SingleTaskGP,
    ref_point: Tensor | None,
    train_x: Tensor,
    train_y: Tensor,
    *,
    minimize_mask: Tensor,
    method: AcquisitionMethod,
    constraints: list | None,
) -> AcquisitionFunction:
    """Dispatch multi-objective acquisition function creation.

    Validates that a reference point is provided, then delegates to
    create_multi_objective_acquisition.

    Args:
        model: Fitted ModelListGP
        ref_point: Reference point for hypervolume computation, in
            minimization form
        train_x: Training inputs
        train_y: Training outputs in minimization form
        minimize_mask: Boolean tensor of shape ``(n_objectives,)`` recording
            the user-facing direction of each objective (see
            :mod:`bo_engine.types`).
        method: Acquisition method
        constraints: Optional constraint callables

    Returns:
        Multi-objective acquisition function
    """
    if ref_point is None:
        raise ValueError("Reference point required for multi-objective optimization")

    return create_multi_objective_acquisition(
        model=model,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        ref_point=ref_point,
        train_x=train_x,
        train_y=train_y,
        minimize_mask=minimize_mask,
        method=method,
        constraints=constraints,
    )


def _make_outcome_constraint_callable(
    _constraint_model: SingleTaskGP,
    threshold: float,
) -> Callable[[Tensor], Tensor]:
    """Create a constraint callable for outcome constraints.

    Args:
        constraint_model: GP model predicting feasibility
        threshold: Probability threshold (typically 0.5)

    Returns:
        Callable that returns constraint satisfaction (positive = satisfied)
    """

    def constraint_callable(samples: Tensor) -> Tensor:
        """Constraint function: positive means feasible."""
        # samples shape: (num_samples, batch_size, 1)
        # For outcome constraints, the model predicts P(feasible)
        # We want P(feasible) > threshold
        # Constraint is satisfied when samples > threshold
        return samples.squeeze(-1) - threshold

    return constraint_callable


def create_cost_aware_acquisition(
    model: SingleTaskGP,
    train_y: Tensor,
    cost_model: SingleTaskGP,
    *,
    minimize: bool,
    outcome_constraint_models: list[tuple[SingleTaskGP, float]] | None = None,
) -> AcquisitionFunction:
    """Create EIpu (Expected Improvement per Unit cost) acquisition.

    EIpu = EI(x) / E[cost(x)]

    ``train_y`` must be in the canonical minimization form; see
    :mod:`bo_engine.types`.

    Args:
        model: Fitted objective model
        train_y: Training outputs in minimization form
        cost_model: Fitted cost model
        minimize: User-facing objective direction (required keyword so the
            caller cannot silently mismatch the sign convention)
        outcome_constraint_models: Optional outcome constraint models

    Returns:
        EIpuAcquisition function
    """
    _assert_minimization_form(minimize)
    # train_y is in minimization form (lower = better); min() is best.
    best_f = train_y.min().item()

    return EIpuAcquisition(
        model=model,
        cost_model=cost_model,
        best_f=best_f,
        outcome_constraint_models=outcome_constraint_models,
    )


class EIpuAcquisition(AcquisitionFunction):
    """Expected Improvement per Unit cost acquisition function.

    Computes EI(x) / E[cost(x)] to optimize for efficiency.
    """

    def __init__(
        self,
        model: SingleTaskGP,
        cost_model: SingleTaskGP,
        best_f: float,
        outcome_constraint_models: list[tuple[SingleTaskGP, float]] | None = None,
    ) -> None:
        """Initialize EIpu acquisition.

        Args:
            model: Fitted objective model
            cost_model: Fitted cost model
            best_f: Best observed value (for minimization)
            outcome_constraint_models: Optional outcome constraint models
        """
        super().__init__(model)
        self.ei = ExpectedImprovement(model=model, best_f=best_f)
        self.cost_model = cost_model
        self.outcome_constraint_models = outcome_constraint_models
        # X_pending is required for sequential optimization
        self._X_pending: Tensor | None = None

    @property
    def X_pending(self) -> Tensor | None:  # noqa: N802
        """Get pending candidates."""
        return self._X_pending

    @X_pending.setter
    def X_pending(self, value: Tensor | None) -> None:  # noqa: N802
        """Set pending candidates."""
        self._X_pending = value
        # Also set on the inner EI
        self.ei.X_pending = value

    def forward(self, X: Tensor) -> Tensor:  # noqa: N803
        """Compute EIpu acquisition value.

        Args:
            X: Candidate points of shape (..., q, d) where:
               - ... are batch dimensions (e.g., num_restarts, num_samples)
               - q is the batch size for joint acquisition (usually 1)
               - d is the input dimension

        Returns:
            Acquisition values of shape (...) matching EI output
        """
        # Compute EI - returns shape (...)
        ei_val = self.ei(X)

        # Compute expected cost
        # X has shape (..., q, d) and we need to average cost over q dimension
        self.cost_model.eval()
        with torch.no_grad():
            cost_posterior = self.cost_model.posterior(X)
            # posterior.mean has shape (..., q, 1)
            expected_cost = cost_posterior.mean

            # Average over q dimension and squeeze output dimension
            # Shape: (..., q, 1) -> (..., q) -> (...)
            expected_cost = expected_cost.squeeze(-1)  # (..., q)
            if expected_cost.dim() > ei_val.dim():
                # Average over q dimension to match EI shape
                expected_cost = expected_cost.mean(dim=-1)

            # Ensure positive cost
            expected_cost = expected_cost.clamp(min=1e-6)

        # Compute EI per unit cost (EIpu)
        eipu = ei_val / expected_cost

        # Apply outcome constraints if any
        if self.outcome_constraint_models:
            constraint_prob = torch.ones_like(eipu)
            for constraint_model, threshold in self.outcome_constraint_models:
                constraint_model.eval()
                with torch.no_grad():
                    prob_posterior = constraint_model.posterior(X)
                    prob_feasible = prob_posterior.mean.squeeze(-1)
                    if prob_feasible.dim() > eipu.dim():
                        prob_feasible = prob_feasible.mean(dim=-1)
                    # Binary thresholding: zero out infeasible points rather than
                    # smooth weighting (prob * acq).  This is a deliberate stability
                    # choice — BoTorch's smooth ConstrainedMCObjective can cause
                    # gradient vanishing near the threshold, making L-BFGS-B stall.
                    # The hard cutoff is more robust for the EIpu case where the
                    # cost denominator already introduces numerical sensitivity.
                    constraint_prob = constraint_prob * (prob_feasible > threshold).float()
            eipu = eipu * constraint_prob

        return eipu


def optimize_acquisition(
    acqf: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int = 1,
    num_restarts: int = 20,
    raw_samples: int = 512,
    spec: OptimizationSpec | None = None,
    x_avoid: Tensor | None = None,  # noqa: N803
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize acquisition function to find next candidates.

    Dispatches to the appropriate optimizer based on the search space type:
    - CONTINUOUS: standard L-BFGS-B via optimize_acqf
    - PURELY_CATEGORICAL: exhaustive enumeration via optimize_acqf_discrete
    - MIXED: per-combo continuous optimization via optimize_acqf_mixed

    Uses sequential greedy optimization for batch_size > 1.

    Args:
        acqf: Acquisition function to optimize
        bounds: Parameter bounds of shape (2, n_dims)
        batch_size: Number of candidates to generate
        num_restarts: Number of optimization restarts
        raw_samples: Number of raw samples for initialization
        spec: Optimization specification for discrete/mixed dispatch.
            If None, falls back to continuous optimization.
        x_avoid: Points to avoid (e.g., already-evaluated training data).
            Used by optimize_acqf_discrete to exclude known points.
        inequality_constraints: BoTorch linear inequality constraints (Ax <= b).
            Each tuple is (indices, coefficients, rhs).
        equality_constraints: BoTorch linear equality constraints (Ax = b).
            Each tuple is (indices, coefficients, rhs).

    Returns:
        Tuple of (candidates, acquisition_values) where:
        - candidates has shape (batch_size, n_dims)
        - acquisition_values has shape (batch_size,)
    """
    bounds = to_device(bounds)

    if spec is None:
        return _optimize_continuous(
            acqf,
            bounds,
            batch_size,
            num_restarts,
            raw_samples,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )

    space_type = classify_search_space(spec)

    if space_type == SearchSpaceType.PURELY_CATEGORICAL:
        return _optimize_discrete(acqf, spec, batch_size, x_avoid)
    elif space_type == SearchSpaceType.MIXED:
        n_combos = count_categorical_combinations(spec)
        if n_combos > MIXED_CATEGORICAL_COMBO_THRESHOLD:
            raise NotImplementedError(
                f"Mixed spaces with more than {MIXED_CATEGORICAL_COMBO_THRESHOLD} "
                f"categorical combinations are not yet supported (this space has "
                f"{n_combos}). Consider reducing the number of categories. "
                "A future version will support optimize_acqf_mixed_alternating "
                "with integer encoding for larger mixed spaces."
            )
        return _optimize_mixed(acqf, bounds, spec, batch_size, num_restarts, raw_samples)
    else:
        return _optimize_continuous(
            acqf,
            bounds,
            batch_size,
            num_restarts,
            raw_samples,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )


def _optimize_continuous(
    acqf: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize acquisition over continuous space using L-BFGS-B.

    Args:
        acqf: Acquisition function to optimize
        bounds: Parameter bounds of shape (2, n_dims)
        batch_size: Number of candidates to generate
        num_restarts: Number of optimization restarts
        raw_samples: Number of raw samples for initialization
        inequality_constraints: BoTorch linear inequality constraints (Ax <= b)
        equality_constraints: BoTorch linear equality constraints (Ax = b)

    Returns:
        Tuple of (candidates, acquisition_values)
    """
    kwargs: dict = {
        "acq_function": acqf,
        "bounds": bounds,
        "q": batch_size,
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
        "sequential": True,
        "options": {
            "batch_limit": 5,
            "maxiter": 200,
        },
    }
    if inequality_constraints:
        kwargs["inequality_constraints"] = inequality_constraints
    if equality_constraints:
        kwargs["equality_constraints"] = equality_constraints

    candidates, acq_values = optimize_acqf(**kwargs)
    return candidates, acq_values


def _optimize_discrete(
    acqf: AcquisitionFunction,
    spec: OptimizationSpec,
    batch_size: int,
    x_avoid: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Optimize acquisition over a purely categorical space.

    Enumerates all one-hot-encoded combinations and uses BoTorch's
    optimize_acqf_discrete with unique=True to avoid duplicate suggestions.

    Reference:
        https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf_discrete

    Args:
        acqf: Acquisition function to optimize
        spec: Optimization specification (purely categorical)
        batch_size: Number of candidates to generate
        x_avoid: Points to exclude from the choice set

    Returns:
        Tuple of (candidates, acquisition_values)
    """
    choices = enumerate_discrete_choices(spec)
    candidates, acq_values = optimize_acqf_discrete(
        acq_function=acqf,
        q=batch_size,
        choices=choices,
        unique=True,
        X_avoid=x_avoid,
    )
    return candidates, acq_values


def _optimize_mixed(
    acqf: AcquisitionFunction,
    bounds: Tensor,
    spec: OptimizationSpec,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
) -> tuple[Tensor, Tensor]:
    """Optimize acquisition over a mixed continuous + categorical space.

    For each categorical combination, runs L-BFGS-B optimization over the
    continuous dimensions, then returns the best result.

    Reference:
        https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf_mixed

    Args:
        acqf: Acquisition function to optimize
        bounds: Parameter bounds of shape (2, n_dims)
        spec: Optimization specification (mixed space)
        batch_size: Number of candidates to generate
        num_restarts: Number of optimization restarts
        raw_samples: Number of raw samples for initialization

    Returns:
        Tuple of (candidates, acquisition_values)
    """
    fixed_features = build_fixed_features_list(spec)
    candidates, acq_values = optimize_acqf_mixed(
        acq_function=acqf,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        fixed_features_list=fixed_features,
        options={
            "batch_limit": 5,
            "maxiter": 200,
        },
    )
    return candidates, acq_values


def get_best_observed_value(
    train_y: Tensor,
    minimize: bool = True,
) -> float:
    """Get best observed value for single-objective optimization.

    Args:
        train_y: Training outputs of shape (n_samples, 1) or (n_samples,)
        minimize: If True, return minimum; else return maximum

    Returns:
        Best observed value
    """
    if train_y.dim() > 1:
        train_y = train_y.squeeze(-1)

    if minimize:
        return train_y.min().item()
    else:
        return train_y.max().item()
