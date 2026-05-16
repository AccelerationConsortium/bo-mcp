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

from bo_engine.constants import (
    MIXED_CATEGORICAL_COMBO_THRESHOLD,
    NUMERICAL_EPSILON,
    RESTART_WARN_TOLERANCE,
)
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


def _validate_linear_constraint_entry(
    entry: tuple[Tensor, Tensor, float],
    n_dims: int,
    label: str,
) -> None:
    """Validate a single ``(indices, coefficients, rhs)`` tuple.

    Encapsulates the per-entry shape checks so the outer driver can
    iterate without exceeding cognitive-complexity limits.
    """
    if not isinstance(entry, tuple) or len(entry) != 3:
        raise ValueError(f"{label} must be a (indices, coefficients, rhs) tuple")
    indices, coefficients, rhs = entry
    if not isinstance(indices, Tensor) or indices.dim() != 1:
        raise ValueError(f"{label}.indices must be a 1-D tensor")
    if not isinstance(coefficients, Tensor) or coefficients.dim() != 1:
        raise ValueError(f"{label}.coefficients must be a 1-D tensor")
    if indices.numel() != coefficients.numel():
        raise ValueError(
            f"{label} indices ({indices.numel()}) and "
            f"coefficients ({coefficients.numel()}) must have the same length"
        )
    if indices.numel() == 0:
        raise ValueError(f"{label} must reference at least one parameter")
    if not torch.isfinite(torch.as_tensor(rhs, dtype=torch.float64)).item():
        raise ValueError(f"{label}.rhs must be finite, got {rhs!r}")
    index_min = int(indices.min().item())
    index_max = int(indices.max().item())
    if index_min < 0 or index_max >= n_dims:
        raise ValueError(
            f"{label} references parameter index out of range "
            f"[0, {n_dims}); got min={index_min}, max={index_max}"
        )


def _validate_linear_constraints(
    constraints: list[tuple[Tensor, Tensor, float]] | None,
    n_dims: int,
    kind: str,
) -> None:
    """Shape-validate BoTorch linear constraints before they reach ``optimize_acqf``.

    BoTorch surfaces shape mismatches deep inside its inner loops with messages
    that don't name the offending tuple, so we check the contract here while we
    still know which constraint failed.
    """
    if not constraints:
        return
    for position, entry in enumerate(constraints):
        _validate_linear_constraint_entry(entry, n_dims, f"{kind}_constraints[{position}]")


def _resolve_restart_budget(
    spec: OptimizationSpec | None,
    bounds: Tensor,
    num_restarts: int | None,
    raw_samples: int | None,
) -> tuple[int, int]:
    """Return the effective ``(num_restarts, raw_samples)`` for this call.

    Explicit ``num_restarts`` / ``raw_samples`` arguments override the spec's
    :class:`AcquisitionOptimizationConfig`. Otherwise the dimension-adaptive
    defaults derived from the acquisition-input width of ``bounds`` apply.
    """
    from bo_engine.types import AcquisitionOptimizationConfig  # local to avoid cycles

    n_dims = int(bounds.shape[-1])
    config = (
        spec.acquisition_optimization
        if spec is not None and spec.acquisition_optimization is not None
        else AcquisitionOptimizationConfig()
    )
    default_restarts, default_samples = config.resolve(n_dims)
    restarts = int(num_restarts) if num_restarts is not None else default_restarts
    samples = int(raw_samples) if raw_samples is not None else default_samples
    return restarts, samples


def optimize_acquisition(
    acqf: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int = 1,
    num_restarts: int | None = None,
    raw_samples: int | None = None,
    spec: OptimizationSpec | None = None,
    x_avoid: Tensor | None = None,  # noqa: N803
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    X_pending: Tensor | None = None,  # noqa: N803
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
        num_restarts: Optional override for the restart count. When ``None``,
            the value is derived from ``spec.acquisition_optimization`` and
            falls back to the dimension-adaptive default in
            :class:`AcquisitionOptimizationConfig`.
        raw_samples: Optional override for the raw-sample budget. Same
            resolution rules as ``num_restarts``.
        spec: Optimization specification for discrete/mixed dispatch.
            If None, falls back to continuous optimization.
        x_avoid: Points to avoid (e.g., already-evaluated training data).
            Used by optimize_acqf_discrete to exclude known points.
        inequality_constraints: BoTorch linear inequality constraints (Ax <= b).
            Each tuple is (indices, coefficients, rhs).
        equality_constraints: BoTorch linear equality constraints (Ax = b).
            Each tuple is (indices, coefficients, rhs).
        X_pending: In-flight candidates (shape ``(n_pending, n_dims)``) that
            should condition the acquisition so new suggestions are diverse
            from pending experiments.  For continuous and mixed spaces this
            is forwarded to the acquisition via ``set_X_pending`` (consumed
            by the MC acquisition's joint optimization).  For purely
            categorical spaces the pending rows are concatenated into
            ``x_avoid`` so the discrete optimizer excludes them.

    Returns:
        Tuple of (candidates, acquisition_values) where:
        - candidates has shape (batch_size, n_dims)
        - acquisition_values has shape (batch_size,)
    """
    bounds = to_device(bounds)
    n_dims = int(bounds.shape[-1])
    _validate_linear_constraints(inequality_constraints, n_dims, "inequality")
    _validate_linear_constraints(equality_constraints, n_dims, "equality")

    if X_pending is not None:
        X_pending = to_device(X_pending)

    effective_restarts, effective_samples = _resolve_restart_budget(
        spec, bounds, num_restarts, raw_samples
    )
    logger.debug(
        "Acquisition optimization budget: n_dims=%d, num_restarts=%d, raw_samples=%d",
        int(bounds.shape[-1]),
        effective_restarts,
        effective_samples,
    )

    if spec is None:
        _apply_pending_to_acqf(acqf, X_pending)
        return _optimize_continuous(
            acqf,
            bounds,
            batch_size,
            effective_restarts,
            effective_samples,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )

    space_type = classify_search_space(spec)

    if space_type == SearchSpaceType.PURELY_CATEGORICAL:
        merged_avoid = _merge_avoid_tensors(x_avoid, X_pending)
        return _optimize_discrete(acqf, spec, batch_size, merged_avoid)
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
        _apply_pending_to_acqf(acqf, X_pending)
        return _optimize_mixed(
            acqf, bounds, spec, batch_size, effective_restarts, effective_samples
        )
    else:
        _apply_pending_to_acqf(acqf, X_pending)
        return _optimize_continuous(
            acqf,
            bounds,
            batch_size,
            effective_restarts,
            effective_samples,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )


def _apply_pending_to_acqf(
    acqf: AcquisitionFunction,
    X_pending: Tensor | None,  # noqa: N803
) -> None:
    """Route pending points into the acquisition's batch-conditioning slot.

    MC acquisitions (``qLog*``) expose ``set_X_pending`` so the sequential
    greedy optimizer inside ``optimize_acqf`` conditions each new candidate
    on both the previously-selected batch members *and* the supplied
    pending in-flight experiments.  Without this call parallel / batch BO
    silently clusters candidates around pending regions.
    """
    if X_pending is None or X_pending.numel() == 0:
        return
    set_pending = getattr(acqf, "set_X_pending", None)
    if callable(set_pending):
        set_pending(X_pending)
    else:
        # Fall back to the attribute for custom acquisitions (e.g. EIpu)
        # that expose a property setter but no ``set_X_pending`` method.
        acqf.X_pending = X_pending


def _merge_avoid_tensors(
    x_avoid: Tensor | None,  # noqa: N803
    X_pending: Tensor | None,  # noqa: N803
) -> Tensor | None:
    """Concatenate pending rows into an ``x_avoid`` tensor.

    Purely-categorical spaces use ``optimize_acqf_discrete``, which doesn't
    read ``X_pending`` from the acquisition — pending points must be
    excluded from the choice set instead.  Both tensors share shape
    ``(n_points, n_dims)`` and live on the same device after
    ``to_device``.
    """
    if X_pending is None or X_pending.numel() == 0:
        return x_avoid
    if x_avoid is None or x_avoid.numel() == 0:
        return X_pending
    return torch.cat([x_avoid, X_pending], dim=0)


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

    Diagnostics: when ``batch_size == 1`` we ask BoTorch for the full set of
    per-restart results (``return_best_only=False``) and route them through
    :func:`_log_restart_diagnostics` before collapsing to the best restart.
    For ``batch_size > 1`` we keep the existing sequential greedy path —
    BoTorch does not support ``return_best_only=False`` together with
    sequential greedy optimization, so diagnostics are skipped there.
    """
    capture_restarts = batch_size == 1
    kwargs: dict = {
        "acq_function": acqf,
        "bounds": bounds,
        "q": batch_size,
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
        "sequential": not capture_restarts,
        "return_best_only": not capture_restarts,
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

    if capture_restarts:
        _log_restart_diagnostics(acq_values, candidates)
        best_idx = int(acq_values.argmax().item())
        candidates = candidates[best_idx]
        acq_values = acq_values[best_idx : best_idx + 1]
    return candidates, acq_values


def _log_restart_diagnostics(acq_values: Tensor, candidates: Tensor) -> None:
    """Log per-restart acquisition diagnostics for ``optimize_acqf``.

    Captures the dispersion across the multi-start optimization so the
    practitioner can tell whether the restarts are exploring distinct basins
    or whether they all collapsed into the same neighbourhood — a hallmark of
    pervasive local minima or a near-uniform acquisition landscape.

    The top-3 (acquisition value, candidate) pairs are emitted at DEBUG; a
    WARNING is emitted when the relative gap between the best and median
    restart is below :data:`RESTART_WARN_TOLERANCE`, i.e.

    ``(best - median) / max(|best|, |median|, ε) < RESTART_WARN_TOLERANCE``.

    Args:
        acq_values: Per-restart acquisition values of shape
            ``(num_restarts,)`` from ``optimize_acqf(return_best_only=False)``.
        candidates: Per-restart candidate solutions of shape
            ``(num_restarts, q, d)``. Only used for the DEBUG dump.
    """
    if acq_values.numel() == 0:
        return

    values = acq_values.detach().reshape(-1)
    n_restarts = int(values.numel())

    sorted_vals, sorted_idx = torch.sort(values, descending=True)
    top_n = min(3, n_restarts)
    if logger.isEnabledFor(logging.DEBUG):
        for rank in range(top_n):
            idx = int(sorted_idx[rank].item())
            value = float(sorted_vals[rank].item())
            candidate = candidates[idx].detach().cpu().tolist()
            logger.debug(
                "Acquisition restart rank %d: value=%.6g candidate=%s",
                rank + 1,
                value,
                candidate,
            )

    if n_restarts < 2:
        return

    best = float(sorted_vals[0].item())
    median = float(values.median().item())
    denom = max(abs(best), abs(median), NUMERICAL_EPSILON)
    relative_gap = (best - median) / denom
    if relative_gap < RESTART_WARN_TOLERANCE:
        logger.warning(
            "Acquisition restart dispersion is tight: best=%.6g median=%.6g "
            "relative_gap=%.3g < %.3g. Restarts may be trapped in widespread "
            "local minima or the acquisition surface may be near-uniform.",
            best,
            median,
            relative_gap,
            RESTART_WARN_TOLERANCE,
        )


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
