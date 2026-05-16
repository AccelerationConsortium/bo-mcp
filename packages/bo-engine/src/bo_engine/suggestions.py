"""Suggestion generation for Bayesian Optimization.

Supports both single-objective and multi-objective optimization.

v1.0.1: Added automatic single-objective detection and handling
v1.1: Added acquisition method selection and input warping support
v1.2: Added TuRBO integration for high-dimensional optimization
v1.3: Added outcome constraints and cost-aware optimization
v2.3: Added GPU auto-detection and acceleration
"""

from __future__ import annotations

import logging
import random
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP

    GPModel = SingleTaskGP | ModelListGP

import numpy as np
import torch
from torch import Tensor

from bo_engine.acquisition import (
    create_acquisition,
    optimize_acquisition,
)
from bo_engine.constants import (
    CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD,
    CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD,
    MAX_RANDOM_SEED,
    MIN_OBSERVATIONS_FOR_MODEL,
    SEED_ITERATION_OFFSET,
)
from bo_engine.constraints import (
    build_botorch_linear_constraints,
)
from bo_engine.device import get_device, get_dtype
from bo_engine.models import (
    create_and_fit_model,
    create_and_fit_single_task_model,
)
from bo_engine.reference_point import (
    ReferencePointConfig,
    ReferencePointStrategy,
    get_reference_point_dynamic,
)
from bo_engine.transforms import (
    decode_categorical,
    encode_categorical,
    get_bounds_tensor,
    stack_encoded_values,
)
from bo_engine.turbo import (
    TurboState,
    assert_unit_scale_targets,
    create_turbo_state,
    get_turbo_bounds,
    should_use_turbo,
    update_turbo_state,
)
from bo_engine.types import (
    AcquisitionMethod,
    GenerationContext,
    ObservationData,
    OptimizationSpec,
    SuggestionResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-exports: initial-design helpers live in :mod:`bo_engine.initial_design`
# after the suggestions god-module split. Imports from this module continue
# to resolve here so external callers do not need to change.
# ---------------------------------------------------------------------------

from bo_engine.initial_design import (  # noqa: E402
    _apply_constraints_to_samples,
    _guard_categorical_space_exhaustion,
    generate_initial_design,
)


def _resolve_acquisition_seed(
    spec: OptimizationSpec,
    iteration: int,
    rng: np.random.Generator | None,
) -> int:
    """Derive the acquisition seed for :func:`generate_next_batch`.

    Precedence:

    1. ``rng`` wins whenever supplied so external callers keep control
       of their RNG pipeline.
    2. ``spec.random_seed`` seeds a deterministic per-iteration stride
       so two independent replays of the same campaign state produce
       identical acquisition candidates.
    3. Falls back to :func:`random.randint` only when neither is
       supplied; this path is documented as non-reproducible.
    """
    if rng is not None:
        return int(rng.integers(0, MAX_RANDOM_SEED))
    if spec.random_seed is not None:
        return (spec.random_seed + iteration * SEED_ITERATION_OFFSET) % MAX_RANDOM_SEED
    # Deliberately non-reproducible — no seed was supplied. Used in tests
    # and ad-hoc campaigns where reproducibility is not required.
    return random.randint(0, MAX_RANDOM_SEED)  # noqa: S311


def generate_next_batch(
    spec: OptimizationSpec,
    observations: list[ObservationData],
    batch_size: int | None = None,
    iteration: int = 0,
    turbo_state: TurboState | None = None,
    rng: np.random.Generator | None = None,
    pending_points: list[dict[str, Any]] | None = None,
) -> tuple[list[SuggestionResult], TurboState | None]:
    """Generate next batch of suggestions using Bayesian Optimization.

    Automatically detects single vs multi-objective optimization and selects
    appropriate model and acquisition function. Supports TuRBO for high-dimensional
    single-objective problems.

    Args:
        spec: Optimization specification
        observations: Historical observations
        batch_size: Number of suggestions to generate (default: spec.batch_size)
        iteration: Current iteration number
        turbo_state: Optional TuRBO state for trust region optimization
        rng: Optional NumPy random generator for deterministic behavior.
            Create with np.random.default_rng(seed) for reproducibility.
        pending_points: In-flight suggestions (parameter-value dicts) that
            have not yet been observed.  When supplied they are
            (a) excluded from the initial-design fallback so Sobol does
            not re-issue a pending combination, and (b) forwarded to the
            acquisition optimizer as ``X_pending`` so parallel / batch BO
            conditions new candidates on the in-flight batch instead of
            silently clustering around it.

    Reproducibility:
        The acquisition seed is resolved with the following precedence:

        1. ``rng`` is supplied — the seed is drawn from it, taking
           precedence over any other source so callers that already
           manage a :class:`numpy.random.Generator` stay in control.
        2. ``spec.random_seed`` is set — the seed is derived
           deterministically as
           ``(spec.random_seed + iteration * SEED_ITERATION_OFFSET)
           % MAX_RANDOM_SEED``.  Two independent calls on the same
           campaign state at the same iteration therefore produce
           identical acquisition candidates.
        3. Neither is provided — the seed is drawn from the Python
           stdlib ``random`` module, which is explicitly
           non-reproducible across process runs.

        The resolved seed is installed on the global ``torch`` RNG
        via ``torch.manual_seed`` because BoTorch's ``optimize_acqf``
        sampler path consults the global state rather than accepting
        an explicit generator.  The mutation is scoped inside a
        ``torch.random.fork_rng(devices=[])`` block so concurrent
        callers (e.g. under ``asyncio.to_thread``) cannot race on
        the process-wide seed: the prior RNG state is saved on entry
        and restored on every return path.

    Returns:
        Tuple of (List of SuggestionResult objects, Updated TurboState or None)
    """
    if batch_size is None:
        batch_size = spec.batch_size

    random_seed = _resolve_acquisition_seed(spec, iteration, rng)

    # fork_rng isolates the torch global RNG mutation below so concurrent
    # callers (e.g. under asyncio.to_thread) cannot race on the seed —
    # prior state is saved on entry and restored on every return path.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)

        # Short-circuit for finite (purely-categorical) spaces whose unique
        # combinations are already exhausted — neither the initial-design
        # fallback nor the discrete acquisition optimizer can invent new
        # points, and BoTorch's optimize_acqf_discrete will otherwise raise
        # an opaque error when X_avoid covers the entire choice set.
        _guard_categorical_space_exhaustion(spec, observations, batch_size)

        # If not enough data, fall back to initial design.
        # Require at least n_params+1 observations so the GP kernel has more data points
        # than lengthscale hyperparameters to estimate (slightly overdetermined).
        # Note: initial_design_size (default 2*n_params+1) is a separate, stricter
        # recommendation for how many Sobol points to generate — but a user who provides
        # n_params+1 observations from prior data should not be forced to wait longer.
        min_model_data = max(MIN_OBSERVATIONS_FOR_MODEL, spec.n_parameters + 1)
        if spec.initial_design_size is not None:
            min_data = max(min_model_data, spec.initial_design_size)
        else:
            min_data = min_model_data
        if len(observations) < min_data:
            pending = pending_points or []
            excluded = [obs.parameter_values for obs in observations] + list(pending)
            designs = generate_initial_design(
                spec,
                batch_size,
                n_drawn=len(observations),
                excluded_points=excluded,
            )
            suggestions = [
                SuggestionResult(
                    parameter_values=design,
                    iteration=iteration,
                    batch_index=i,
                    generation_method="initial_design",
                    random_seed=random_seed,
                    explanation=(
                        f"Initial design point {i + 1}/{len(designs)} using Sobol sequence. "
                        "Initial designs explore the parameter space before "
                        "model-guided suggestions."
                    ),
                )
                for i, design in enumerate(designs)
            ]
            return suggestions, turbo_state

        # Determine if single or multi-objective
        is_single_objective = spec.n_objectives == 1

        # Prepare training data
        train_x, train_y = _prepare_training_data(observations, spec)
        bounds = get_bounds_tensor(spec)

        # Per-observation measurement uncertainty (variance). Only used when
        # every observation carries every objective's stddev; partial
        # coverage falls back to the trainable noise prior.
        train_yvar = _prepare_train_yvar(observations, spec)

        # Prepare cost data if cost-aware optimization is enabled
        train_costs = None
        if spec.use_cost_aware:
            train_costs = _prepare_cost_data(observations, use_cost_aware=True)

        # Encode pending points to the same coordinate system as train_x so the
        # acquisition optimizer sees them as X_pending.  ``None`` skips the
        # X_pending branch entirely; ``numel()==0`` means "no valid pending".
        pending_tensor = _encode_pending_points(pending_points, spec) if pending_points else None

        # Create generation context to bundle parameters
        ctx = GenerationContext(
            spec=spec,
            train_x=train_x,
            train_y=train_y,
            bounds=bounds,
            batch_size=batch_size,
            iteration=iteration,
            random_seed=random_seed,
            turbo_state=turbo_state,
            observations=observations,
            train_costs=train_costs,
            pending_x=pending_tensor,
            train_yvar=train_yvar,
        )

        if is_single_objective:
            return _generate_single_objective_batch(ctx)

        # TuRBO not supported for multi-objective
        if turbo_state is not None or spec.use_turbo:
            warnings.warn(
                "TuRBO is designed for single-objective optimization. "
                "It will be ignored for multi-objective problems. "
                "Standard L-BFGS-B optimization will be used instead.",
                UserWarning,
                stacklevel=2,
            )
        suggestions = _generate_multi_objective_batch(ctx)
        return suggestions, None


def _encode_pending_points(
    pending_points: list[dict[str, Any]] | None,
    spec: OptimizationSpec,
) -> Tensor | None:
    """Encode pending parameter dicts into the tensor form consumed by BoTorch.

    Returns ``None`` when no pending points are supplied or when every
    candidate is missing at least one spec parameter (encoding would raise
    ``KeyError``).  Malformed entries are skipped with a debug log rather
    than crashing the suggestion path — the acquisition still benefits
    from the subset that encodes cleanly.
    """
    if not pending_points:
        return None
    valid: list[dict[str, Any]] = []
    for params in pending_points:
        if all(p.name in params for p in spec.parameters):
            valid.append(params)
        else:
            logger.debug("Skipping pending point missing required parameter(s): %s", params)
    if not valid:
        return None
    return stack_encoded_values(valid, spec)


def _initialize_turbo_state(
    spec: OptimizationSpec,
    train_y_bo: Tensor,
    batch_size: int,
    turbo_state: TurboState | None,
) -> TurboState | None:
    """Initialize TuRBO state if applicable.

    Forwards ``spec.turbo_config`` overrides (or paper defaults when unset)
    into :func:`bo_engine.turbo.create_turbo_state` and asserts that the
    training targets sit close to unit scale before construction — see
    :class:`bo_engine.turbo.TurboState` for the scale assumption.

    Args:
        spec: Optimization specification
        train_y_bo: Training outputs (BoTorch convention - minimization)
        batch_size: Number of suggestions per batch
        turbo_state: Existing TuRBO state if any

    Returns:
        Initialized TuRBO state or None if not using TuRBO
    """
    use_turbo = spec.use_turbo or (turbo_state is not None) or should_use_turbo(spec.n_parameters)
    if use_turbo and turbo_state is None:
        assert_unit_scale_targets(train_y_bo)
        best_y = train_y_bo.min().item()
        config = spec.turbo_config
        if config is None:
            turbo_state = create_turbo_state(
                dim=spec.n_parameters,
                batch_size=batch_size,
                initial_best_value=-best_y,
            )
        else:
            turbo_state = create_turbo_state(
                dim=spec.n_parameters,
                batch_size=batch_size,
                initial_best_value=-best_y,
                initial_length=config.initial_length,
                length_min=config.length_min,
                length_max=config.length_max,
                success_tolerance=config.success_tolerance,
                failure_tolerance=config.failure_tolerance,
            )
    return turbo_state


def _compute_turbo_bounds(
    turbo_state: TurboState | None,
    train_x: Tensor,
    train_y_bo: Tensor,
    model: SingleTaskGP,
    bounds: Tensor,
) -> tuple[Tensor, str]:
    """Compute trust region bounds for TuRBO.

    Args:
        turbo_state: Current TuRBO state
        train_x: Training inputs
        train_y_bo: Training outputs (BoTorch convention)
        model: Fitted single-objective GP model
        bounds: Original parameter bounds

    Returns:
        Tuple of (optimization_bounds, turbo_info_string)
    """
    if turbo_state is None or turbo_state.restart_triggered:
        return bounds.clone(), ""

    tr_lb, tr_ub = get_turbo_bounds(turbo_state, train_x, train_y_bo, model)
    opt_bounds = torch.stack(
        [
            bounds[0] + tr_lb * (bounds[1] - bounds[0]),
            bounds[0] + tr_ub * (bounds[1] - bounds[0]),
        ]
    )
    turbo_info = f" TuRBO length={turbo_state.length:.4f}."
    return opt_bounds, turbo_info


def _get_model_predictions(model: GPModel, candidates: Tensor) -> tuple[Tensor, Tensor]:
    """Get model predictions (mean and std) at candidate points.

    Args:
        model: Fitted GP model
        candidates: Candidate points to evaluate

    Returns:
        Tuple of (means, stds) where each has shape matching the model output
    """
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(candidates)
        means = posterior.mean.squeeze(-1)
        variances = posterior.variance.squeeze(-1)
        if means.dim() == 0:
            means = means.unsqueeze(0)
        if variances.dim() == 0:
            variances = variances.unsqueeze(0)
        stds = variances.sqrt()
    return means, stds


def _extract_scalar_prediction(
    tensor: Tensor,
    index: int,
) -> float | None:
    """Safely extract a scalar from a 0-d or 1-d tensor at the given index.

    Returns None if the index is out of bounds.

    Args:
        tensor: Tensor of acquisition values, means, or stds
        index: Batch index

    Returns:
        Float value or None
    """
    if tensor.dim() == 0:
        return tensor.item() if index == 0 else None
    return tensor[index].item() if index < tensor.numel() else None


def _build_single_objective_provenance(
    means: Tensor,
    stds: Tensor,
    acq_values: Tensor,
    index: int,
    obj_name: str,
    minimize: bool,
) -> tuple[
    float | None,
    float | None,
    float | None,
    dict[str, float] | None,
    dict[str, float] | None,
]:
    """Build provenance data for a single-objective candidate.

    Extracts acquisition value, uncertainty, and predicted objectives for one
    candidate in the batch.

    Args:
        means: Posterior mean predictions
        stds: Posterior std predictions
        acq_values: Acquisition function values
        index: Candidate index in the batch
        obj_name: Objective name
        minimize: Whether the objective is minimized

    Returns:
        Tuple of (acq_val, std_val, confidence_level_unused,
                  predicted_objectives, predicted_std)
    """
    std_val = _extract_scalar_prediction(stds, index)
    acq_val = _extract_scalar_prediction(acq_values, index)

    # For minimization, means are negated (BoTorch convention) — un-negate for storage.
    pred_mean = _extract_scalar_prediction(means, index)
    if pred_mean is not None and not minimize:
        pred_mean = -pred_mean  # undo BoTorch negation

    predicted_objectives = {obj_name: pred_mean} if pred_mean is not None else None
    predicted_std_dict = {obj_name: std_val} if std_val is not None else None

    return acq_val, std_val, None, predicted_objectives, predicted_std_dict


def _create_single_objective_suggestions(
    candidates: Tensor,
    acq_values: Tensor,
    means: Tensor,
    stds: Tensor,
    spec: OptimizationSpec,
    train_y: Tensor,
    iteration: int,
    random_seed: int,
    method: AcquisitionMethod,
    turbo_state: TurboState | None,
    turbo_info: str,
) -> list[SuggestionResult]:
    """Create SuggestionResult objects from optimization results.

    Args:
        candidates: Optimized candidate points
        acq_values: Acquisition function values
        means: Posterior mean predictions at candidates
        stds: Posterior std predictions at candidates
        spec: Optimization specification
        train_y: Original training outputs (not negated)
        iteration: Current iteration number
        random_seed: Random seed for reproducibility
        method: Acquisition method used
        turbo_state: Optional TuRBO state
        turbo_info: TuRBO info string for explanation

    Returns:
        List of SuggestionResult objects
    """
    minimize = spec.objectives[0].minimize
    batch_size = candidates.shape[0]

    # Get best observed value for context
    if minimize:
        best_observed = train_y.min().item()
        best_str = "lowest"
    else:
        best_observed = train_y.max().item()
        best_str = "highest"

    obj_name = spec.objectives[0].name
    acq_name = method.value
    gen_method = "bo" if turbo_state is None else "turbo"

    suggestions = []
    for i in range(batch_size):
        values = decode_categorical(candidates[i], spec)

        acq_val, std_val, _, predicted_objectives, predicted_std_dict = (
            _build_single_objective_provenance(means, stds, acq_values, i, obj_name, minimize)
        )
        confidence_level = _get_confidence_level(std_val)

        explanation = (
            f"Suggested by {acq_name} acquisition function. "
            f"Current {best_str} observed value: {best_observed:.4f}. "
            f"This point is predicted to improve the objective.{turbo_info}"
        )

        suggestion = SuggestionResult(
            parameter_values=values,
            iteration=iteration,
            batch_index=i,
            generation_method=gen_method,
            random_seed=random_seed,
            acquisition_value=acq_val,
            model_uncertainty=std_val,
            acquisition_function=acq_name,
            model_type="SingleTaskGP (Gaussian Process)",
            model_version=iteration,
            confidence_level=confidence_level,
            explanation=explanation,
            predicted_objectives=predicted_objectives,
            predicted_std=predicted_std_dict,
        )
        suggestions.append(suggestion)

    return suggestions


def _generate_single_objective_batch(
    ctx: GenerationContext,
) -> tuple[list[SuggestionResult], TurboState | None]:
    """Generate suggestions for single-objective optimization.

    Uses noisy EI (or EI) acquisition function. Supports TuRBO for
    high-dimensional problems, outcome constraints, and cost-aware optimization.

    Args:
        ctx: Generation context containing all parameters

    Returns:
        Tuple of (suggestions, updated turbo_state)
    """
    spec = ctx.spec
    train_x = ctx.train_x
    train_y = ctx.train_y
    bounds = ctx.bounds
    batch_size = ctx.batch_size
    turbo_state = ctx.turbo_state
    observations = ctx.observations
    train_costs = ctx.train_costs
    train_yvar = ctx.train_yvar

    minimize = spec.objectives[0].minimize
    log_transform = spec.objectives[0].log_transform
    if log_transform and not minimize:
        # BoTorch's ``Log`` outcome transform requires strictly positive
        # targets and is applied inside the GP after we negate ``train_y``
        # to enforce minimization. For a maximize objective negation flips
        # the sign of every positive observation to negative, which makes
        # the log step ill-defined. Surface this at the boundary instead of
        # letting BoTorch raise a less actionable error during fit.
        raise ValueError(
            "ObjectiveSpec.log_transform=True is only supported for "
            "minimize=True objectives. For a maximize objective with a "
            "multi-decade target, either flip the objective definition "
            "(minimize the negative log) or pre-transform the data."
        )

    # Negate if maximizing (BoTorch assumes minimization). Variance is
    # sign-invariant -- ``Var(-Y) == Var(Y)`` -- so ``train_yvar`` flows
    # through unchanged.
    train_y_bo = -train_y if not minimize else train_y.clone()

    # Create and fit model. When every observation has measurement
    # uncertainty for this objective, route through the
    # ``FixedNoiseGaussianLikelihood`` path so the GP trusts the user's
    # known noise instead of re-estimating it from MLL.
    model = create_and_fit_single_task_model(
        train_x,
        train_y_bo,
        bounds,
        use_input_warping=spec.use_input_warping,
        train_yvar=train_yvar,
        log_transform=log_transform,
    )

    # Outcome constraint models (constraints on OUTPUT space)
    outcome_constraints = None
    if spec.outcome_constraints and observations:
        outcome_constraints = _build_outcome_constraint_models(spec, observations, train_x, bounds)

    # Cost model for EIpu
    cost_model = None
    if spec.use_cost_aware and train_costs is not None:
        cost_model = create_and_fit_single_task_model(
            train_x, train_costs.unsqueeze(-1), bounds, use_input_warping=False
        )

    # TuRBO state management
    turbo_state = _initialize_turbo_state(spec, train_y_bo, batch_size, turbo_state)
    opt_bounds, turbo_info = _compute_turbo_bounds(turbo_state, train_x, train_y_bo, model, bounds)

    # Acquisition method selection
    method = spec.acquisition_method
    if method == AcquisitionMethod.AUTO:
        method = (
            AcquisitionMethod.COST_WEIGHTED_EI
            if spec.use_cost_aware
            else AcquisitionMethod.NOISY_EI
        )

    # Build native linear constraints for the optimizer
    ineq_constraints = None
    eq_constraints = None
    projection_constraints: list = []
    if spec.constraints:
        ineq_constraints, eq_constraints, projection_constraints = build_botorch_linear_constraints(
            spec
        )

    # Create acquisition function.  ``train_y_bo`` is already in the
    # canonical minimization form (see bo_engine.types) because we
    # negated above when ``not minimize``; the factory therefore always
    # receives ``minimize=True``.
    acqf = create_acquisition(
        model=model,
        ref_point=None,
        train_x=train_x,
        train_y=train_y_bo,
        n_objectives=1,
        minimize=True,
        method=method,
        constraints=None,
        outcome_constraint_models=outcome_constraints,
        cost_model=cost_model,
    )
    # Optimize with native constraints where possible
    candidates, acq_values = optimize_acquisition(
        acqf,
        opt_bounds,
        batch_size,
        spec=spec,
        x_avoid=train_x,
        inequality_constraints=ineq_constraints or None,
        equality_constraints=eq_constraints or None,
        X_pending=ctx.pending_x,
    )

    # Post-hoc projection only for constraints that couldn't be handled natively
    if projection_constraints:
        candidates = _apply_constraints_to_samples(candidates, spec, bounds)

    # Get model predictions and create suggestions
    means, stds = _get_model_predictions(model, candidates)
    suggestions = _create_single_objective_suggestions(
        candidates,
        acq_values,
        means,
        stds,
        spec,
        train_y,
        ctx.iteration,
        ctx.random_seed,
        method,
        turbo_state,
        turbo_info,
    )

    return suggestions, turbo_state


def _build_multi_objective_provenance(
    means: Tensor,
    stds: Tensor,
    acq_values: Tensor,
    index: int,
    spec: OptimizationSpec,
) -> tuple[float | None, float | None, dict[str, float], dict[str, float]]:
    """Build provenance data for a multi-objective candidate.

    Extracts acquisition value, average uncertainty, and per-objective
    predictions for one candidate in the batch.

    Args:
        means: Posterior mean predictions (batch x n_objectives)
        stds: Posterior std predictions (batch x n_objectives)
        acq_values: Acquisition function values
        index: Candidate index in the batch
        spec: Optimization specification

    Returns:
        Tuple of (acq_val, avg_std, predicted_objectives, predicted_std)
    """
    acq_val = _extract_scalar_prediction(acq_values, index)

    # Average std across objectives for confidence level
    if stds.dim() > 1 and index < stds.shape[0]:
        avg_std: float | None = stds[index].mean().item()
    elif index < stds.numel():
        avg_std = stds[index].item()
    else:
        avg_std = None

    # Per-objective predictions; un-negate maximization objectives.
    predicted_objectives: dict[str, float] = {}
    predicted_std_dict: dict[str, float] = {}
    for j, obj in enumerate(spec.objectives):
        if means.dim() > 1 and index < means.shape[0] and j < means.shape[1]:
            pred = means[index, j].item()
            predicted_objectives[obj.name] = pred if obj.minimize else -pred
        if stds.dim() > 1 and index < stds.shape[0] and j < stds.shape[1]:
            predicted_std_dict[obj.name] = stds[index, j].item()

    return acq_val, avg_std, predicted_objectives, predicted_std_dict


def _build_multi_objective_explanation(
    acq_name: str,
    acq_val: float | None,
) -> str:
    """Build a human-readable explanation for a multi-objective suggestion.

    Args:
        acq_name: Name of the acquisition function
        acq_val: Acquisition function value (may be None)

    Returns:
        Explanation string
    """
    if acq_val is not None and acq_val > 0:
        return (
            f"Suggested by {acq_name} acquisition function with expected improvement "
            f"of {acq_val:.4f}. This point is predicted to expand the Pareto front."
        )
    return (
        f"Suggested by {acq_name} acquisition function to explore promising regions "
        "and expand the Pareto front of non-dominated solutions."
    )


def _create_multi_objective_suggestions(
    candidates: Tensor,
    acq_values: Tensor,
    means: Tensor,
    stds: Tensor,
    spec: OptimizationSpec,
    iteration: int,
    random_seed: int,
    method: AcquisitionMethod,
) -> list[SuggestionResult]:
    """Create SuggestionResult objects for multi-objective optimization.

    Args:
        candidates: Optimized candidate points
        acq_values: Acquisition function values
        means: Posterior mean predictions at candidates (shape: batch x n_objectives)
        stds: Posterior std predictions at candidates (shape: batch x n_objectives)
        spec: Optimization specification
        iteration: Current iteration number
        random_seed: Random seed for reproducibility
        method: Acquisition method used

    Returns:
        List of SuggestionResult objects
    """
    batch_size = candidates.shape[0]
    acq_name = method.value
    suggestions = []

    for i in range(batch_size):
        values = decode_categorical(candidates[i], spec)

        acq_val, avg_std, predicted_objectives, predicted_std_dict = (
            _build_multi_objective_provenance(means, stds, acq_values, i, spec)
        )
        confidence_level = _get_confidence_level(avg_std)
        explanation = _build_multi_objective_explanation(acq_name, acq_val)

        suggestion = SuggestionResult(
            parameter_values=values,
            iteration=iteration,
            batch_index=i,
            generation_method="bo",
            random_seed=random_seed,
            acquisition_value=acq_val,
            model_uncertainty=avg_std,
            acquisition_function=acq_name,
            model_type="ModelListGP (Gaussian Process)",
            model_version=iteration,
            confidence_level=confidence_level,
            explanation=explanation,
            predicted_objectives=predicted_objectives or None,
            predicted_std=predicted_std_dict or None,
        )
        suggestions.append(suggestion)

    return suggestions


def _generate_multi_objective_batch(
    ctx: GenerationContext,
) -> list[SuggestionResult]:
    """Generate suggestions for multi-objective optimization.

    Uses hypervolume improvement or scalarized multi-objective acquisition.

    Args:
        ctx: Generation context containing all parameters

    Returns:
        List of suggestion results
    """
    spec = ctx.spec
    train_x = ctx.train_x
    train_y = ctx.train_y
    bounds = ctx.bounds
    batch_size = ctx.batch_size

    # Get minimize mask for objectives
    minimize_mask = torch.tensor([obj.minimize for obj in spec.objectives], dtype=torch.bool)

    # ``log_transform`` is incompatible with a maximize objective on this
    # path: BoTorch's ``Log`` outcome transform fires *after* we negate the
    # column to enforce minimization, so a positive raw target becomes
    # negative and ``log`` is undefined. Surface this at the boundary
    # rather than letting BoTorch fail deep inside model fit.
    log_flags = [obj.log_transform for obj in spec.objectives]
    for idx, (flag, mini) in enumerate(zip(log_flags, minimize_mask.tolist(), strict=True)):
        if flag and not mini:
            raise ValueError(
                f"ObjectiveSpec.log_transform=True is only supported for "
                f"minimize=True objectives (objective[{idx}] "
                f"'{spec.objectives[idx].name}' is maximize). Flip the "
                "objective definition or pre-transform the data."
            )

    # Negate maximization objectives (BoTorch assumes minimization). Variance
    # is sign-invariant, so ``train_yvar`` flows through unchanged.
    train_y_bo = train_y.clone()
    train_y_bo[:, ~minimize_mask] = -train_y_bo[:, ~minimize_mask]

    # Create and fit model -- pass per-objective measurement variance when
    # every observation supplied it for every objective.
    model = create_and_fit_model(
        train_x,
        train_y_bo,
        bounds,
        use_input_warping=spec.use_input_warping,
        train_yvar=ctx.train_yvar,
        log_transform=log_flags,
    )

    # Get reference point (using STATIC strategy for backward compatibility;
    # DYNAMIC can be enabled via ReferencePointConfig when exposed in OptimizationSpec)
    ref_point_config = ReferencePointConfig(strategy=ReferencePointStrategy.STATIC)
    ref_point, _info = get_reference_point_dynamic(train_y_bo, minimize_mask, ref_point_config)

    # Determine acquisition method
    method = spec.acquisition_method
    if method == AcquisitionMethod.AUTO:
        method = AcquisitionMethod.HYPERVOLUME_IMPROVEMENT

    # Build native linear constraints for the optimizer
    ineq_constraints = None
    eq_constraints = None
    projection_constraints: list = []
    if spec.constraints:
        ineq_constraints, eq_constraints, projection_constraints = build_botorch_linear_constraints(
            spec
        )

    # Create acquisition function.  ``train_y_bo`` has maximization
    # columns pre-negated so every objective is in minimization form
    # (see bo_engine.types); the factory therefore receives an
    # all-True ``minimize_mask``.
    acqf = create_acquisition(
        model=model,
        ref_point=ref_point,
        train_x=train_x,
        train_y=train_y_bo,
        n_objectives=spec.n_objectives,
        minimize_mask=torch.ones(spec.n_objectives, dtype=torch.bool),
        method=method,
        constraints=None,
    )
    # Optimize with native constraints where possible
    candidates, acq_values = optimize_acquisition(
        acqf,
        bounds,
        batch_size,
        spec=spec,
        x_avoid=train_x,
        inequality_constraints=ineq_constraints or None,
        equality_constraints=eq_constraints or None,
        X_pending=ctx.pending_x,
    )

    # Post-hoc projection only for constraints that couldn't be handled natively
    if projection_constraints:
        candidates = _apply_constraints_to_samples(candidates, spec, bounds)

    # Get model predictions for provenance
    means, stds = _get_model_predictions(model, candidates)

    return _create_multi_objective_suggestions(
        candidates,
        acq_values,
        means,
        stds,
        spec,
        ctx.iteration,
        ctx.random_seed,
        method,
    )


def _get_confidence_level(uncertainty: float | None) -> str:
    """Determine confidence level from uncertainty."""
    if uncertainty is None:
        return "medium"
    if uncertainty < CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD:
        return "high"
    elif uncertainty < CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD:
        return "medium"
    else:
        return "low"


def _prepare_cost_data(
    observations: list[ObservationData], use_cost_aware: bool = False
) -> Tensor | None:
    """Extract cost data from observations.

    Args:
        observations: List of observations with optional cost field
        use_cost_aware: Whether cost-aware mode was requested (for warning)

    Returns:
        Tensor of costs if all observations have costs, else None
    """
    costs = []
    has_some_costs = False
    for obs in observations:
        if obs.cost is None:
            if has_some_costs and use_cost_aware:
                logger.warning(
                    "Cost-aware mode requested but %d/%d observations lack cost data. "
                    "Falling back to non-cost-aware optimization.",
                    sum(1 for o in observations if o.cost is None),
                    len(observations),
                )
            return None
        has_some_costs = True
        costs.append(obs.cost)
    return torch.tensor(costs, dtype=get_dtype(), device=get_device())


def _build_outcome_constraint_models(
    spec: OptimizationSpec,
    observations: list[ObservationData],
    train_x: Tensor,
    bounds: Tensor,
) -> list[tuple[Any, float]] | None:
    """Build constraint models for outcome constraints.

    For each outcome constraint, trains a GP to predict feasibility
    and returns (model, threshold) pairs for use in constrained acquisition.

    Args:
        spec: Optimization specification with outcome_constraints
        observations: Historical observations
        train_x: Training inputs (already encoded)
        bounds: Parameter bounds

    Returns:
        List of (constraint_model, threshold) tuples, or None if no constraints
    """
    if not spec.outcome_constraints:
        return None

    constraint_models = []
    for oc in spec.outcome_constraints:
        # Extract the objective values for this constraint
        obj_values = []
        for obs in observations:
            if oc.objective_name not in obs.objective_values:
                logger.warning(
                    "Outcome constraint on '%s' disabled: observation missing this objective. "
                    "All observations must include the constrained objective.",
                    oc.objective_name,
                )
                return None
            obj_values.append(obs.objective_values[oc.objective_name])

        obj_tensor = torch.tensor(obj_values, dtype=get_dtype(), device=get_device()).unsqueeze(-1)

        # Compute feasibility (binary: 1 if feasible, 0 if not)
        if oc.greater_than:
            feasible = (obj_tensor >= oc.threshold).double()
        else:
            feasible = (obj_tensor <= oc.threshold).double()

        # Train constraint GP on feasibility
        constraint_model = create_and_fit_single_task_model(
            train_x, feasible, bounds, use_input_warping=False
        )
        # Use per-constraint feasibility threshold from the spec (default 0.5)
        constraint_models.append((constraint_model, oc.feasibility_threshold))

    return constraint_models if constraint_models else None


def update_turbo_after_evaluation(
    turbo_state: TurboState,
    new_observations: list[ObservationData],
    spec: OptimizationSpec,
) -> TurboState:
    """Update TuRBO state after new observations are collected.

    Should be called after evaluating candidates from the last batch.
    Updates success/failure counters and adjusts trust region size.

    Args:
        turbo_state: Current TuRBO state
        new_observations: New observations from the latest batch
        spec: Optimization specification

    Returns:
        Updated TurboState
    """
    if not new_observations:
        return turbo_state

    # Extract raw objective values — update_turbo_state handles negation internally
    minimize = spec.objectives[0].minimize
    obj_name = spec.objectives[0].name
    y_values = [obs.objective_values[obj_name] for obs in new_observations]

    y_tensor = torch.tensor(y_values, dtype=get_dtype(), device=get_device())
    return update_turbo_state(turbo_state, y_tensor, minimize=minimize)


def _prepare_training_data(
    observations: list[ObservationData],
    spec: OptimizationSpec,
) -> tuple[Tensor, Tensor]:
    """Prepare training data from observations.

    Args:
        observations: List of ObservationData entities
        spec: Optimization specification

    Returns:
        Tuple of (train_x, train_y) tensors
    """
    x_list = []
    y_list = []

    for obs in observations:
        # Encode parameters
        x = encode_categorical(obs.parameter_values, spec)
        x_list.append(x)

        # Get objective values in spec order
        y = torch.tensor(
            [obs.objective_values[obj.name] for obj in spec.objectives],
            dtype=get_dtype(),
            device=get_device(),
        )
        y_list.append(y)

    train_x = torch.stack(x_list)
    train_y = torch.stack(y_list)

    return train_x, train_y


def _prepare_train_yvar(
    observations: list[ObservationData],
    spec: OptimizationSpec,
) -> Tensor | None:
    """Build the per-observation noise-variance tensor in objective order.

    Returns ``None`` when any observation is missing measurement uncertainty
    for any objective — partial coverage falls back to the trainable noise
    path (rather than imputing zeros, which would silently claim the
    uncovered points are noise-free). Returned tensor has shape
    ``(n_observations, n_objectives)`` and units of variance (stddev**2).
    """
    if not observations:
        return None
    objective_names = [obj.name for obj in spec.objectives]
    rows: list[list[float]] = []
    for obs in observations:
        unc = obs.measurement_uncertainty
        if unc is None:
            return None
        row: list[float] = []
        for name in objective_names:
            if name not in unc:
                return None
            stddev = float(unc[name])
            row.append(stddev * stddev)
        rows.append(row)
    return torch.tensor(rows, dtype=get_dtype(), device=get_device())
