"""Single-objective acquisition pipeline.

Split from :mod:`bo_engine.suggestions` so the single-objective GP fit /
acquisition-optimize / provenance-build path lives in a dedicated module
alongside its TuRBO trust-region helpers. :mod:`bo_engine.suggestions`
calls :func:`_generate_single_objective_batch` from the unified
:func:`generate_next_batch` dispatcher when ``spec.n_objectives == 1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from gpytorch.priors import GammaPrior
from torch import Tensor

from bo_engine.acquisition import (
    create_acquisition,
    optimize_acquisition,
)
from bo_engine.constraints import build_botorch_linear_constraints
from bo_engine.initial_design import _apply_constraints_to_samples
from bo_engine.models import (
    create_and_fit_single_task_model,
    inspect_standardize_stdvs,
    post_fit_verification,
)
from bo_engine.suggestions_common import (
    _extract_scalar_prediction,
    _get_confidence_level,
    _get_model_predictions,
    _normalize_uncertainty,
)
from bo_engine.suggestions_outcome_constraints import (
    _build_outcome_constraint_models,
)
from bo_engine.suggestions_training import categorical_blocks_for_model
from bo_engine.transforms import (
    decode_categorical,
)
from bo_engine.turbo import (
    TurboState,
    assert_unit_scale_targets,
    create_turbo_state,
    get_turbo_bounds,
    should_use_turbo,
)
from bo_engine.types import (
    AcquisitionMethod,
    GenerationContext,
    OptimizationSpec,
    SuggestionResult,
)

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP


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

    A state carrying ``restart_triggered=True`` (the trust region contracted
    below ``length_min``) is discarded and rebuilt from scratch, so a
    triggered restart actually restarts the trust region as in Eriksson et
    al. (NeurIPS 2019) — fresh length and counters, incumbent re-anchored to
    the current data. Keeping the stale state would silently revert to
    global search while provenance keeps reporting ``"turbo"``, and a later
    success streak could re-expand the collapsed sub-minimum region.

    Args:
        spec: Optimization specification
        train_y_bo: Training outputs in maximization form (higher = better;
            see :mod:`bo_engine.types`), matching TuRBO's internal convention
        batch_size: Number of suggestions per batch
        turbo_state: Existing TuRBO state if any

    Returns:
        Initialized TuRBO state or None if not using TuRBO
    """
    use_turbo = spec.use_turbo or (turbo_state is not None) or should_use_turbo(spec.n_parameters)
    if not use_turbo:
        return None
    if turbo_state is not None and turbo_state.restart_triggered:
        turbo_state = None
    if turbo_state is not None:
        return turbo_state

    assert_unit_scale_targets(train_y_bo)
    best_y = train_y_bo.max().item()
    config = spec.turbo_config
    if config is None:
        return create_turbo_state(
            dim=spec.n_parameters,
            batch_size=batch_size,
            initial_best_value=best_y,
        )
    return create_turbo_state(
        dim=spec.n_parameters,
        batch_size=batch_size,
        initial_best_value=best_y,
        initial_length=config.initial_length,
        length_min=config.length_min,
        length_max=config.length_max,
        success_tolerance=config.success_tolerance,
        failure_tolerance=config.failure_tolerance,
    )


def _compute_turbo_bounds(
    turbo_state: TurboState | None,
    train_x: Tensor,
    train_y_bo: Tensor,
    model: SingleTaskGP,
    bounds: Tensor,
) -> tuple[Tensor, str]:
    """Compute trust region bounds for TuRBO.

    :func:`bo_engine.turbo.get_turbo_bounds` operates in the normalized
    [0, 1] cube (trust-region lengths are defined there), so the raw-scale
    ``train_x`` is normalized before centering and the resulting trust
    region is mapped back to raw parameter bounds for the optimizer.

    Args:
        turbo_state: Current TuRBO state
        train_x: Training inputs on the raw parameter scale
        train_y_bo: Training outputs in maximization form (higher = better),
            so the trust region centers on the best observed point
        model: Fitted single-objective GP model
        bounds: Original parameter bounds

    Returns:
        Tuple of (optimization_bounds, turbo_info_string)
    """
    if turbo_state is None or turbo_state.restart_triggered:
        return bounds.clone(), ""

    train_x_norm = (train_x - bounds[0]) / (bounds[1] - bounds[0])
    tr_lb, tr_ub = get_turbo_bounds(turbo_state, train_x_norm, train_y_bo, model)
    opt_bounds = torch.stack(
        [
            bounds[0] + tr_lb * (bounds[1] - bounds[0]),
            bounds[0] + tr_ub * (bounds[1] - bounds[0]),
        ]
    )
    turbo_info = f" TuRBO length={turbo_state.length:.4f}."
    return opt_bounds, turbo_info


def _build_single_objective_provenance(
    means: Tensor,
    stds: Tensor,
    acq_values: Tensor,
    index: int,
    obj_name: str,
    minimize: bool,
    auto_shift: float = 0.0,
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
        auto_shift: Additive shift applied at fit time when the campaign
            opted into ``auto_shift_for_log`` on a log-transformed
            objective (see ``bo_engine.models.compute_log_auto_shift``).
            The GP is fit on ``train_y + shift`` so its posterior mean is
            on the shifted scale; we subtract the shift before populating
            ``predicted_objectives`` so the user-facing value matches the
            raw observation scale.

    Returns:
        Tuple of (acq_val, std_val, confidence_level_unused,
                  predicted_objectives, predicted_std)
    """
    std_val = _extract_scalar_prediction(stds, index)
    acq_val = _extract_scalar_prediction(acq_values, index)

    # Minimize objectives are negated into maximization form before the GP
    # fit (see bo_engine.types) — un-negate the posterior mean for storage.
    pred_mean = _extract_scalar_prediction(means, index)
    if pred_mean is not None and minimize:
        pred_mean = -pred_mean  # undo the maximization-form negation
    if pred_mean is not None and auto_shift:
        # Subtract the shift so callers see predictions on the raw scale.
        pred_mean = pred_mean - auto_shift

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
    model_warnings: tuple[str, ...] = (),
    auto_shift: float = 0.0,
    confidence_scale: float = 1.0,
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
        model_warnings: Non-fatal warnings produced during model fitting
            that should be threaded into each suggestion's provenance.
        auto_shift: Offset applied to ``train_y`` before fitting (e.g. for
            log-transformed objectives); subtracted from posterior means
            so predictions are reported on the raw scale.
        confidence_scale: The model's ``Standardize.stdvs`` for the
            objective; the raw-scale candidate std is divided by it before
            the confidence-level thresholds are applied so the verdict is
            invariant to the objective's numeric scale.

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
            _build_single_objective_provenance(
                means, stds, acq_values, i, obj_name, minimize, auto_shift=auto_shift
            )
        )
        confidence_level = _get_confidence_level(_normalize_uncertainty(std_val, confidence_scale))

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
            model_warnings=model_warnings,
        )
        suggestions.append(suggestion)

    return suggestions


def _generate_single_objective_batch(
    ctx: GenerationContext,
    *,
    noise_prior: GammaPrior | None = None,
) -> tuple[list[SuggestionResult], TurboState | None]:
    """Generate suggestions for single-objective optimization.

    Uses noisy EI (or EI) acquisition function. Supports TuRBO for
    high-dimensional problems, outcome constraints, and cost-aware optimization.

    Args:
        ctx: Generation context containing all parameters
        noise_prior: Optional GammaPrior override resolved by the caller;
            ``None`` falls through to the model factory's default prior.

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
        # ``log_transform`` is part of the public spec contract only for
        # minimize objectives. The engine's maximization-form convention
        # handles the minimize case via the ``Negate`` outcome-transform
        # stage (see ``bo_engine.models``); the maximize combination stays
        # rejected at the boundary so the supported surface is unchanged.
        msg = (
            "ObjectiveSpec.log_transform=True is only supported for "
            "minimize=True objectives. For a maximize objective with a "
            "multi-decade target, either flip the objective definition "
            "(minimize the negative log) or pre-transform the data."
        )
        raise ValueError(msg)

    # Negate if minimizing (BoTorch's acquisition stack assumes
    # maximization; see bo_engine.types for the canonical convention).
    # Variance is sign-invariant -- ``Var(-Y) == Var(Y)`` -- so
    # ``train_yvar`` flows through unchanged.
    train_y_bo = -train_y if minimize else train_y.clone()

    # Create and fit model. When every observation has measurement
    # uncertainty for this objective, route through the
    # ``FixedNoiseGaussianLikelihood`` path so the GP trusts the user's
    # known noise instead of re-estimating it from MLL. ``target_negated``
    # tells the factory that a minimize objective arrives negated, which
    # the ``Log`` outcome stage must undo (and redo on the posterior).
    cat_blocks = categorical_blocks_for_model(spec)
    model = create_and_fit_single_task_model(
        train_x,
        train_y_bo,
        bounds,
        use_input_warping=spec.use_input_warping,
        train_yvar=train_yvar,
        log_transform=log_transform,
        categorical_blocks=cat_blocks,
        auto_shift_for_log=spec.auto_shift_for_log,
        noise_prior=noise_prior,
        target_negated=minimize,
    )

    # Post-fit standardization audit: read-only inspection of each
    # sub-model's recorded ``Standardize.stdvs`` (mutating it would break
    # the forward / inverse transform round-trip) plus the unit-variance
    # invariant check. Failures surface as batch-level warnings to the
    # caller.
    model_warnings = post_fit_verification(
        model, objective_names=[obj.name for obj in spec.objectives]
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
    # canonical maximization form (see bo_engine.types) because we
    # negated above when ``minimize``; the factory therefore always
    # receives ``maximize=True``.
    acqf = create_acquisition(
        model=model,
        ref_point=None,
        train_x=train_x,
        train_y=train_y_bo,
        n_objectives=1,
        maximize=True,
        method=method,
        constraints=None,
        outcome_constraint_models=outcome_constraints,
        cost_model=cost_model,
        beta=spec.acquisition_beta,
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
        random_seed=ctx.random_seed,
        domain_bounds=bounds,
    )

    # Post-hoc projection only for constraints that couldn't be handled natively
    if projection_constraints:
        candidates = _apply_constraints_to_samples(candidates, spec, bounds)

    # Get model predictions and create suggestions
    means, stds = _get_model_predictions(model, candidates)
    # Pull the recorded auto-shift off the model (only populated when the
    # caller opted into ``auto_shift_for_log`` AND the data had non-positive
    # observations). The provenance helper subtracts it so user-facing
    # ``predicted_objectives`` are reported on the raw scale.
    auto_shift = float(getattr(model, "_auto_shift_for_log", 0.0))
    # The posterior std is reported on the raw objective scale (Standardize
    # un-transforms it); normalize the confidence threshold by the model's
    # recorded Standardize.stdvs so the verdict is scale-invariant.
    confidence_scale = inspect_standardize_stdvs(model)[0]
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
        model_warnings=tuple(model_warnings),
        auto_shift=auto_shift,
        confidence_scale=confidence_scale,
    )

    return suggestions, turbo_state
