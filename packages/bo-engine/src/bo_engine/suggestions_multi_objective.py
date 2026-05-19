"""Multi-objective acquisition pipeline.

Split from :mod:`bo_engine.suggestions` so the multi-objective GP fit /
hypervolume-improvement / Pareto-front provenance path lives in its own
module. :mod:`bo_engine.suggestions` calls
:func:`_generate_multi_objective_batch` from the unified
:func:`generate_next_batch` dispatcher when ``spec.n_objectives > 1``.
"""

from __future__ import annotations

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
    create_and_fit_model,
    post_fit_verification,
)
from bo_engine.reference_point import (
    ReferencePointConfig,
    ReferencePointStrategy,
    get_reference_point_dynamic,
)
from bo_engine.suggestions_common import (
    _extract_scalar_prediction,
    _get_confidence_level,
    _get_model_predictions,
)
from bo_engine.suggestions_outcome_constraints import (
    _build_outcome_constraint_models,
)
from bo_engine.transforms import (
    decode_categorical,
    get_categorical_dim_indices,
)
from bo_engine.types import (
    AcquisitionMethod,
    GenerationContext,
    OptimizationSpec,
    SuggestionResult,
)


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
    model_warnings: tuple[str, ...] = (),
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
        model_warnings: Non-fatal warnings produced during model fitting that
            should be threaded into each suggestion's provenance.

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
            model_warnings=model_warnings,
        )
        suggestions.append(suggestion)

    return suggestions


def _generate_multi_objective_batch(
    ctx: GenerationContext,
    *,
    noise_prior: GammaPrior | None = None,
) -> list[SuggestionResult]:
    """Generate suggestions for multi-objective optimization.

    Uses hypervolume improvement or scalarized multi-objective acquisition.

    Args:
        ctx: Generation context containing all parameters
        noise_prior: Optional GammaPrior override resolved by the caller;
            ``None`` falls through to the model factory's default prior.

    Returns:
        List of suggestion results
    """
    spec = ctx.spec
    train_x = ctx.train_x
    train_y = ctx.train_y
    bounds = ctx.bounds
    batch_size = ctx.batch_size
    observations = ctx.observations

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
            msg = (
                f"ObjectiveSpec.log_transform=True is only supported for "
                f"minimize=True objectives (objective[{idx}] "
                f"'{spec.objectives[idx].name}' is maximize). Flip the "
                "objective definition or pre-transform the data."
            )
            raise ValueError(msg)

    # ``auto_shift_for_log`` is a single-objective-only opt-in. The
    # multi-objective factory builds per-objective sub-models but doesn't
    # currently track per-objective shifts in suggestion provenance.
    # Refusing the combination at the boundary keeps the contract honest
    # rather than silently fitting half the campaign on shifted data.
    if spec.auto_shift_for_log and any(log_flags):
        msg = (
            "OptimizationSpec.auto_shift_for_log=True is only supported on "
            "single-objective campaigns. Drop the flag or pre-shift the "
            "non-positive observations before submitting the campaign."
        )
        raise ValueError(msg)

    # Negate maximization objectives (BoTorch assumes minimization). Variance
    # is sign-invariant, so ``train_yvar`` flows through unchanged.
    train_y_bo = train_y.clone()
    train_y_bo[:, ~minimize_mask] = -train_y_bo[:, ~minimize_mask]

    # Create and fit model -- pass per-objective measurement variance when
    # every observation supplied it for every objective.
    cat_dim_indices = get_categorical_dim_indices(spec) if spec.use_categorical_kernel else None
    model = create_and_fit_model(
        train_x,
        train_y_bo,
        bounds,
        use_input_warping=spec.use_input_warping,
        train_yvar=ctx.train_yvar,
        log_transform=log_flags,
        categorical_dim_indices=cat_dim_indices,
        noise_prior=noise_prior,
    )

    # Post-fit standardization audit (per-objective). Inspects the
    # recorded Standardize stddev on each sub-model (read-only) and
    # emits a warning naming any collapsed objective.
    model_warnings = post_fit_verification(
        model, objective_names=[obj.name for obj in spec.objectives]
    )

    # Outcome constraint models (constraints on OUTPUT space). Mirrors
    # the single-objective path so multi-objective campaigns don't
    # silently ignore ``outcome_constraints``. The constraint GPs are
    # bundled into the acquisition ModelListGP by the dispatcher and
    # constraint callables are wired to BoTorch's negative-feasible
    # convention; qLogNEHVI restricts hypervolume to the objective
    # channels via ``IdentityMCMultiOutputObjective``.
    outcome_constraints = None
    if spec.outcome_constraints and observations is not None:
        outcome_constraints = _build_outcome_constraint_models(spec, observations, train_x, bounds)

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
        outcome_constraint_models=outcome_constraints,
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
        model_warnings=tuple(model_warnings),
    )
