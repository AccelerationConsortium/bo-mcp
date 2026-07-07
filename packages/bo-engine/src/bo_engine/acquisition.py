"""Acquisition function creation and optimization.

Supports both single-objective (noisy EI, EI) and multi-objective
(hypervolume improvement, scalarized multi-objective) acquisition
functions.

All public factories in this module require an explicit ``maximize`` (or
``maximize_mask``) argument and expect ``train_y`` / ``best_f`` /
``ref_point`` to be in *maximization form* (higher = better) — BoTorch's
native convention for the qLog* acquisition family.  See the canonical
sign-convention documentation in :mod:`bo_engine.types` for details and
for caller responsibilities.

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
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.optim.optimize import optimize_acqf_discrete, optimize_acqf_mixed
from torch import Tensor

from bo_engine.constants import (
    ACQF_LBFGS_BATCH_LIMIT,
    ACQF_LBFGS_MAXITER,
    COST_AWARE_MIN_EXPECTED_COST,
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


def _assert_maximization_form(maximize: bool) -> None:
    """Guard against silently passing minimization-form data into a factory.

    The canonical internal convention is maximization form (higher is
    better) — BoTorch's native convention; callers must negate any
    minimization objectives before invoking a factory.  See
    :mod:`bo_engine.types` for the convention.
    """
    if not maximize:
        msg = (
            "bo_engine acquisition factories operate in maximization form "
            "(higher = better), matching BoTorch's qLog* family.  Negate "
            "minimization objectives at the call site and pass "
            "``maximize=True``.  See the sign-convention note in "
            "bo_engine.types for details."
        )
        raise ValueError(msg)


def _assert_maximization_mask(maximize_mask: Tensor) -> None:
    """Multi-objective counterpart of :func:`_assert_maximization_form`."""
    if not bool(maximize_mask.all()):
        msg = (
            "bo_engine multi-objective factories operate in maximization "
            "form (every objective treated as higher = better).  Negate any "
            "minimization columns at the call site before building the "
            "acquisition function.  See bo_engine.types for details."
        )
        raise ValueError(msg)


def create_single_objective_acquisition(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    *,
    maximize: bool,
    best_f: float | None = None,
    use_noisy: bool = True,
    constraints: list | None = None,
) -> AcquisitionFunction:
    """Create acquisition function for single-objective optimization.

    ``train_y`` and ``best_f`` must be supplied in the canonical
    maximization form (higher = better).  See :mod:`bo_engine.types` for the
    sign convention and caller responsibilities.

    When ``model`` is a :class:`ModelListGP` (objective at index 0 plus one
    constraint GP per outcome constraint), we wrap a
    :class:`~botorch.acquisition.objective.GenericMCObjective` that
    extracts output channel 0 so qLogNEI's baseline / improvement logic
    keeps reading the objective. The ``constraints`` callables are
    expected to index into the constraint channels (``samples[..., 1+i]``)
    so BoTorch's feasibility weighting reads the constraint GP's
    posterior rather than the objective GP's samples — that was the bug
    behind the historical outcome-constraint path.

    Args:
        model: Fitted ``SingleTaskGP`` *or* :class:`ModelListGP` whose
            first output is the objective. Pass a model list when outcome
            constraints are active.
        train_x: Training inputs for baseline sampling
        train_y: Training outputs in maximization form
        maximize: Direction of the data handed to the factory.  Required
            keyword so the caller cannot silently pass the wrong sign
            convention.  Only ``True`` is accepted at the construction
            boundary because the internal engine convention is
            maximization form — pass ``maximize=True`` after negating any
            minimization objectives at the call site.
        best_f: Best observed maximization-form value.  If ``None`` and
            ``use_noisy=False``, computed as ``train_y.max()``.
        use_noisy: If True, use noisy EI (handles noise), else EI
        constraints: Optional list of constraint callables. Each callable
            receives ``samples`` of shape ``(..., n_outputs)`` when
            ``model`` is a ``ModelListGP``. BoTorch's MC feasibility
            convention is **negative return → feasible** (see
            :func:`botorch.utils.objective.compute_smoothed_feasibility_indicator`);
            the callable produced by :func:`_make_outcome_constraint_callable`
            already follows that convention.

    Returns:
        Noisy EI or EI acquisition function
    """
    _assert_maximization_form(maximize)
    train_x, train_y = ensure_device(train_x, train_y)

    if use_noisy:
        # Noisy EI handles noisy observations - recommended default
        acqf_kwargs: dict[str, Any] = {
            "model": model,
            "X_baseline": train_x,
            "prune_baseline": True,
            "cache_root": False,
        }
        if isinstance(model, ModelListGP):
            # Multi-output model: the objective is channel 0 and constraint
            # GPs occupy channels 1..k. Without an explicit objective
            # callable BoTorch tries to reduce all channels into a scalar,
            # which would silently mix constraint outputs into the EI
            # improvement signal.
            # ``GenericMCObjective`` calls our callable as
            # ``objective(samples, X=X)`` — the parameter must be named
            # ``X`` (BoTorch's convention) even though we don't read it.
            acqf_kwargs["objective"] = GenericMCObjective(
                lambda samples, X=None: samples[..., 0]  # noqa: N803, ARG005
            )
        if constraints is not None and len(constraints) > 0:
            acqf_kwargs["constraints"] = constraints
        return qLogNoisyExpectedImprovement(**acqf_kwargs)

    # EI for noiseless observations - requires best_f.
    # The analytic ``qLogExpectedImprovement`` path does not support
    # multi-output models without an explicit objective / posterior
    # transform; reject ``ModelListGP`` here with a clear error rather
    # than letting BoTorch raise ``UnsupportedError`` deep inside its
    # acquisition optimizer. The internal constrained dispatch upgrades
    # to ``qLogNoisyExpectedImprovement`` before reaching this branch
    # (see ``_create_single_objective_dispatch``), so this guard fires
    # only when a direct caller wires a multi-output model into the
    # analytic path.
    if isinstance(model, ModelListGP):
        msg = (
            "Analytic EXPECTED_IMPROVEMENT (qLogExpectedImprovement) does "
            "not support ModelListGP / multi-output models without an "
            "explicit objective. Either pass a single-output SingleTaskGP "
            "or switch to use_noisy=True (NOISY_EI) which accepts a "
            "GenericMCObjective channel selector."
        )
        raise TypeError(msg)
    if best_f is None:
        # train_y is in maximization form (higher = better); max() is best.
        best_f = train_y.max().item()
    return qLogExpectedImprovement(model=model, best_f=best_f)


def create_multi_objective_acquisition(
    model: ModelListGP,
    ref_point: Tensor,
    train_x: Tensor,
    train_y: Tensor,
    *,
    maximize_mask: Tensor,
    method: AcquisitionMethod = AcquisitionMethod.HYPERVOLUME_IMPROVEMENT,
    constraints: list | None = None,
) -> AcquisitionFunction:
    """Create acquisition function for multi-objective optimization.

    ``train_y`` and ``ref_point`` must be supplied in the canonical
    maximization form — every objective treated as higher = better — with
    minimization columns pre-negated at the call site.  See
    :mod:`bo_engine.types` for the convention.

    **Outcome-constraint bundling.** When the caller bundles outcome-
    constraint GPs into ``model`` (so ``len(model.models) > n_objectives``)
    the first ``n_objectives`` output channels are the optimization
    targets and the remaining channels are constraint GPs. We pass an
    :class:`IdentityMCMultiOutputObjective` restricted to the objective
    channels so qLogNEHVI / qLogNParEGO compute hypervolume / scalarized
    improvement over the objectives only; the ``constraints`` callables
    are expected to index into the trailing constraint channels.

    Args:
        model: Fitted ModelListGP (trained on maximization-form y). May
            contain extra trailing output channels for outcome-constraint
            GPs — see the note above.
        ref_point: Reference point in maximization form (below every
            observed point)
        train_x: Training inputs for sampling baseline
        train_y: Training outputs in maximization form, shape
            ``(n_samples, n_objectives)``. The column count determines
            the objective channel set.
        maximize_mask: Boolean tensor of shape ``(n_objectives,)`` recording
            the direction of the data handed to the factory.  Required
            keyword so the call site cannot silently pass mismatched signs;
            must be all-True because the engine operates in maximization
            form.
        method: Acquisition method (HYPERVOLUME_IMPROVEMENT or SCALARIZED_MULTI_OBJ)
        constraints: Optional list of constraint callables

    Returns:
        Multi-objective acquisition function
    """
    _assert_maximization_mask(maximize_mask)
    train_x, train_y, ref_point = ensure_device(train_x, train_y, ref_point)

    n_objectives = int(maximize_mask.numel())
    objective_outcomes = list(range(n_objectives))
    has_extra_outputs = isinstance(model, ModelListGP) and len(model.models) > n_objectives
    mo_objective: IdentityMCMultiOutputObjective | None = (
        IdentityMCMultiOutputObjective(outcomes=objective_outcomes) if has_extra_outputs else None
    )

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
        if mo_objective is not None:
            acqf_kwargs["objective"] = mo_objective
        if constraints is not None and len(constraints) > 0:
            acqf_kwargs["constraints"] = constraints

        return qLogNParEGO(**acqf_kwargs)

    # Default: hypervolume improvement (qLogNEHVI)
    acqf_kwargs: dict = {
        "model": model,
        "ref_point": ref_point.tolist(),
        "X_baseline": train_x,
        "prune_baseline": True,
        "cache_root": False,
    }
    if mo_objective is not None:
        acqf_kwargs["objective"] = mo_objective
    if constraints is not None and len(constraints) > 0:
        acqf_kwargs["constraints"] = constraints

    return qLogNoisyExpectedHypervolumeImprovement(**acqf_kwargs)


def create_acquisition_from_config(config: AcquisitionConfig) -> AcquisitionFunction:
    """Create acquisition function from configuration object.

    This is the preferred way to create acquisition functions as it reduces
    parameter count and improves code clarity.

    Args:
        config: Acquisition configuration containing all parameters.  The
            caller must populate ``config.maximize`` /
            ``config.maximize_mask`` consistent with the canonical
            maximization-form convention (see :mod:`bo_engine.types`).

    Returns:
        Acquisition function appropriate for the problem
    """
    return create_acquisition(
        model=config.model,
        ref_point=config.ref_point,
        train_x=config.train_x,
        train_y=config.train_y,
        n_objectives=config.n_objectives,
        maximize=config.maximize,
        maximize_mask=config.maximize_mask,
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
    maximize: bool | None = None,
    maximize_mask: Tensor | None = None,
    method: AcquisitionMethod = AcquisitionMethod.AUTO,
    constraints: list | None = None,
    outcome_constraint_models: list[tuple[SingleTaskGP, float]] | None = None,
    cost_model: SingleTaskGP | None = None,
) -> AcquisitionFunction:
    """Create appropriate acquisition function based on problem type.

    Automatically selects between single-objective and multi-objective
    acquisition functions based on n_objectives.

    ``train_y`` (and ``ref_point`` for the multi-objective path) must be in
    the canonical maximization form — see :mod:`bo_engine.types`.  Callers
    **must** supply ``maximize`` for single-objective problems and
    ``maximize_mask`` for multi-objective problems so the direction is
    explicit at the construction boundary.

    Note: Consider using create_acquisition_from_config() with an
    AcquisitionConfig object for cleaner code with fewer parameters.

    Args:
        model: Fitted GP model (SingleTaskGP or ModelListGP)
        ref_point: Reference point for hypervolume (multi-objective only)
        train_x: Training inputs for sampling baseline
        train_y: Training outputs in maximization form
        n_objectives: Number of objectives (1 = single, 2+ = multi)
        maximize: Direction of the data handed to the factory for
            single-objective problems.  Required when ``n_objectives == 1``.
        maximize_mask: Boolean tensor recording the per-objective direction
            of the data handed to the factory.  Required when
            ``n_objectives >= 2``.
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
        if maximize is None:
            msg = (
                "create_acquisition requires ``maximize`` for "
                "single-objective problems; see bo_engine.types for the "
                "sign convention."
            )
            raise ValueError(msg)
        return _create_single_objective_dispatch(
            model=model,
            train_x=train_x,
            train_y=train_y,
            maximize=maximize,
            method=method,
            constraints=constraints,
            outcome_constraint_models=outcome_constraint_models,
            cost_model=cost_model,
        )

    if maximize_mask is None:
        msg = (
            "create_acquisition requires ``maximize_mask`` for "
            "multi-objective problems; see bo_engine.types for the sign "
            "convention."
        )
        raise ValueError(msg)
    return _create_multi_objective_dispatch(
        model=model,
        ref_point=ref_point,
        train_x=train_x,
        train_y=train_y,
        maximize_mask=maximize_mask,
        method=method,
        constraints=constraints,
        outcome_constraint_models=outcome_constraint_models,
    )


def _create_single_objective_dispatch(
    model: ModelListGP | SingleTaskGP,
    train_x: Tensor,
    train_y: Tensor,
    *,
    maximize: bool,
    method: AcquisitionMethod,
    constraints: list | None,
    outcome_constraint_models: list[tuple[SingleTaskGP, float]] | None,
    cost_model: SingleTaskGP | None,
) -> AcquisitionFunction:
    """Dispatch single-objective acquisition function creation.

    Handles model extraction from ModelListGP, cost-aware EIpu, and
    standard EI/noisy-EI with optional outcome constraints.

    **Outcome-constraint wiring.** When ``outcome_constraint_models`` is
    non-empty we bundle the objective GP plus every constraint GP into a
    single :class:`ModelListGP` and rewrite the per-constraint callable
    to index into the corresponding channel of the bundled posterior.
    This is the bug-fix to the legacy path: the previous callable
    ``samples - threshold`` operated on the *objective* model's samples
    because that was the only model passed to BoTorch, so the constraint
    GP we fit was discarded. With the bundle the callable reads the
    constraint GP's posterior directly and BoTorch's feasibility
    weighting becomes meaningful (Gardner et al. ICML 2014).

    Args:
        model: Fitted GP model (SingleTaskGP or single-model ModelListGP)
        train_x: Training inputs
        train_y: Training outputs in maximization form
        maximize: Direction of the data handed to the dispatch (see
            :mod:`bo_engine.types`)
        method: Acquisition method
        constraints: Optional constraint callables
        outcome_constraint_models: Optional (model, threshold) pairs
        cost_model: Optional cost model for EIpu

    Returns:
        Single-objective acquisition function
    """
    objective_gp: SingleTaskGP
    if isinstance(model, SingleTaskGP):
        objective_gp = model
    elif isinstance(model, ModelListGP) and len(model.models) == 1:
        objective_gp = cast(SingleTaskGP, model.models[0])
    else:
        msg = "Single-objective requires SingleTaskGP model"
        raise ValueError(msg)

    if method == AcquisitionMethod.COST_WEIGHTED_EI and cost_model is None:
        logger.warning(
            "COST_WEIGHTED_EI requested but no cost model available. "
            "Falling back to standard Noisy Expected Improvement."
        )

    acq_model, all_constraints = _bundle_outcome_constraint_models(
        objective_gp, constraints, outcome_constraint_models
    )

    if method == AcquisitionMethod.COST_WEIGHTED_EI and cost_model is not None:
        # The cost-aware path shares the constrained qLogNEI core with the
        # standard path; ``EIpuAcquisition`` only adds the log expected-cost
        # term on top (see ``create_cost_aware_acquisition``).
        return create_cost_aware_acquisition(
            model=acq_model,
            train_x=train_x,
            train_y=train_y,
            cost_model=cost_model,
            maximize=maximize,
            constraints=all_constraints if all_constraints else None,
        )

    use_noisy = method != AcquisitionMethod.EXPECTED_IMPROVEMENT
    if outcome_constraint_models and not use_noisy:
        # ``qLogExpectedImprovement`` (the analytic EI path) does not
        # support multi-output models without an explicit objective /
        # posterior transform, so a constrained run with
        # ``EXPECTED_IMPROVEMENT`` would raise ``UnsupportedError`` deep
        # inside BoTorch. The MC sibling ``qLogNoisyExpectedImprovement``
        # accepts ``constraints=`` and a ``GenericMCObjective`` channel
        # selector; we transparently upgrade and log a warning so the
        # caller knows the analytic path was unavailable.
        logger.warning(
            "AcquisitionMethod.EXPECTED_IMPROVEMENT does not support "
            "outcome constraints. Routing through NOISY_EI so BoTorch "
            "can apply the feasibility weighting; set "
            "acquisition_method=NOISY_EI to suppress this warning."
        )
        use_noisy = True
    return create_single_objective_acquisition(
        model=acq_model,
        train_x=train_x,
        train_y=train_y,
        maximize=maximize,
        use_noisy=use_noisy,
        constraints=all_constraints if all_constraints else None,
    )


def _bundle_outcome_constraint_models(
    objective_gp: SingleTaskGP,
    constraints: list | None,
    outcome_constraint_models: list[tuple[SingleTaskGP, float]] | None,
) -> tuple[SingleTaskGP | ModelListGP, list]:
    """Bundle the objective GP with outcome-constraint GPs for MC feasibility.

    Returns the acquisition model together with the combined list of
    constraint callables. Without outcome constraints the objective GP and
    the caller-supplied callables pass through untouched. With them, the
    objective GP (channel 0) and every constraint GP (channels 1..k) are
    wrapped in a single :class:`ModelListGP` and one channel-indexing
    callable per constraint is appended (see
    :func:`_make_outcome_constraint_callable`), so BoTorch's feasibility
    weighting reads each constraint GP's posterior rather than the
    objective's samples (Gardner et al. ICML 2014).
    """
    all_constraints = list(constraints) if constraints else []
    if not outcome_constraint_models:
        return objective_gp, all_constraints
    constraint_gps = [m for m, _ in outcome_constraint_models]
    acq_model = ModelListGP(objective_gp, *constraint_gps)
    for idx, (_constraint_gp, threshold) in enumerate(outcome_constraint_models):
        all_constraints.append(
            _make_outcome_constraint_callable(
                output_index=idx + 1,
                threshold=threshold,
            )
        )
    return acq_model, all_constraints


def _create_multi_objective_dispatch(
    model: ModelListGP | SingleTaskGP,
    ref_point: Tensor | None,
    train_x: Tensor,
    train_y: Tensor,
    *,
    maximize_mask: Tensor,
    method: AcquisitionMethod,
    constraints: list | None,
    outcome_constraint_models: list[tuple[SingleTaskGP, float]] | None = None,
) -> AcquisitionFunction:
    """Dispatch multi-objective acquisition function creation.

    Validates that a reference point is provided, then delegates to
    create_multi_objective_acquisition.

    **Outcome-constraint wiring.** Mirrors the single-objective dispatch:
    when ``outcome_constraint_models`` is non-empty we bundle the
    per-objective ModelListGP with the constraint GPs into a single
    composite ModelListGP whose first ``n_objectives`` output channels
    are the targets and whose trailing channels are the constraint GPs.
    The per-constraint callable returned by
    :func:`_make_outcome_constraint_callable` indexes into the matching
    trailing channel; :func:`create_multi_objective_acquisition` then
    wraps an ``IdentityMCMultiOutputObjective`` restricted to the
    objective channels so qLogNEHVI computes hypervolume only over the
    targets, with BoTorch's feasibility weighting applied via the
    constraint callables (Gardner et al. ICML 2014 — negative output =
    feasible).

    Args:
        model: Fitted ModelListGP whose outputs match the objectives
            declared in the spec.
        ref_point: Reference point for hypervolume computation, in
            maximization form
        train_x: Training inputs
        train_y: Training outputs in maximization form
        maximize_mask: Boolean tensor of shape ``(n_objectives,)`` recording
            the per-objective direction of the data handed to the dispatch
            (see :mod:`bo_engine.types`).
        method: Acquisition method
        constraints: Optional constraint callables passed through
            unchanged (e.g. native acquisition constraints).
        outcome_constraint_models: Optional list of ``(constraint_gp,
            threshold)`` pairs; when non-empty the model is extended and
            extra callables are appended for the constraint channels.

    Returns:
        Multi-objective acquisition function
    """
    if ref_point is None:
        msg = "Reference point required for multi-objective optimization"
        raise ValueError(msg)

    objective_model: ModelListGP
    if isinstance(model, ModelListGP):
        objective_model = model
    else:
        # Single-task model in the multi-objective path is a programming
        # error upstream; surface it loudly here.
        msg = "Multi-objective requires ModelListGP model"
        raise TypeError(msg)

    acq_model: ModelListGP = objective_model
    all_constraints = list(constraints) if constraints else []
    if outcome_constraint_models:
        n_objectives = int(maximize_mask.numel())
        constraint_gps = [m for m, _ in outcome_constraint_models]
        # ``ModelListGP.models`` is a ``torch.nn.ModuleList`` of fitted GPs;
        # unpacking it together with the constraint GPs yields a single
        # ``ModelListGP`` whose first ``n_objectives`` outputs are the
        # objectives and whose trailing outputs are the constraint GPs.
        acq_model = ModelListGP(*objective_model.models, *constraint_gps)  # ty: ignore[invalid-argument-type]
        for idx, (_constraint_gp, threshold) in enumerate(outcome_constraint_models):
            all_constraints.append(
                _make_outcome_constraint_callable(
                    output_index=n_objectives + idx,
                    threshold=threshold,
                )
            )

    return create_multi_objective_acquisition(
        model=acq_model,  # type: ignore[arg-type]
        ref_point=ref_point,
        train_x=train_x,
        train_y=train_y,
        maximize_mask=maximize_mask,
        method=method,
        constraints=all_constraints if all_constraints else None,
    )


def _make_outcome_constraint_callable(
    *,
    output_index: int,
    threshold: float,
) -> Callable[[Tensor], Tensor]:
    """Create a constraint callable for outcome constraints.

    The callable is consumed by BoTorch's
    :func:`~botorch.utils.objective.compute_smoothed_feasibility_indicator`
    via ``qLogNoisyExpectedImprovement.constraints`` (and the multi-
    objective sibling). BoTorch's documented convention is that
    **negative constraint values mean feasible** — the sigmoid
    feasibility weight saturates toward 1 when the callable returns a
    large negative number. We therefore return ``threshold - samples``,
    which is negative exactly when ``samples > threshold``.

    Sign-convention bookkeeping for the continuous outcome-constraint
    path is done at fit time by ``_fit_outcome_constraint_model``: a
    ``<=`` constraint trains on the negated objective values with a
    matching negated threshold, so the same ``threshold - samples``
    formula remains "negative = feasible" regardless of direction. The
    legacy binary path interprets ``samples`` as ``P(feasible)`` and the
    threshold as the per-constraint feasibility cut-off (default 0.5);
    again ``threshold - samples`` is negative when ``P(feasible)`` exceeds
    the cut-off.

    Args:
        output_index: Channel index in the ``ModelListGP`` posterior that
            holds the constraint GP's predictions (objective sits at 0,
            constraints at 1..k).
        threshold: Signed threshold (already encoded for the constraint
            direction by the fitting helper).

    Returns:
        Callable that returns constraint satisfaction in BoTorch's
        signed-feasibility convention (negative = satisfied).
    """

    def constraint_callable(samples: Tensor) -> Tensor:
        """Constraint function: negative means feasible (BoTorch convention).

        ``samples`` has shape ``(..., n_outputs)`` because the underlying
        acquisition model is a ``ModelListGP``; we index into the
        constraint channel rather than reducing over outputs.
        """
        return threshold - samples[..., output_index]

    return constraint_callable


def create_cost_aware_acquisition(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    cost_model: SingleTaskGP,
    *,
    maximize: bool,
    constraints: list | None = None,
) -> AcquisitionFunction:
    """Create the cost-aware EIpu (Expected Improvement per Unit cost) acquisition.

    Builds the same Monte-Carlo improvement core the standard
    single-objective path uses (``qLogNoisyExpectedImprovement``, including
    any outcome-constraint feasibility weighting carried by ``model`` /
    ``constraints``) and wraps it in :class:`EIpuAcquisition`, which
    subtracts the log expected cost. Maximizing the result is equivalent to
    maximizing ``qNEI(x) / E[cost(x)]`` — Snoek et al. (2012)'s
    "EI per second" generalized to the noisy, batched setting.

    ``train_y`` must be in the canonical maximization form; see
    :mod:`bo_engine.types`.

    Args:
        model: Fitted objective model, or a ``ModelListGP`` bundle whose
            channel 0 is the objective and channels 1..k are outcome
            constraint GPs (see :func:`_bundle_outcome_constraint_models`)
        train_x: Training inputs for the noisy-EI baseline
        train_y: Training outputs in maximization form
        cost_model: Fitted cost model
        maximize: Direction of the data handed to the factory (required
            keyword so the caller cannot silently mismatch the sign
            convention)
        constraints: Optional constraint callables consumed by the inner
            qLogNEI's feasibility weighting (negative return = feasible)

    Returns:
        EIpuAcquisition function
    """
    _assert_maximization_form(maximize)
    improvement = cast(
        qLogNoisyExpectedImprovement,
        create_single_objective_acquisition(
            model=model,
            train_x=train_x,
            train_y=train_y,
            maximize=maximize,
            use_noisy=True,
            constraints=constraints,
        ),
    )
    return EIpuAcquisition(improvement=improvement, cost_model=cost_model)


def _positive_cost_objective() -> GenericMCObjective:
    """Cost objective that squeezes the single output and floors it positive.

    ``InverseCostWeightedUtility`` requires strictly positive costs (here it
    takes their logarithm); a GP cost posterior can dip to (near-)zero or
    slightly negative in extrapolation, so we clamp to
    ``COST_AWARE_MIN_EXPECTED_COST`` before the weighting.
    """
    return GenericMCObjective(
        lambda samples, X=None: samples.squeeze(-1).clamp_min(COST_AWARE_MIN_EXPECTED_COST)  # noqa: ARG005, N803
    )


class EIpuAcquisition(AcquisitionFunction):
    """Expected Improvement per Unit cost acquisition function, in log space.

    Computes ``log qNEI(X) - log E[cost(X)]``, whose maximizer coincides
    with that of ``qNEI(X) / E[cost(X)]`` (Snoek et al. 2012). Two design
    points matter here:

    * The improvement term is a Monte-Carlo
      :class:`~botorch.acquisition.logei.qLogNoisyExpectedImprovement`, so
      :meth:`set_X_pending` genuinely conditions the surface on pending
      points — BoTorch's sequential-greedy batch loop relies on that
      conditioning to produce distinct batch members, and in-flight
      experiments passed as ``X_pending`` are honored. An analytic EI core
      cannot do either (its ``set_X_pending`` is unsupported).
    * The inverse-cost weighting is delegated to BoTorch's
      :class:`~botorch.acquisition.cost_aware.InverseCostWeightedUtility` in
      log mode, which evaluates the cost posterior *with* gradients so
      L-BFGS-B sees the full quotient-rule gradient of ``EI/cost``. Staying
      in log space turns the numerically sensitive division into a
      subtraction and matches the log-scale output of the improvement term.

    Outcome constraints are handled inside the wrapped improvement term via
    BoTorch's smoothed feasibility weighting (the same path the standard
    non-cost acquisition uses), not by this wrapper.
    """

    def __init__(
        self,
        improvement: qLogNoisyExpectedImprovement,
        cost_model: SingleTaskGP,
    ) -> None:
        """Initialize EIpu acquisition.

        Args:
            improvement: Fitted MC improvement acquisition (already carrying
                any outcome-constraint feasibility weighting)
            cost_model: Fitted cost model
        """
        super().__init__(improvement.model)
        self.improvement = improvement
        self.cost_model = cost_model
        self.cost_utility = InverseCostWeightedUtility(
            cost_model=cost_model,
            use_mean=True,
            cost_objective=_positive_cost_objective(),
            log=True,
        )

    @property
    def X_pending(self) -> Tensor | None:  # noqa: N802
        """Pending candidates, as tracked by the inner improvement term."""
        return self.improvement.X_pending

    @X_pending.setter
    def X_pending(self, value: Tensor | None) -> None:  # noqa: N802
        """Set pending candidates by delegating to :meth:`set_X_pending`."""
        self.set_X_pending(value)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:  # noqa: N802, N803
        """Condition the improvement term on pending points.

        qLogNEI folds pending points into its incumbent baseline (its
        incremental mode), so subsequent evaluations measure improvement
        *beyond* the pending picks — the mechanism BoTorch's sequential
        greedy optimizer uses to diversify batch members.
        """
        self.improvement.set_X_pending(X_pending)

    def forward(self, X: Tensor) -> Tensor:  # noqa: N803
        """Compute the log EI-per-unit-cost acquisition value.

        Args:
            X: Candidate points of shape (..., q, d) where:
               - ... are batch dimensions (e.g., num_restarts, num_samples)
               - q is the batch size for joint acquisition (usually 1)
               - d is the input dimension

        Returns:
            Log-space acquisition values of shape (...) matching the
            improvement term's output
        """
        # Log-space improvement - returns shape (...) = batch_shape
        log_improvement = self.improvement(X)

        # In log mode the utility subtracts ``log(sum_q E[cost])`` from the
        # log improvement, differentiating through the cost posterior so the
        # optimizer sees the full quotient gradient. ``deltas`` is
        # ``num_fantasies x batch_shape``; the improvement is not fantasized,
        # so we add a singleton leading dim and drop it after.
        return self.cost_utility(X=X, deltas=log_improvement.unsqueeze(0)).squeeze(0)


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
        msg = f"{label} must be a (indices, coefficients, rhs) tuple"
        raise ValueError(msg)
    indices, coefficients, rhs = entry
    if not isinstance(indices, Tensor) or indices.dim() != 1:
        msg = f"{label}.indices must be a 1-D tensor"
        raise ValueError(msg)
    if not isinstance(coefficients, Tensor) or coefficients.dim() != 1:
        msg = f"{label}.coefficients must be a 1-D tensor"
        raise ValueError(msg)
    if indices.numel() != coefficients.numel():
        msg = (
            f"{label} indices ({indices.numel()}) and "
            f"coefficients ({coefficients.numel()}) must have the same length"
        )
        raise ValueError(msg)
    if indices.numel() == 0:
        msg = f"{label} must reference at least one parameter"
        raise ValueError(msg)
    if not torch.isfinite(torch.as_tensor(rhs, dtype=torch.float64)).item():
        msg = f"{label}.rhs must be finite, got {rhs!r}"
        raise ValueError(msg)
    index_min = int(indices.min().item())
    index_max = int(indices.max().item())
    if index_min < 0 or index_max >= n_dims:
        msg = (
            f"{label} references parameter index out of range "
            f"[0, {n_dims}); got min={index_min}, max={index_max}"
        )
        raise ValueError(msg)


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


def _move_linear_constraints_to_bounds(
    constraints: list[tuple[Tensor, Tensor, float]] | None,
    bounds: Tensor,
) -> list[tuple[Tensor, Tensor, float]] | None:
    """Move BoTorch linear constraint tensors onto the optimization tensor device."""
    if not constraints:
        return constraints
    return [
        (
            indices.to(device=bounds.device, dtype=torch.long),
            coefficients.to(device=bounds.device, dtype=bounds.dtype),
            float(rhs),
        )
        for indices, coefficients, rhs in constraints
    ]


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
    x_avoid: Tensor | None = None,
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
        inequality_constraints: BoTorch linear inequality constraints. Each
            tuple is (indices, coefficients, rhs) enforcing
            ``sum_i X[indices[i]] * coefficients[i] >= rhs`` (BoTorch's
            documented convention; see ``build_botorch_linear_constraints``).
            Enforced on the CONTINUOUS and MIXED branches; constraints that
            touch categorical parameters never reach this function — they
            are classified for post-hoc projection upstream.
        equality_constraints: BoTorch linear equality constraints. Each
            tuple is (indices, coefficients, rhs) enforcing
            ``sum_i X[indices[i]] * coefficients[i] = rhs``. Same branch
            coverage as ``inequality_constraints``.
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
    inequality_constraints = _move_linear_constraints_to_bounds(inequality_constraints, bounds)
    equality_constraints = _move_linear_constraints_to_bounds(equality_constraints, bounds)

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
    if space_type == SearchSpaceType.MIXED:
        n_combos = count_categorical_combinations(spec)
        if n_combos > MIXED_CATEGORICAL_COMBO_THRESHOLD:
            msg = (
                f"Mixed spaces with more than {MIXED_CATEGORICAL_COMBO_THRESHOLD} "
                f"categorical combinations are not yet supported (this space has "
                f"{n_combos}). Consider reducing the number of categories. "
                "A future version will support optimize_acqf_mixed_alternating "
                "with integer encoding for larger mixed spaces."
            )
            raise NotImplementedError(msg)
        _apply_pending_to_acqf(acqf, X_pending)
        return _optimize_mixed(
            acqf,
            bounds,
            spec,
            batch_size,
            effective_restarts,
            effective_samples,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )
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
        # Fall back to the attribute for custom acquisitions that expose a
        # property setter but no ``set_X_pending`` method.
        acqf.X_pending = X_pending


def _merge_avoid_tensors(
    x_avoid: Tensor | None,
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


def _lbfgs_options() -> dict[str, bool | float | int | str]:
    """L-BFGS-B inner-loop options shared by the continuous and mixed paths.

    Centralizing the dict keeps the two ``optimize_acqf`` call sites from
    drifting and routes the budget knobs through ``constants`` per project
    policy. The return type matches BoTorch's ``optimize_acqf(options=...)``
    parameter so the (invariant) ``dict`` value type checks at both call sites.
    """
    return {
        "batch_limit": ACQF_LBFGS_BATCH_LIMIT,
        "maxiter": ACQF_LBFGS_MAXITER,
    }


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
        inequality_constraints: BoTorch linear inequality constraints
            (``coefficients @ X[indices] >= rhs``)
        equality_constraints: BoTorch linear equality constraints
            (``coefficients @ X[indices] = rhs``)

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
        "options": _lbfgs_options(),
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
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize acquisition over a mixed continuous + categorical space.

    For each categorical combination, runs L-BFGS-B optimization over the
    continuous dimensions, then returns the best result. Native linear
    constraints (which by construction reference only non-categorical
    dimensions — see ``build_botorch_linear_constraints``) are enforced
    inside each per-combination run by ``optimize_acqf_mixed``.

    Reference:
        https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf_mixed

    Args:
        acqf: Acquisition function to optimize
        bounds: Parameter bounds of shape (2, n_dims)
        spec: Optimization specification (mixed space)
        batch_size: Number of candidates to generate
        num_restarts: Number of optimization restarts
        raw_samples: Number of raw samples for initialization
        inequality_constraints: BoTorch linear inequality constraints
            (``coefficients @ X[indices] >= rhs``)
        equality_constraints: BoTorch linear equality constraints
            (``coefficients @ X[indices] = rhs``)

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
        options=_lbfgs_options(),
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
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
    return train_y.max().item()
