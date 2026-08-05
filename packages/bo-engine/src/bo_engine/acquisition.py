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

import itertools
import logging
import math
import random
from collections.abc import Callable
from typing import Any, cast

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.objective import GenericMCObjective
from botorch.exceptions.errors import InfeasibilityError
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.optim.optimize import optimize_acqf_discrete, optimize_acqf_mixed
from botorch.utils.sampling import get_polytope_samples
from torch import Tensor

from bo_engine.constants import (
    ACQF_FALLBACK_MAX_ASSIGNMENTS,
    ACQF_FALLBACK_MIN_SAMPLES,
    ACQF_FALLBACK_SAMPLE_SEED,
    ACQF_LBFGS_BATCH_LIMIT,
    ACQF_LBFGS_MAXITER,
    COST_AWARE_MIN_EXPECTED_COST,
    DEFAULT_UCB_BETA,
    NUMERICAL_EPSILON,
    RESTART_WARN_TOLERANCE,
)
from bo_engine.device import ensure_device, to_device
from bo_engine.initial_design import SearchSpaceExhaustedError
from bo_engine.reproducibility import create_reproducible_sobol
from bo_engine.result_validation import duplicate_row_mask
from bo_engine.transforms import (
    SearchSpaceType,
    build_fixed_features_list,
    classify_search_space,
    enumerate_discrete_choices,
    enumerate_numeric_discrete_grid,
    mixed_space_combo_limit_message,
    mixed_space_combo_overflow,
    numeric_discrete_axes,
    snap_discrete_columns,
)
from bo_engine.types import (
    SINGLE_OBJECTIVE_ONLY_ACQUISITION,
    UCB_FAMILY_ACQUISITION,
    AcquisitionConfig,
    AcquisitionMethod,
    OptimizationSpec,
    ParameterType,
)

logger = logging.getLogger(__name__)

# Acquisition methods this BoTorch engine cannot express with its current
# optimize path (Thompson sampling needs per-sample posterior draws,
# knowledge gradient needs the one-shot fantasy optimizer, qNIPV needs an
# integration point set, and the pure posterior-statistic methods have no
# batch-aware constrained wiring here). The BoTorch backend's
# ``validate_capabilities`` reports these UNSUPPORTED (acknowledgeable to
# IGNORED — the run then falls back to the objective-family default below),
# mirroring the ``BAYBE_UNSUPPORTED_ACQUISITION`` pattern on the BayBE side.
BOTORCH_UNSUPPORTED_ACQUISITION: frozenset[AcquisitionMethod] = frozenset(
    {
        AcquisitionMethod.THOMPSON_SAMPLING,
        AcquisitionMethod.KNOWLEDGE_GRADIENT,
        AcquisitionMethod.ACTIVE_LEARNING,
        AcquisitionMethod.POSTERIOR_MEAN,
        AcquisitionMethod.POSTERIOR_STANDARD_DEVIATION,
    }
)

# Single-objective methods built directly by ``_build_simple_mc_acquisition``.
# They share the plain (unconstrained) MC construction path; constrained
# runs route through NOISY_EI's feasibility weighting instead (see
# ``_create_single_objective_dispatch``).
_SIMPLE_MC_METHODS: frozenset[AcquisitionMethod] = frozenset(
    {
        AcquisitionMethod.UPPER_CONFIDENCE_BOUND,
        AcquisitionMethod.PROBABILITY_OF_IMPROVEMENT,
        AcquisitionMethod.SIMPLE_REGRET,
        AcquisitionMethod.EXPECTED_IMPROVEMENT_NONLOG,
        AcquisitionMethod.NOISY_EI_NONLOG,
    }
)


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

    # Hypervolume improvement: the default log formulation (qLogNEHVI), or
    # the explicit non-log sibling (qNEHVI) when the caller requested it.
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

    if method == AcquisitionMethod.HYPERVOLUME_IMPROVEMENT_NONLOG:
        return qNoisyExpectedHypervolumeImprovement(**acqf_kwargs)
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
        beta=config.beta,
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
    beta: float | None = None,
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
        beta: UCB-family exploration weight. Only accepted together with a
            member of ``UCB_FAMILY_ACQUISITION``; ``None`` falls back to
            ``DEFAULT_UCB_BETA``.

    Returns:
        Acquisition function appropriate for the problem
    """
    if beta is not None and method not in UCB_FAMILY_ACQUISITION:
        msg = (
            f"acquisition beta={beta} is only valid for the UCB acquisition "
            f"family; got method={method.value!r}. Remove beta or select "
            "upper_confidence_bound."
        )
        raise ValueError(msg)

    # Determine method if AUTO
    if method == AcquisitionMethod.AUTO:
        if n_objectives == 1:
            method = AcquisitionMethod.NOISY_EI
        else:
            method = AcquisitionMethod.HYPERVOLUME_IMPROVEMENT

    if method in BOTORCH_UNSUPPORTED_ACQUISITION:
        # Reachable only when the caller acknowledged the degradation at
        # intake (validate_capabilities reports these UNSUPPORTED); fall
        # back to the objective-family default, mirroring the BayBE side.
        fallback = (
            AcquisitionMethod.NOISY_EI
            if n_objectives == 1
            else AcquisitionMethod.HYPERVOLUME_IMPROVEMENT
        )
        logger.warning(
            "Acquisition method %s is not supported by the BoTorch engine; falling back to %s.",
            method.value,
            fallback.value,
        )
        method = fallback

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
            beta=beta,
        )

    if maximize_mask is None:
        msg = (
            "create_acquisition requires ``maximize_mask`` for "
            "multi-objective problems; see bo_engine.types for the sign "
            "convention."
        )
        raise ValueError(msg)
    if method in SINGLE_OBJECTIVE_ONLY_ACQUISITION:
        # Reachable only when the caller acknowledged the degradation at
        # intake (validate_capabilities reports the combination); mirror
        # the single-objective family fallback above, warning included,
        # so the discarded request (and any acquisition_beta riding on
        # it) is visible in the logs instead of vanishing silently.
        fallback = AcquisitionMethod.HYPERVOLUME_IMPROVEMENT
        logger.warning(
            "Acquisition method %s has single-objective semantics only; "
            "falling back to %s for this multi-objective problem.",
            method.value,
            fallback.value,
        )
        method = fallback
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
    beta: float | None = None,
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
        beta: UCB-family exploration weight (``None`` = engine default)

    Returns:
        Single-objective acquisition function
    """
    objective_gp = _extract_objective_gp(model)

    if method == AcquisitionMethod.COST_WEIGHTED_EI and cost_model is None:
        logger.warning(
            "COST_WEIGHTED_EI requested but no cost model available. "
            "Falling back to standard Noisy Expected Improvement."
        )

    acq_model, all_constraints = _bundle_outcome_constraint_models(
        objective_gp, constraints, outcome_constraint_models
    )

    if method in _SIMPLE_MC_METHODS and not all_constraints:
        return _build_simple_mc_acquisition(method, objective_gp, train_x, train_y, beta=beta)
    if method in _SIMPLE_MC_METHODS:
        # The simple MC family has no feasibility-weighted wiring here;
        # route through NOISY_EI (the same upgrade the analytic-EI path
        # performs) so constrained runs keep BoTorch's smoothed
        # feasibility weighting instead of silently ignoring constraints.
        logger.warning(
            "Acquisition method %s does not support constraint weighting in "
            "this engine. Routing through NOISY_EI so BoTorch can apply the "
            "feasibility weighting.",
            method.value,
        )
        method = AcquisitionMethod.NOISY_EI

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


def _extract_objective_gp(model: ModelListGP | SingleTaskGP) -> SingleTaskGP:
    """Return the single-objective GP, unwrapping a one-model ``ModelListGP``."""
    if isinstance(model, SingleTaskGP):
        return model
    if isinstance(model, ModelListGP) and len(model.models) == 1:
        return cast(SingleTaskGP, model.models[0])
    msg = "Single-objective requires SingleTaskGP model"
    raise ValueError(msg)


def _build_simple_mc_acquisition(
    method: AcquisitionMethod,
    model: SingleTaskGP,
    train_x: Tensor,
    train_y: Tensor,
    *,
    beta: float | None,
) -> AcquisitionFunction:
    """Build one of the plain (unconstrained) Monte-Carlo acquisition functions.

    ``train_y`` is in maximization form, so ``best_f = train_y.max()`` for
    the improvement-threshold members. ``beta`` applies only to the UCB
    member (validated upstream by :func:`create_acquisition`); ``None``
    falls back to :data:`~bo_engine.constants.DEFAULT_UCB_BETA`.
    """
    if method == AcquisitionMethod.UPPER_CONFIDENCE_BOUND:
        return qUpperConfidenceBound(
            model=model,
            beta=beta if beta is not None else DEFAULT_UCB_BETA,
        )
    if method == AcquisitionMethod.PROBABILITY_OF_IMPROVEMENT:
        return qProbabilityOfImprovement(model=model, best_f=train_y.max().item())
    if method == AcquisitionMethod.SIMPLE_REGRET:
        return qSimpleRegret(model=model)
    if method == AcquisitionMethod.EXPECTED_IMPROVEMENT_NONLOG:
        return qExpectedImprovement(model=model, best_f=train_y.max().item())
    if method == AcquisitionMethod.NOISY_EI_NONLOG:
        return qNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x,
            prune_baseline=True,
            cache_root=False,
        )
    msg = f"{method} is not a simple MC acquisition method"
    raise ValueError(msg)


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
    random_seed: int | None = None,
    domain_bounds: Tensor | None = None,
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
            Matching uses the engine's duplicate-detection tolerance on
            canonical coordinates (numeric-discrete columns snapped to their
            grids first). Enforcement is per space type:

            - Continuous, ``batch_size == 1``: the highest-valued unseen
              restart is selected, with a seeded, acquisition-ranked sampling
              fallback if every restart collapses onto an avoided point.
              When the spec has numeric-discrete parameters, candidates are
              returned in canonical (grid-snapped) coordinates and ranked by
              the acquisition value at those coordinates, so the reported
              value describes the experiment that will actually run.
            - Continuous, ``batch_size > 1``: not enforced (logged at debug
              level); sequential greedy optimization has no per-restart
              selection point to filter.
            - Purely categorical: excluded from the enumerated choice set.
            - Mixed: not enforced (logged at debug level).
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
            ``x_avoid`` so the discrete optimizer excludes them; the
            continuous path merges them into ``x_avoid`` as well, so the
            ``batch_size == 1`` selection cannot return a pending point.
        random_seed: Seed for the continuous unseen-candidate fallback
            sampler. ``None`` falls back to a fixed default seed so
            suggestions stay reproducible run-to-run.
        domain_bounds: Full campaign bounds of shape ``(2, n_dims)`` when
            ``bounds`` is a narrowed optimization region (e.g. a TuRBO trust
            region). Canonical candidates are validated — and search-space
            exhaustion is judged — against this domain, never against the
            local region: a trust region that happens to contain no grid
            point must not terminate a viable campaign. Defaults to
            ``bounds``.

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
    if x_avoid is not None:
        x_avoid = to_device(x_avoid)
    domain_bounds = to_device(domain_bounds) if domain_bounds is not None else bounds

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
            spec=None,
            x_avoid=_merge_avoid_tensors(x_avoid, X_pending),
            random_seed=random_seed,
            domain_bounds=domain_bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )

    space_type = classify_search_space(spec)

    if space_type == SearchSpaceType.PURELY_CATEGORICAL:
        merged_avoid = _merge_avoid_tensors(x_avoid, X_pending)
        return _optimize_discrete(acqf, spec, batch_size, merged_avoid)
    if space_type == SearchSpaceType.MIXED:
        combo_overflow = mixed_space_combo_overflow(spec)
        if combo_overflow is not None:
            raise NotImplementedError(mixed_space_combo_limit_message(combo_overflow))
        if x_avoid is not None and x_avoid.numel() > 0:
            logger.debug(
                "x_avoid is not enforced for mixed-space acquisition optimization; "
                "%d avoided point(s) are ignored.",
                int(x_avoid.shape[0]),
            )
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
        spec=spec,
        x_avoid=_merge_avoid_tensors(x_avoid, X_pending),
        random_seed=random_seed,
        domain_bounds=domain_bounds,
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
    spec: OptimizationSpec | None = None,
    x_avoid: Tensor | None = None,
    random_seed: int | None = None,
    domain_bounds: Tensor | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize acquisition over continuous space using L-BFGS-B.

    Args:
        acqf: Acquisition function to optimize
        bounds: Parameter bounds of shape (2, n_dims) used by the optimizer
            (possibly a narrowed trust region)
        batch_size: Number of candidates to generate
        num_restarts: Number of optimization restarts
        raw_samples: Number of raw samples for initialization
        spec: Optimization specification; supplies the numeric-discrete grids
            used to canonicalize coordinates before avoided-point matching
        x_avoid: Previously evaluated or pending points. Enforced only for
            ``batch_size == 1`` (see ``optimize_acquisition``); larger batches
            log the dropped points at debug level
        random_seed: Seed for the unseen-candidate fallback sampler
        domain_bounds: Full campaign bounds used to validate canonical
            candidates and judge exhaustion; defaults to ``bounds``
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
        domain = domain_bounds if domain_bounds is not None else bounds
        avoid_canonical = snap_discrete_columns(x_avoid, spec) if x_avoid is not None else None
        restart_points = snap_discrete_columns(candidates[:, 0, :], spec)
        eligible = ~_matches_avoided_points(restart_points, avoid_canonical)
        if _spec_has_numeric_discrete(spec):
            # Snapping moved the restarts onto their grids, so the optimizer's
            # values describe coordinates that will never be executed —
            # selection must rank the canonical points on the acquisition
            # surface, and the returned value must describe the returned point.
            # Snapping also happens after the optimizer's feasibility
            # handling, so a feasible relaxed restart can land on a grid
            # value outside the campaign domain or across a linear
            # constraint — canonical points must be re-validated. Validation
            # uses the campaign domain, not the (possibly narrower) trust
            # region: a snapped grid point just outside a shrunken region is
            # still an executable experiment.
            with torch.no_grad():
                acq_values = acqf(restart_points.unsqueeze(-2)).reshape(-1)
            eligible &= _domain_feasible_mask(
                restart_points, domain, inequality_constraints, equality_constraints
            )
        eligible_indices = eligible.nonzero(as_tuple=True)[0]
        if eligible_indices.numel() > 0:
            best_idx = _select_best_restart_index(
                restart_points, acq_values, eligible, eligible_indices, bounds
            )
            candidates = restart_points[best_idx : best_idx + 1]
            acq_values = acq_values[best_idx : best_idx + 1]
        else:
            candidates, acq_values = _continuous_unseen_fallback(
                acqf,
                bounds,
                spec=spec,
                x_avoid=avoid_canonical,
                sample_count=max(raw_samples, ACQF_FALLBACK_MIN_SAMPLES),
                random_seed=random_seed,
                domain_bounds=domain,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
            )
    elif x_avoid is not None and x_avoid.numel() > 0:
        logger.debug(
            "x_avoid is not enforced for continuous acquisition optimization with "
            "batch_size=%d; %d avoided point(s) are ignored.",
            batch_size,
            int(x_avoid.shape[0]),
        )
    return candidates, acq_values


def _select_best_restart_index(
    restart_points: Tensor,
    acq_values: Tensor,
    eligible: Tensor,
    eligible_indices: Tensor,
    local_bounds: Tensor,
) -> int:
    """Pick the highest-valued eligible restart, preferring the local region.

    One locality policy everywhere: prefer candidates inside the local
    optimization region, then widen to the domain with a warning. Pure
    continuous restarts always lie inside ``local_bounds``, so this only
    bites when snapping moved a grid point outside a trust region.
    """
    local_indices = (eligible & _rows_within_bounds(restart_points, local_bounds)).nonzero(
        as_tuple=True
    )[0]
    if local_indices.numel() > 0:
        selection_pool = local_indices
    else:
        logger.warning(_LOCAL_REGION_FALLBACK_WARNING, int(eligible_indices.numel()))
        selection_pool = eligible_indices
    return int(selection_pool[acq_values[selection_pool].argmax()].item())


def _spec_has_numeric_discrete(spec: OptimizationSpec | None) -> bool:
    """True when the spec contains at least one numeric-discrete parameter."""
    return spec is not None and any(p.type == ParameterType.DISCRETE for p in spec.parameters)


def _linear_constraint_mask(
    points: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None,
) -> Tensor:
    """Return one boolean per point marking feasibility under linear constraints.

    Uses BoTorch's constraint convention (``coefficients @ X[indices] >= rhs``
    for inequalities, ``= rhs`` for equalities) with ``NUMERICAL_EPSILON``
    slack so boundary points are not rejected by rounding noise.
    """
    mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    for indices, coefficients, rhs in inequality_constraints or []:
        lhs = (points[:, indices] * coefficients).sum(dim=-1)
        mask &= lhs >= rhs - NUMERICAL_EPSILON
    for indices, coefficients, rhs in equality_constraints or []:
        lhs = (points[:, indices] * coefficients).sum(dim=-1)
        mask &= (lhs - rhs).abs() <= NUMERICAL_EPSILON
    return mask


# Shared warning for the one locality policy: candidates inside the local
# optimization region are preferred; when none exist, selection widens to the
# full campaign domain instead of failing or terminating the campaign.
_LOCAL_REGION_FALLBACK_WARNING = (
    "No unseen executable candidate lies inside the current optimization "
    "region; selecting among %d candidate(s) from the full campaign domain."
)


def _rows_within_bounds(points: Tensor, bounds: Tensor) -> Tensor:
    """Return one boolean per row marking containment in ``bounds`` (with slack)."""
    within = (points >= bounds[0] - NUMERICAL_EPSILON) & (points <= bounds[1] + NUMERICAL_EPSILON)
    return within.all(dim=-1)


def _prefer_local_rows(choices: Tensor, local_bounds: Tensor) -> Tensor:
    """Restrict candidates to the local optimization region when possible.

    Locality (a TuRBO trust region) is a soft preference, not a validity
    requirement: candidates outside it are still executable experiments, so
    when no candidate lies inside, the domain-wide set is used with a
    warning instead of failing.
    """
    within_local = _rows_within_bounds(choices, local_bounds).nonzero(as_tuple=True)[0]
    if within_local.numel() > 0:
        return choices[within_local]
    logger.warning(_LOCAL_REGION_FALLBACK_WARNING, int(choices.shape[0]))
    return choices


def _domain_feasible_mask(
    points: Tensor,
    bounds: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None,
) -> Tensor:
    """Return one boolean per point marking bounds and constraint feasibility.

    Canonical (grid-snapped or enumerated) coordinates are produced *after*
    the optimizer's own feasibility handling, so they must be re-validated:
    snapping can move a feasible relaxed point onto a grid value outside the
    campaign domain or across a linear constraint. ``NUMERICAL_EPSILON``
    slack keeps boundary points feasible.
    """
    return _rows_within_bounds(points, bounds) & _linear_constraint_mask(
        points, inequality_constraints, equality_constraints
    )


def _matches_avoided_points(points: Tensor, x_avoid: Tensor | None) -> Tensor:
    """Return one boolean per point marking duplicates of avoided points.

    Delegates to :func:`bo_engine.result_validation.duplicate_row_mask` so
    acquisition-level exclusion shares the engine's single definition of
    "same experiment" (``DUPLICATE_DETECTION_TOLERANCE``, dimension-scaled
    Euclidean distance). Callers pass canonical (grid-snapped) coordinates.
    """
    flat_points = points.reshape(-1, points.shape[-1])
    if x_avoid is None or x_avoid.numel() == 0:
        return torch.zeros(flat_points.shape[0], dtype=torch.bool, device=flat_points.device)
    avoided = x_avoid.to(device=flat_points.device, dtype=flat_points.dtype).reshape(
        -1, flat_points.shape[-1]
    )
    return duplicate_row_mask(flat_points, avoided)


def _enumerated_unseen_choices(
    grid: Tensor,
    bounds: Tensor,
    domain_bounds: Tensor,
    x_avoid: Tensor | None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None,
) -> Tensor:
    """Filter a fully enumerated grid down to feasible unseen points.

    Eligibility — and therefore exhaustion — is judged against the campaign
    ``domain_bounds``: a narrowed optimization region (TuRBO trust region)
    that happens to contain no grid point must never masquerade as global
    exhaustion. Among the eligible points, those inside the local ``bounds``
    are preferred so trust-region locality is kept whenever possible.

    Raises:
        SearchSpaceExhaustedError: If nothing remains anywhere in the
            campaign domain — the grid covers the entire space, so an empty
            result proves exhaustion.
    """
    choices = grid.to(device=domain_bounds.device, dtype=domain_bounds.dtype)
    eligible = _domain_feasible_mask(
        choices, domain_bounds, inequality_constraints, equality_constraints
    ) & ~_matches_avoided_points(choices, x_avoid)
    choices = choices[eligible]
    if choices.shape[0] == 0:
        raise SearchSpaceExhaustedError(
            n_requested=1,
            n_available=0,
            n_total_combinations=int(grid.shape[0]),
        )
    return _prefer_local_rows(choices, bounds)


def _constrained_discrete_columns(
    axes: list[tuple[int, list[float]]],
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None,
) -> bool:
    """True when any linear constraint references a numeric-discrete column."""
    discrete_columns = {column for column, _ in axes}
    for indices, _, _ in itertools.chain(inequality_constraints or [], equality_constraints or []):
        if any(int(i) in discrete_columns for i in indices.tolist()):
            return True
    return False


def _reduce_constraints_for_assignment(
    assignment: dict[int, float],
    continuous_columns: list[int],
    constraints: list[tuple[Tensor, Tensor, float]] | None,
    is_equality: bool,
) -> list[tuple[Tensor, Tensor, float]] | None:
    """Substitute a discrete assignment into linear constraints.

    Each constraint's discrete terms are folded into the right-hand side and
    the remaining indices are remapped into the continuous-only column space.
    Returns ``None`` when a constraint with no continuous terms left is
    violated by the assignment — the assignment is infeasible outright.
    """
    reduced: list[tuple[Tensor, Tensor, float]] = []
    for indices, coefficients, rhs in constraints or []:
        index_list = [int(i) for i in indices.tolist()]
        kept = [pos for pos, column in enumerate(index_list) if column not in assignment]
        new_rhs = float(rhs) - sum(
            float(coefficients[pos]) * assignment[column]
            for pos, column in enumerate(index_list)
            if column in assignment
        )
        if not kept:
            equality_violated = is_equality and abs(new_rhs) > NUMERICAL_EPSILON
            inequality_violated = not is_equality and new_rhs > NUMERICAL_EPSILON
            if equality_violated or inequality_violated:
                return None
            continue
        reduced.append(
            (
                torch.tensor(
                    [continuous_columns.index(index_list[pos]) for pos in kept],
                    dtype=torch.long,
                    device=indices.device,
                ),
                coefficients[kept],
                new_rhs,
            )
        )
    return reduced


def _select_assignments(
    axes: list[tuple[int, list[float]]],
    budget: int,
    seed: int,
) -> list[tuple[float, ...]]:
    """Choose the discrete assignments to condition on, within a fixed budget.

    The assignment cardinality is computed from axis sizes *before* anything
    is materialized — a Cartesian product over large grids must never be
    built just to count it. Within the budget the full product is used; above
    it, exactly ``budget`` distinct assignments are drawn by sampling flat
    indices without replacement (``random.sample`` over an index range stays
    O(budget) for budget << cardinality) and decoding them mixed-radix into
    axis coordinates. Drawing with replacement and deduplicating would
    silently underfill the budget and raise the odds of missing the only
    feasible assignment. Large constraint-coupled spaces thus stay supported
    (never a reversion to snap-and-filter, which destroys discrete-coupled
    equalities) at bounded cost.
    """
    axis_values = [axis for _, axis in axes]
    cardinality = math.prod(len(axis) for axis in axis_values)
    if cardinality <= budget:
        return list(itertools.product(*axis_values))
    logger.info(
        "Constraint-coupled discrete space has %d assignments; conditioning "
        "on a seeded subsample of %d.",
        cardinality,
        budget,
    )
    # Not a security context: this is reproducibility-seeded numerical
    # subsampling, and ``random.sample`` over an index range is the only
    # stdlib O(budget) without-replacement draw that supports cardinalities
    # beyond int64 (unlike ``torch.randint``).
    flat_indices = random.Random(seed).sample(range(cardinality), budget)  # noqa: S311
    assignments: list[tuple[float, ...]] = []
    for flat_index in flat_indices:
        coordinates: list[float] = []
        remainder = flat_index
        for axis in reversed(axis_values):
            coordinates.append(axis[remainder % len(axis)])
            remainder //= len(axis)
        assignments.append(tuple(reversed(coordinates)))
    return assignments


def _assignment_conditioned_cloud(
    domain_bounds: Tensor,
    axes: list[tuple[int, list[float]]],
    sample_count: int,
    seed: int,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None,
) -> Tensor:
    """Build a canonical candidate cloud conditioned on discrete assignments.

    Constraints that couple discrete and continuous columns break under
    post-hoc snapping (a relaxed sample satisfying ``d + c = 1`` almost never
    still does once ``d`` snaps to its grid), so feasible candidates must be
    generated per canonical assignment: substitute each discrete assignment
    into the constraints and sample the reduced continuous subproblem
    directly. Assignments whose reduced subproblem is infeasible are skipped;
    the assignment set itself is budgeted by :func:`_select_assignments`.
    """
    assignments = _select_assignments(axes, ACQF_FALLBACK_MAX_ASSIGNMENTS, seed)
    n_dims = int(domain_bounds.shape[-1])
    discrete_columns = [column for column, _ in axes]
    continuous_columns = [i for i in range(n_dims) if i not in discrete_columns]
    continuous_bounds = domain_bounds[:, continuous_columns]
    samples_per_assignment = max(1, sample_count // len(assignments))

    clouds: list[Tensor] = []
    for offset, values in enumerate(assignments):
        assignment = dict(zip(discrete_columns, [float(v) for v in values], strict=True))
        reduced_ineq = _reduce_constraints_for_assignment(
            assignment, continuous_columns, inequality_constraints, is_equality=False
        )
        reduced_eq = _reduce_constraints_for_assignment(
            assignment, continuous_columns, equality_constraints, is_equality=True
        )
        if reduced_ineq is None or reduced_eq is None:
            continue
        try:
            if reduced_ineq or reduced_eq:
                continuous = get_polytope_samples(
                    n=samples_per_assignment,
                    bounds=continuous_bounds,
                    inequality_constraints=reduced_ineq or None,
                    equality_constraints=reduced_eq or None,
                    seed=seed + offset,
                )
            else:
                continuous = create_reproducible_sobol(
                    len(continuous_columns),
                    samples_per_assignment,
                    seed + offset,
                    continuous_bounds,
                )
        except InfeasibilityError:
            # The assignment leaves an empty continuous polytope (e.g. an
            # equality forcing a value outside the continuous bounds).
            continue
        cloud = torch.empty(
            samples_per_assignment, n_dims, dtype=domain_bounds.dtype, device=domain_bounds.device
        )
        cloud[:, continuous_columns] = continuous
        for column, value in assignment.items():
            cloud[:, column] = value
        clouds.append(cloud)
    if not clouds:
        return torch.empty(0, n_dims, dtype=domain_bounds.dtype, device=domain_bounds.device)
    return torch.cat(clouds, dim=0)


def _sampled_unseen_choices(
    bounds: Tensor,
    *,
    domain_bounds: Tensor,
    spec: OptimizationSpec | None,
    x_avoid: Tensor | None,
    sample_count: int,
    random_seed: int | None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None,
) -> Tensor:
    """Draw a seeded candidate cloud and filter it to feasible unseen points.

    Constraints that reference numeric-discrete columns are handled by
    per-assignment conditional generation over the full campaign domain (see
    :func:`_assignment_conditioned_cloud`); everything else samples the local
    ``bounds`` and canonicalizes afterwards. The final selection prefers
    candidates inside the local region (see :func:`_prefer_local_rows`).

    Raises:
        RuntimeError: If nothing remains. A sampled cloud cannot prove
            exhaustion, so the error explicitly leaves that open.
    """
    seed = ACQF_FALLBACK_SAMPLE_SEED if random_seed is None else random_seed
    axes = numeric_discrete_axes(spec) if spec is not None else None
    if axes and _constrained_discrete_columns(axes, inequality_constraints, equality_constraints):
        choices = _assignment_conditioned_cloud(
            domain_bounds, axes, sample_count, seed, inequality_constraints, equality_constraints
        )
    else:
        if inequality_constraints or equality_constraints:
            choices = get_polytope_samples(
                n=sample_count,
                bounds=bounds,
                inequality_constraints=inequality_constraints or None,
                equality_constraints=equality_constraints or None,
                seed=seed,
            )
        else:
            choices = create_reproducible_sobol(int(bounds.shape[-1]), sample_count, seed, bounds)
        # The executed experiment is the canonical coordinate, so eligibility,
        # ranking, and the returned candidate all use the snapped cloud.
        choices = snap_discrete_columns(choices, spec)
    eligible = ~_matches_avoided_points(choices, x_avoid)
    if _spec_has_numeric_discrete(spec):
        # Snapping (or assignment substitution) happens after the sampler's
        # feasibility handling, so canonical points are re-validated against
        # the campaign domain.
        eligible &= _domain_feasible_mask(
            choices, domain_bounds, inequality_constraints, equality_constraints
        )
    choices = choices[eligible]
    if choices.shape[0] == 0:
        msg = (
            "All continuous acquisition restarts matched avoided points, and "
            "no unseen alternative was found among the sampled fallback "
            "candidates. The space may still contain unseen points."
        )
        raise RuntimeError(msg)
    return _prefer_local_rows(choices, bounds)


def _continuous_unseen_fallback(
    acqf: AcquisitionFunction,
    bounds: Tensor,
    *,
    spec: OptimizationSpec | None,
    x_avoid: Tensor | None,
    sample_count: int,
    random_seed: int | None,
    domain_bounds: Tensor | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None,
) -> tuple[Tensor, Tensor]:
    """Choose the best unseen point from a feasible candidate cloud.

    This path is used only when every L-BFGS-B restart converges to an already
    evaluated point. A finite alternative set is built without perturbing or
    silently resubmitting that maximizer: an all-numeric-discrete spec
    enumerates its full Cartesian grid (which proves exhaustion when nothing
    unseen remains), unconstrained spaces draw a seeded Sobol design, and
    constrained spaces draw feasible points from the polytope directly
    (hit-and-run), so linear inequality *and* equality constraints are
    honored by construction. Sampled candidates are canonicalized (grid
    columns snapped) before ranking so the returned coordinate is the one
    that will be executed, and canonical points are re-validated against
    bounds and native linear constraints because snapping happens after the
    sampler's own feasibility handling.

    Args:
        acqf: Acquisition function used to rank the candidate cloud
        bounds: Optimization-region bounds of shape (2, n_dims)
        spec: Supplies numeric-discrete grids for canonical matching
        x_avoid: Canonical (grid-snapped) avoided points
        sample_count: Number of candidates to draw
        random_seed: Sampler seed; ``None`` uses the fixed default seed
        domain_bounds: Full campaign bounds used for canonical validation
            and exhaustion judgement; defaults to ``bounds``
        inequality_constraints: BoTorch linear inequality constraints
        equality_constraints: BoTorch linear equality constraints

    Returns:
        Tuple of (candidates, acquisition_values), each of length one.

    Raises:
        SearchSpaceExhaustedError: If the spec is a fully enumerable
            numeric-discrete grid and every grid point is avoided or
            infeasible — proven exhaustion of a finite space.
        RuntimeError: If every *sampled* candidate matches an avoided point.
            The space may still contain unseen points; the sample simply
            failed to cover one.
    """
    domain = domain_bounds if domain_bounds is not None else bounds
    grid = enumerate_numeric_discrete_grid(spec)
    if grid is not None:
        choices = _enumerated_unseen_choices(
            grid, bounds, domain, x_avoid, inequality_constraints, equality_constraints
        )
    else:
        choices = _sampled_unseen_choices(
            bounds,
            domain_bounds=domain,
            spec=spec,
            x_avoid=x_avoid,
            sample_count=sample_count,
            random_seed=random_seed,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )

    with torch.no_grad():
        values = acqf(choices.unsqueeze(-2)).reshape(-1)
    best_idx = int(values.argmax().item())
    logger.warning(
        "All continuous acquisition restarts matched previously evaluated or "
        "pending points; selected the best of %d unseen fallback candidates.",
        int(choices.shape[0]),
    )
    return choices[best_idx : best_idx + 1], values[best_idx : best_idx + 1]


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
