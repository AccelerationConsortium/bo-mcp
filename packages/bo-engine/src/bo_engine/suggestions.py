"""Suggestion generation for Bayesian Optimization.

Supports both single-objective and multi-objective optimization.

v1.0.1: Added automatic single-objective detection and handling
v1.1: Added acquisition method selection and input warping support
v1.2: Added TuRBO integration for high-dimensional optimization
v1.3: Added outcome constraints and cost-aware optimization
v2.3: Added GPU auto-detection and acceleration

The implementation has been split across companion modules so each file
owns one concern and stays well under the 1k LOC cognitive-load ceiling:

* :mod:`bo_engine.initial_design` — Sobol initial-design draws,
  exclusion / deduplication, and the purely-categorical exhaustion
  guard. (Pre-existing split.)
* :mod:`bo_engine.suggestions_training` — training-data assembly
  helpers (``_prepare_training_data``, ``_prepare_train_yvar``,
  ``_prepare_cost_data``) and pending-point encoding.
* :mod:`bo_engine.suggestions_outcome_constraints` — outcome-constraint
  GP construction plus the
  :class:`OutcomeConstraintConfigurationError` typed error.
* :mod:`bo_engine.suggestions_single_objective` — single-objective
  acquisition pipeline (model fit, TuRBO trust region, provenance).
* :mod:`bo_engine.suggestions_multi_objective` — multi-objective
  acquisition pipeline (hypervolume improvement, Pareto provenance).
* :mod:`bo_engine.suggestions_common` — prediction-extraction and
  confidence-level helpers shared by both batch builders.

This module retains the public dispatcher (:func:`generate_next_batch`),
the seed / noise-prior resolution helpers, the TuRBO post-evaluation
update, and the typed exceptions. Private helper symbols that the test
suite imports from ``bo_engine.suggestions`` continue to resolve here
through explicit re-exports below.
"""

from __future__ import annotations

import logging
import random
import warnings
from typing import Any

import numpy as np
import torch
from gpytorch.priors import GammaPrior

from bo_engine.constants import (
    MAX_RANDOM_SEED,
    MIN_OBSERVATIONS_FOR_MODEL,
)

# Re-exports — initial-design helpers live in :mod:`bo_engine.initial_design`
# after the suggestions god-module split. Imports from this module continue
# to resolve here so external callers do not need to change.
from bo_engine.initial_design import (
    _apply_constraints_to_samples,
    _guard_categorical_space_exhaustion,
    generate_initial_design,
)
from bo_engine.reproducibility import derive_seed
from bo_engine.suggestions_common import (
    _extract_scalar_prediction,
    _get_confidence_level,
    _get_model_predictions,
)
from bo_engine.suggestions_multi_objective import (
    _build_multi_objective_explanation,
    _build_multi_objective_provenance,
    _create_multi_objective_suggestions,
    _generate_multi_objective_batch,
)
from bo_engine.suggestions_outcome_constraints import (
    OutcomeConstraintConfigurationError,
    _build_outcome_constraint_models,
    _collect_constraint_values,
    _fit_outcome_constraint_model,
)
from bo_engine.suggestions_single_objective import (
    _build_single_objective_provenance,
    _compute_turbo_bounds,
    _create_single_objective_suggestions,
    _generate_single_objective_batch,
    _initialize_turbo_state,
)
from bo_engine.suggestions_training import (
    _encode_pending_points,
    _prepare_cost_data,
    _prepare_train_yvar,
    _prepare_training_data,
)
from bo_engine.transforms import get_bounds_tensor
from bo_engine.turbo import (
    TurboState,
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


__all__ = [
    # Public surface
    "MultiFidelityNotSupportedError",
    "OutcomeConstraintConfigurationError",
    "SAASBONotSupportedError",
    "TransferLearningNotSupportedError",
    # Re-exported private helpers — test suite imports these by name.
    "_apply_constraints_to_samples",
    "_build_multi_objective_explanation",
    "_build_multi_objective_provenance",
    "_build_outcome_constraint_models",
    "_build_single_objective_provenance",
    "_collect_constraint_values",
    "_compute_turbo_bounds",
    "_create_multi_objective_suggestions",
    "_create_single_objective_suggestions",
    "_encode_pending_points",
    "_extract_scalar_prediction",
    "_fit_outcome_constraint_model",
    "_generate_multi_objective_batch",
    "_generate_single_objective_batch",
    "_get_confidence_level",
    "_get_model_predictions",
    "_guard_categorical_space_exhaustion",
    "_initialize_turbo_state",
    "_prepare_cost_data",
    "_prepare_train_yvar",
    "_prepare_training_data",
    "_resolve_acquisition_seed",
    "_resolve_noise_prior",
    "generate_initial_design",
    "generate_next_batch",
    "update_turbo_after_evaluation",
]


class MultiFidelityNotSupportedError(ValueError):
    """Raised when a spec requests multi-fidelity dispatch.

    The ``bo_engine.multifidelity`` module exposes standalone helpers, but
    the active suggestion pipeline does not route
    ``AcquisitionMethod.MULTI_FIDELITY_KG`` to BoTorch's qMFKG and does
    not build a ``SingleTaskMultiFidelityGP`` when
    ``spec.fidelity_parameter`` is set. Raising here keeps the
    advertisement honest — silent downgrade to single-fidelity would let
    callers think they were getting cost-amortized exploration when they
    were not.
    """


class SAASBONotSupportedError(ValueError):
    """Raised when a spec requests SAASBO dispatch.

    The ``bo_engine.saasbo`` module exposes standalone helpers
    (``create_and_fit_saasbo_model``, ``generate_saasbo_suggestions``),
    but the active suggestion pipeline does not consume
    ``spec.saasbo_config`` — a spec carrying it would silently run a
    plain dense-ARD ``SingleTaskGP`` instead of the sparsity-inducing
    SAAS priors the caller asked for. Raising here keeps the
    advertisement honest, mirroring the multi-fidelity precedent above.
    """


class TransferLearningNotSupportedError(ValueError):
    """Raised when a spec requests RGPE transfer-learning dispatch.

    The ``bo_engine.transfer_learning`` module exposes standalone RGPE
    helpers (``create_rgpe_model``, ``generate_rgpe_suggestions``), but
    the active suggestion pipeline does not consume
    ``spec.transfer_learning`` — a spec carrying it would silently run
    an ordinary GP with no prior-campaign transfer at all. Raising here
    keeps the advertisement honest, following the same rule as the
    multi-fidelity and SAASBO rejections above. (BayBE's native
    ``TaskParameter`` mechanism remains the supported campaign-level
    transfer path.)
    """


def _resolve_noise_prior(spec: OptimizationSpec) -> GammaPrior | None:
    """Build a :class:`GammaPrior` from ``spec.noise_prior_params`` when set.

    Returns ``None`` when the spec does not override the prior so the
    model factory falls back to its ``_default_noise_prior`` (calibrated
    for unit-standardized targets). The override is only consulted on the
    trainable-noise path; the ``FixedNoiseGaussianLikelihood`` path bypasses
    the prior entirely.
    """
    if spec.noise_prior_params is None:
        return None

    concentration, rate = spec.noise_prior_params
    if concentration <= 0 or rate <= 0:
        msg = (
            "noise_prior_params must be positive (concentration, rate); "
            f"got {(concentration, rate)}."
        )
        raise ValueError(msg)
    return GammaPrior(float(concentration), float(rate))


def _resolve_acquisition_seed(
    spec: OptimizationSpec,
    iteration: int,
    rng: np.random.Generator | None,
) -> int:
    """Derive the acquisition seed for :func:`generate_next_batch`.

    Precedence:

    1. ``rng`` wins whenever supplied so external callers keep control
       of their RNG pipeline.
    2. ``spec.random_seed`` flows through
       :func:`bo_engine.reproducibility.derive_seed` with role tag
       ``"acquisition:iter_{iteration}"`` so two independent replays of the
       same campaign state produce identical acquisition candidates and
       the seed for any other phase at the same iteration (Sobol initial
       design, MCMC chain) cannot collide with it.
    3. Falls back to :func:`random.randint` only when neither is
       supplied; this path is documented as non-reproducible.
    """
    if rng is not None:
        return int(rng.integers(0, MAX_RANDOM_SEED))
    if spec.random_seed is not None:
        return derive_seed(spec.random_seed, f"acquisition:iter_{iteration}")
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
           deterministically via
           :func:`bo_engine.reproducibility.derive_seed` with role tag
           ``"acquisition:iter_{iteration}"``. Two independent calls on
           the same campaign state at the same iteration therefore
           produce identical acquisition candidates, and the per-phase
           role tag prevents the Sobol initial design from sharing the
           same seed as the acquisition multi-start.
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

    # Multi-fidelity routing is not implemented end-to-end in this pipeline.
    # Reject loudly rather than silently downgrading to single-fidelity —
    # the latter is what the audit specifically called out as misleading
    # advertisement (cf. ``BoTorchBackend.supported_features``).
    if (
        spec.fidelity_parameter is not None
        or spec.acquisition_method == AcquisitionMethod.MULTI_FIDELITY_KG
    ):
        msg = (
            "Multi-fidelity dispatch (qMFKG / SingleTaskMultiFidelityGP) is "
            "not wired into generate_next_batch. Remove "
            "spec.fidelity_parameter and acquisition_method="
            "MULTI_FIDELITY_KG to proceed with single-fidelity BO; the "
            "standalone helpers in bo_engine.multifidelity remain available "
            "for direct multi-fidelity workflows."
        )
        raise MultiFidelityNotSupportedError(msg)

    # SAASBO routing is not implemented end-to-end either: the pipeline
    # never consumes spec.saasbo_config, so accepting it would silently
    # fit a dense-ARD SingleTaskGP while the caller believes they get
    # sparsity-inducing SAAS priors. Same honesty rule as multi-fidelity.
    if spec.saasbo_config is not None:
        msg = (
            "SAASBO dispatch (SaasFullyBayesianSingleTaskGP / NUTS) is not "
            "wired into generate_next_batch. Remove spec.saasbo_config to "
            "proceed with a standard GP; the standalone helpers in "
            "bo_engine.saasbo (generate_saasbo_suggestions) remain "
            "available for direct SAASBO workflows."
        )
        raise SAASBONotSupportedError(msg)

    # RGPE transfer learning is likewise not wired: prior-campaign data is
    # never loaded into PriorTaskData and no dispatch to
    # generate_rgpe_suggestions exists, so accepting the option would run
    # a plain GP while the caller believes prior knowledge is transferred.
    if spec.transfer_learning is not None:
        msg = (
            "RGPE transfer-learning dispatch is not wired into "
            "generate_next_batch. Remove spec.transfer_learning to proceed "
            "without prior-campaign transfer, use BayBE's native "
            "TaskParameter mechanism, or drive the standalone helpers in "
            "bo_engine.transfer_learning (generate_rgpe_suggestions) "
            "directly."
        )
        raise TransferLearningNotSupportedError(msg)

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

        noise_prior = _resolve_noise_prior(spec)
        if is_single_objective:
            return _generate_single_objective_batch(ctx, noise_prior=noise_prior)

        # TuRBO not supported for multi-objective
        if turbo_state is not None or spec.use_turbo:
            warnings.warn(
                "TuRBO is designed for single-objective optimization. "
                "It will be ignored for multi-objective problems. "
                "Standard L-BFGS-B optimization will be used instead.",
                UserWarning,
                stacklevel=2,
            )
        suggestions = _generate_multi_objective_batch(ctx, noise_prior=noise_prior)
        return suggestions, None


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

    from bo_engine.device import get_device, get_dtype

    y_tensor = torch.tensor(y_values, dtype=get_dtype(), device=get_device())
    return update_turbo_state(turbo_state, y_tensor, minimize=minimize)
