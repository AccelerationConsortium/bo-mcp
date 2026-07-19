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
import warnings
from typing import Any

import numpy as np
import torch

from bo_engine.constants import (
    MAX_RANDOM_SEED,
    resolve_initial_design_size,
)

# Re-exports — initial-design helpers live in :mod:`bo_engine.initial_design`
# after the suggestions god-module split. Imports from this module continue
# to resolve here so external callers do not need to change.
from bo_engine.device import fork_rng_devices
from bo_engine.initial_design import (
    _apply_constraints_to_samples,
    _guard_categorical_space_exhaustion,
    generate_initial_design,
    generate_initial_design_indexed,
)
from bo_engine.reproducibility import GLOBAL_RNG_LOCK, derive_seed, draw_fallback_seed
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
    _resolve_noise_prior,
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
    "InitialDesignGenerationError",
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


class InitialDesignGenerationError(RuntimeError):
    """Raised when a non-exhaustible initial-design path produces no point."""


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
    3. Falls back to :func:`bo_engine.reproducibility.draw_fallback_seed`
       only when neither is supplied; this path is documented as
       non-reproducible and never touches the process-global RNG stream.
    """
    if rng is not None:
        return int(rng.integers(0, MAX_RANDOM_SEED))
    if spec.random_seed is not None:
        return derive_seed(spec.random_seed, f"acquisition:iter_{iteration}")
    # Deliberately non-reproducible — no seed was supplied. Used in tests
    # and ad-hoc campaigns where reproducibility is not required.
    return draw_fallback_seed()


def _build_initial_design_suggestions(
    spec: OptimizationSpec,
    batch_size: int,
    *,
    iteration: int,
    random_seed: int,
    observations: list[ObservationData],
    pending_points: list[dict[str, Any]] | None,
    initial_design_history: list[dict[str, Any]] | None,
    sobol_cursor: int | None = None,
) -> list[SuggestionResult]:
    """Build the Sobol initial-design batch for the warm-up phase.

    The Sobol continuation *offset* is chosen with the precedence
    ``sobol_cursor`` → ``len(initial_design_history)`` → ``len(observations) +
    len(pending)``. The persisted ``sobol_cursor`` (a raw Sobol index) is the
    authoritative source: the issue-count fallbacks only approximate consumed
    positions and, under a tight constraint where each candidate costs many raw
    draws, would rescan the same window every call and eventually stall. Each
    returned suggestion is stamped with its raw ``sobol_index`` so the caller can
    persist ``max(sobol_index) + 1`` as the next cursor.

    The exclusion set is *always* the union of the issuance history, the current
    observations, and the actionable pending points, so a point that has already
    been observed or is in flight — including a manually imported / freestanding
    result that never drew a Sobol position, and therefore is not in the history —
    is never re-issued. See :func:`generate_next_batch` for the full contract.
    """
    pending = pending_points or []
    observed = [obs.parameter_values for obs in observations]
    if initial_design_history is not None:
        excluded = [*initial_design_history, *observed, *pending]
        history_offset = len(initial_design_history)
    else:
        # Direct-caller fallback: approximate consumed positions from current
        # state (pending initial-design points have consumed positions too).
        excluded = [*observed, *pending]
        history_offset = len(observations) + len(pending)

    # The persisted raw cursor wins when present; otherwise fall back to the
    # issue-count approximation.
    n_drawn = sobol_cursor if sobol_cursor is not None else history_offset

    designs, positions = generate_initial_design_indexed(
        spec,
        batch_size,
        n_drawn=n_drawn,
        excluded_points=excluded,
    )
    # Contract: an *empty* return is a hard failure (a non-exhaustible path
    # produced nothing to suggest). A *short* batch is not — for continuous /
    # mixed spaces the space is effectively infinite, and generate_initial_design
    # already logs a warning and returns the smaller batch; purely-categorical
    # exhaustion raises SearchSpaceExhaustedError upstream instead of returning
    # empty.
    if not designs:
        msg = (
            "Continuous or mixed initial-design generation returned no points for "
            f"a batch of {batch_size} after excluding {len(excluded)} issued points."
        )
        raise InitialDesignGenerationError(msg)

    return [
        SuggestionResult(
            parameter_values=design,
            iteration=iteration,
            batch_index=i,
            generation_method="initial_design",
            random_seed=random_seed,
            sobol_index=positions[i],
            explanation=(
                f"Initial design point {i + 1}/{len(designs)} using Sobol sequence. "
                "Initial designs explore the parameter space before "
                "model-guided suggestions."
            ),
        )
        for i, design in enumerate(designs)
    ]


def generate_next_batch(
    spec: OptimizationSpec,
    observations: list[ObservationData],
    batch_size: int | None = None,
    iteration: int = 0,
    turbo_state: TurboState | None = None,
    rng: np.random.Generator | None = None,
    pending_points: list[dict[str, Any]] | None = None,
    initial_design_history: list[dict[str, Any]] | None = None,
    sobol_cursor: int | None = None,
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
        pending_points: Genuinely actionable in-flight suggestions
            (parameter-value dicts) that have not yet been observed.
            Forwarded to the acquisition optimizer as ``X_pending`` so
            parallel / batch BO conditions new candidates on the in-flight
            batch instead of silently clustering around it. During the
            initial-design phase they are additionally used as the Sobol
            continuation/exclusion source **only when**
            ``initial_design_history`` is not supplied (direct-caller
            fallback); a server that tracks issuance passes the authoritative
            history separately (see below) and keeps ``pending_points``
            strictly actionable.
        initial_design_history: Parameter-value dicts of *every* initial-design
            point already issued for this campaign — observed, actionable,
            rejected, expired, stale, and even soft-deleted rows. Its length is a
            monotonic initial-design *issue count*, used only as a legacy
            continuation approximation when no persisted ``sobol_cursor`` exists.
            It is not necessarily the raw Sobol position: exclusions, constraint
            projection, and finite/mixed-space deduplication can consume multiple
            raw draws for one issued point.
            Together with ``observations`` and ``pending_points`` it forms the
            exclusion set, so no issued or observed point is re-issued. It remains
            the durable audit/exclusion set even when ``sobol_cursor`` supplies the
            offset. ``None`` recovers the legacy behavior of deriving the offset
            from ``observations`` + ``pending_points``.
        sobol_cursor: Persisted raw Sobol continuation index (the authoritative
            offset). When supplied it overrides the issue-count derived from
            ``initial_design_history``: the accumulator can consume many raw
            positions to yield one candidate under a tight constraint, so resuming
            from the raw cursor — rather than the issued-point *count* — is what
            stops a repeatedly-rejected constrained campaign from rescanning the
            same window and permanently stalling. Each returned initial-design
            :class:`SuggestionResult` carries its ``sobol_index``; the backend
            persists ``max(sobol_index) + 1`` as the next cursor. ``None`` (legacy
            state / first call) falls back to the issue count.

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
        ``torch.random.fork_rng(devices=fork_rng_devices())`` block so
        concurrent callers (e.g. under ``asyncio.to_thread``) cannot
        race on the process-wide seed: the prior RNG state is saved on
        entry and restored on every return path. ``fork_rng_devices()``
        adds the active CUDA device when one is selected, because
        ``manual_seed`` also seeds the CUDA generators — a bare
        ``devices=[]`` would leak that mutation past the block on GPU.

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
    # fork_rng_devices() includes the active CUDA device so manual_seed's
    # CUDA-generator mutation is restored too (a bare devices=[] snapshots
    # only the CPU generator and would leak on GPU). GLOBAL_RNG_LOCK
    # serializes this section against every other snapshot/restore consumer
    # of the process-global RNG (other BoTorch calls, Thompson sampling, the
    # BayBE backend's seeded scope): without it an overlapping fork_rng
    # restore rolls a concurrent seeded stream back to a stale snapshot.
    with GLOBAL_RNG_LOCK, torch.random.fork_rng(devices=fork_rng_devices()):
        torch.manual_seed(random_seed)

        # Short-circuit for finite (purely-categorical) spaces whose unique
        # combinations are already exhausted — neither the initial-design
        # fallback nor the discrete acquisition optimizer can invent new
        # points, and BoTorch's optimize_acqf_discrete will otherwise raise
        # an opaque error when X_avoid covers the entire choice set.
        _guard_categorical_space_exhaustion(spec, observations, batch_size)

        # If not enough data, fall back to initial design. See
        # resolve_initial_design_size for the floor this applies.
        min_data = resolve_initial_design_size(spec.n_parameters, spec.initial_design_size)
        if len(observations) < min_data:
            suggestions = _build_initial_design_suggestions(
                spec,
                batch_size,
                iteration=iteration,
                random_seed=random_seed,
                observations=observations,
                pending_points=pending_points,
                initial_design_history=initial_design_history,
                sobol_cursor=sobol_cursor,
            )
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
