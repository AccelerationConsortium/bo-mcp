"""BayBE backend implementing the BOBackend protocol.

Maximizes use of BayBE-native functionality:
- Campaign.recommend() for suggestions
- Campaign.posterior_stats() for model predictions
- Campaign.acquisition_values() for per-suggestion acquisition values
- Campaign.get_surrogate().to_botorch() for model hyperparameter extraction
- Campaign.to_json()/from_json() for state persistence (Approach B)
- ParetoObjective for multi-objective optimization
- allow_recommending_already_measured=False to prevent re-suggestions
- Native pending_experiments support to prevent re-recommending in-flight points

Delegates to bo_engine only for what BayBE doesn't provide:
hypervolume computation, near-duplicate detection, batch diversity metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import pandas as pd
import pydantic
import torch
from baybe import Campaign
from baybe import __version__ as baybe_version
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpaceType
from bo_engine.backend import (
    Feature,
    SuggestionBatch,
)
from bo_engine.backend_base import (
    BackendValidationResult,
    BaseBackend,
    CapabilityReport,
    CapabilityStatus,
    required_features,
)
from bo_engine.device import get_device, get_dtype
from bo_engine.diagnostics import (
    compute_best_value,
    compute_improvement_history,
    compute_single_objective_improvement_rate,
    summarize_pareto_front,
)
from bo_engine.diagnostics import (
    compute_hypervolume as engine_compute_hypervolume,
)
from bo_engine.diagnostics import (
    compute_pareto_front as engine_compute_pareto_front,
)
from bo_engine.progress import ProgressCallback, ProgressEvent, emit
from bo_engine.reference_point import get_reference_point
from bo_engine.result_validation import (
    detect_outliers,
)
from bo_engine.transforms import encode_categorical, get_bounds_tensor
from bo_engine.types import (
    ObservationData,
    OptimizationSpec,
    ParameterType,
)

from bo_engine_baybe.converters import (
    baybe_constraint_support,
    dataframe_to_suggestions,
    observations_to_dataframe,
    pending_points_to_dataframe,
    spec_to_objective,
    spec_to_searchspace,
)
from bo_engine_baybe.options import (
    BayBEParameterOptions,
    BayBEParameterRole,
    extract_baybe_backend_options,
    extract_baybe_parameter_options,
)

logger = logging.getLogger(__name__)

# Version of this backend package. Imported lazily into ``method_info`` so the
# backend module does not need to import ``bo_engine_baybe.__init__`` (which
# would create a circular import via ``__init__`` re-exporting BayBEBackend).
_BO_ENGINE_BAYBE_VERSION = "0.1.0"


def _detect_chemistry_extras() -> tuple[bool, str | None]:
    """Probe whether BayBE's optional chemistry extras are installed.

    BayBE's :class:`SubstanceParameter` requires ``baybe[chem]`` (which
    pulls in ``scikit-fingerprints`` etc.) before any descriptor table
    can be built. The plain ``baybe`` install advertises the class but
    raises :class:`OptionalImportError` at construction time. Probing
    once at import lets :meth:`BayBEBackend.validate_capabilities`
    surface the gap as an ``UNSUPPORTED`` report at intake instead of a
    deferred crash during suggestion generation.
    """
    try:
        import baybe._optional.chem  # noqa: F401 — import side effect only
    except ImportError as e:
        return False, str(e)
    return True, None


_CHEMISTRY_AVAILABLE, _CHEMISTRY_UNAVAILABLE_REASON = _detect_chemistry_extras()

_SUPPORTED_FEATURES = frozenset(
    {
        Feature.MULTI_OBJECTIVE,
        Feature.CONSTRAINTS,
        Feature.CATEGORICAL,
        Feature.MIXED_SEARCH_SPACE,
        Feature.TRANSFER_LEARNING,
    }
)


# BayBE silently drops these BoTorch-only knobs at runtime. Each is
# semantically load-bearing: dropping outcome_constraints changes the
# feasibility region, dropping turbo_config disables the TuRBO trust
# region, etc. Reporting them as plain ``IGNORED`` makes the campaign
# accept silently and run with the wrong semantics, which is the worst
# class of BO bug — wrong answers that look fine. We therefore classify
# them as ``requires_acknowledgement``: by default the feature *and*
# option reports emit ``UNSUPPORTED`` so create-time capability
# enforcement rejects the spec. Callers that have weighed the trade-off
# can opt in by listing the field name in
# :attr:`OptimizationSpec.acknowledge_degradations`; the reports
# downgrade to ``IGNORED`` for those entries and the campaign accepts
# with a prominent warning. ``backend="auto"`` continues to prefer
# backends that need no acknowledgement (``FULL`` tier in the selector
# at :func:`bo_mcp_server.backend.resolve_backend_name`).
# ``transfer_learning`` is intentionally absent — its feature-level
# routing is decided by :meth:`BayBEBackend._transfer_learning_report`
# (TaskParameter ⇒ SUPPORTED, RGPE config ⇒ UNSUPPORTED).
_BAYBE_DEGRADABLE_FEATURE_MAP: dict[str, tuple[Feature, ...]] = {
    "turbo_config": (Feature.HIGH_DIMENSIONAL,),
    "saasbo_config": (Feature.HIGH_DIMENSIONAL,),
    "fidelity_parameter": (Feature.MULTI_FIDELITY,),
    "use_cost_aware": (Feature.COST_AWARE,),
    "use_input_warping": (Feature.INPUT_WARPING,),
    "outcome_constraints": (Feature.OUTCOME_CONSTRAINTS,),
}
_BAYBE_DEGRADABLE_FEATURES: frozenset[Feature] = frozenset(
    f for features in _BAYBE_DEGRADABLE_FEATURE_MAP.values() for f in features
)


def _active_attrs_for_feature(spec: OptimizationSpec, feature: Feature) -> tuple[str, ...]:
    """Return spec attribute(s) actually set on ``spec`` that activate ``feature``.

    Several features in :data:`_BAYBE_DEGRADABLE_FEATURE_MAP` are
    activated by more than one attribute — ``HIGH_DIMENSIONAL`` is
    activated by both ``turbo_config`` and ``saasbo_config``. A naive
    reverse lookup would always name the first entry, so a caller who
    only set ``saasbo_config`` would be told to acknowledge
    ``turbo_config`` instead. Filter the candidates by what is actually
    set on the spec so the diagnostic targets the real culprit.
    """
    active: list[str] = []
    for attr, features in _BAYBE_DEGRADABLE_FEATURE_MAP.items():
        if feature not in features:
            continue
        value = getattr(spec, attr, None)
        is_set = bool(value) if not isinstance(value, list) else len(value) > 0
        if is_set:
            active.append(attr)
    return tuple(active)


_MIN_OBSERVATIONS_FOR_CONFIDENCE = 5
_MODEL_TYPE_SINGLE = "BayBE GP"
_MODEL_TYPE_MULTI = "BayBE GP (CompositeSurrogate)"
_FALLBACK_ACQ_SINGLE = "qLogNoisyExpectedImprovement"
_FALLBACK_ACQ_MULTI = "qLogNoisyExpectedHypervolumeImprovement"
_FALLBACK_LABEL = "(fallback)"

# Minimum observations before fitting a model for diagnostics
_MIN_DATA_ABSOLUTE = 3
_MIN_DATA_PARAM_MULTIPLIER = 2

# Backend-state schema versions. v1 = bare {campaign_json}; v2 adds the
# stable observation identity index introduced for TODO 1.63.
_STATE_SCHEMA_VERSION_LEGACY = 1
_STATE_SCHEMA_VERSION_IDENTITY = 2

# BayBE-specific exceptions that must be caught alongside standard ones.
# IncompatibilityError is raised when the recommender phase (random vs BO)
# doesn't provide the requested API (e.g. posterior_stats during random phase).
_BAYBE_SAFE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
)

try:
    from baybe.exceptions import IncompatibilityError

    _BAYBE_SAFE_EXCEPTIONS = (*_BAYBE_SAFE_EXCEPTIONS, IncompatibilityError)
except ImportError:
    pass  # Older BayBE versions may not expose this


# ---------------------------------------------------------------------------
# Campaign construction & state management
# ---------------------------------------------------------------------------


def _build_campaign(spec: OptimizationSpec) -> Campaign:
    """Create a fresh BayBE Campaign from an OptimizationSpec."""
    searchspace = spec_to_searchspace(spec)
    is_purely_discrete = searchspace.type == SearchSpaceType.DISCRETE
    options = extract_baybe_backend_options(spec.backend_options)

    switch_after = options.recommender.switch_after if options.recommender else 1
    kwargs: dict[str, object] = {
        "searchspace": searchspace,
        "objective": spec_to_objective(spec),
        "recommender": TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(),
            switch_after=switch_after,
        ),
    }

    if is_purely_discrete:
        kwargs["allow_recommending_already_measured"] = False
        kwargs["allow_recommending_already_recommended"] = False
        kwargs["allow_recommending_pending_experiments"] = (
            options.allow_recommending_pending_experiments
        )

    return Campaign(**kwargs)  # ty: ignore[invalid-argument-type]


def _restore_or_build_campaign(
    spec: OptimizationSpec,
    backend_state: dict[str, Any] | None,
) -> Campaign:
    """Restore a Campaign from serialized state, or build a fresh one."""
    if backend_state and "campaign_json" in backend_state:
        try:
            return Campaign.from_json(backend_state["campaign_json"])
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("Failed to restore Campaign from state: %s, rebuilding fresh", e)
    return _build_campaign(spec)


def _observation_fingerprint(
    obs: ObservationData,
    param_names: list[str],
    obj_names: list[str],
) -> str:
    """Compute a stable identity hash for an observation.

    Uses sorted parameter and objective columns so reordering the input
    list (or shuffling the underlying DB query) produces the same hash.
    The hash is intentionally short (12 hex chars) — enough to make
    collisions astronomically unlikely for any realistic campaign and
    cheap to compare during reconciliation.
    """
    payload = {
        "params": [(name, obs.parameter_values.get(name)) for name in param_names],
        "objectives": [(name, obs.objective_values.get(name)) for name in obj_names],
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()


def _build_observation_identity(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> list[str]:
    """Return one fingerprint per observation in input order."""
    param_names = [p.name for p in spec.parameters]
    obj_names = [o.name for o in spec.objectives]
    return [_observation_fingerprint(o, param_names, obj_names) for o in observations]


def _serialize_campaign(
    campaign: Campaign,
    observation_identity: list[str],
) -> dict[str, Any]:
    """Serialize a Campaign and identity index for storage as backend_state.

    The identity index is the list of fingerprints for the measurements
    BayBE believes it has. On restore, the next call compares its incoming
    observation fingerprints to this list and adds only truly unseen rows
    (TODO 1.63). Schema version bumps to ``_STATE_SCHEMA_VERSION_IDENTITY``.
    """
    return {
        "schema_version": _STATE_SCHEMA_VERSION_IDENTITY,
        "campaign_json": campaign.to_json(),
        "observation_identity": list(observation_identity),
    }


def _reconcile_measurements(
    campaign: Campaign,
    spec: OptimizationSpec,
    observations: list[ObservationData],
    backend_state: dict[str, Any] | None,
) -> tuple[Campaign, list[str]]:
    """Add new measurements to a restored campaign via stable identity (TODO 1.63).

    Returns ``(campaign, identity_index)`` — the campaign may be a freshly
    built replacement when the stored identity index references rows that
    have since been deleted from storage. Reconciliation rules:

    * Compute a fingerprint for each observation in the current
      ``observations`` list.
    * Fingerprints are reconciled as a **multiset**, not a set, so two
      observations with identical parameter/objective values (real
      replicates) both stay in the campaign.
    * Stored identities that no longer appear in storage trigger a
      rebuild — the user deleted or rewrote rows, so the BayBE campaign
      must not keep training on data the server no longer owns.
    * Storage emptying out (``observations=[]``) is also a rebuild
      trigger: the restored campaign has measurements the source of
      truth no longer has.
    * Legacy payloads (no identity index) cannot be reconciled safely —
      the previous count-prefix logic was the bug 1.63 set out to fix.
      The migration path is therefore to rebuild from current
      observations and start tracking identities from this call onwards.
    """
    from collections import Counter

    incoming_ids = _build_observation_identity(spec, observations)
    incoming_counts: Counter[str] = Counter(incoming_ids)
    stored_ids = list(_extract_stored_identity(backend_state))
    stored_counts: Counter[str] = Counter(stored_ids)
    has_identity_field = _has_stored_identity_field(backend_state)

    if not has_identity_field and _campaign_has_measurements(campaign):
        logger.info(
            "Restored BayBE state has no identity index; rebuilding campaign from "
            "%d current observation(s).",
            len(observations),
        )
        return _rebuild_from_observations(spec, observations), incoming_ids

    # A stored multiplicity that exceeds the incoming one means a measurement
    # the campaign believes it has is no longer in storage — rebuild.
    if any(stored_counts[k] > incoming_counts[k] for k in stored_counts):
        missing = sum(max(stored_counts[k] - incoming_counts[k], 0) for k in stored_counts)
        logger.warning(
            "Restored BayBE campaign references %d measurement(s) no longer present "
            "in storage; rebuilding from current observations.",
            missing,
        )
        return _rebuild_from_observations(spec, observations), incoming_ids

    # Add the multiplicity delta per fingerprint, preserving observation order
    # so each replicate produces a distinct BayBE row.
    remaining: Counter[str] = stored_counts.copy()
    new_observations: list[ObservationData] = []
    for fingerprint, obs in zip(incoming_ids, observations, strict=True):
        if remaining[fingerprint] > 0:
            remaining[fingerprint] -= 1
            continue
        new_observations.append(obs)

    if new_observations:
        obs_df = observations_to_dataframe(new_observations, spec)
        campaign.add_measurements(obs_df)

    return campaign, incoming_ids


def _has_stored_identity_field(backend_state: dict[str, Any] | None) -> bool:
    """Return True when the payload includes the v2 ``observation_identity`` key."""
    if not backend_state:
        return False
    return "observation_identity" in backend_state


def _campaign_has_measurements(campaign: Campaign) -> bool:
    """Defensive check for whether the restored campaign carries any data."""
    try:
        return len(campaign.measurements) > 0
    except (AttributeError, TypeError):
        return False


def _rebuild_from_observations(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> Campaign:
    """Build a fresh campaign and add all current observations.

    Used as the safe fallback whenever identity reconciliation can no
    longer trust the restored campaign (legacy payload, missing rows).
    """
    fresh = _build_campaign(spec)
    if observations:
        obs_df = observations_to_dataframe(observations, spec)
        fresh.add_measurements(obs_df)
    return fresh


def _extract_stored_identity(backend_state: dict[str, Any] | None) -> list[str]:
    """Pull the identity index out of a stored payload.

    Supports both the new schema (``observation_identity`` field, v2) and
    the legacy schema (no identity field, v1). For v1 payloads the
    reconciliation falls back to ``len(campaign.measurements)`` interpretation
    via an empty identity list (treated as "no prior identities recorded").
    """
    if not backend_state:
        return []
    raw = backend_state.get("observation_identity")
    if not isinstance(raw, list):
        return []
    return [str(v) for v in raw]


def _observations_to_minimization_tensor(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> torch.Tensor:
    """Convert observations to a tensor in BoTorch minimization convention."""
    obj_names = [o.name for o in spec.objectives]
    minimize_mask = torch.tensor([o.minimize for o in spec.objectives], dtype=torch.bool)

    y_list = [
        torch.tensor([obs.objective_values[n] for n in obj_names], dtype=torch.double)
        for obs in observations
    ]
    y_tensor = torch.stack(y_list)
    y_bo = y_tensor.clone()
    y_bo[:, ~minimize_mask] = -y_bo[:, ~minimize_mask]
    return y_bo


def _compute_reference_point(y_bo: torch.Tensor) -> torch.Tensor:
    """Compute a reference point from minimization-convention objective values."""
    worst = y_bo.max(dim=0).values
    ranges = worst - y_bo.min(dim=0).values
    abs_scale = worst.abs().clamp(min=1e-6)
    ranges = torch.where(ranges < 1e-6, abs_scale, ranges)
    return worst + 0.1 * ranges


# ---------------------------------------------------------------------------
# BayBE-native extraction helpers
# ---------------------------------------------------------------------------


def _extract_posterior_stats(
    campaign: Campaign,
    rec_df: pd.DataFrame,
    spec: OptimizationSpec,
) -> tuple[list[dict[str, dict[str, float] | None]], str | None]:
    """Extract BayBE posterior mean and std per objective for each recommendation.

    Returns ``(predictions, warning)`` where ``warning`` is a short
    message explaining why posterior_stats was unavailable for the active
    recommender phase, or ``None`` when extraction succeeded. The warning
    is forwarded to :class:`SuggestionBatch.warnings` (TODO 1.67) so users
    see "no posterior in random-warmup phase" instead of a silent ``None``.
    """
    obj_names = [o.name for o in spec.objectives]
    empty: dict[str, dict[str, float] | None] = {
        "predicted_objectives": None,
        "predicted_std": None,
    }
    try:
        stats_df = campaign.posterior_stats(candidates=rec_df, stats=("mean", "std"))
        return (
            [_parse_prediction_row(stats_df.iloc[i], obj_names) for i in range(len(rec_df))],
            None,
        )
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("Posterior stats extraction failed: %s", e)
        return [empty] * len(rec_df), str(e)


def _parse_prediction_row(
    row: pd.Series,  # type: ignore[type-arg]
    obj_names: list[str],
) -> dict[str, dict[str, float] | None]:
    """Parse a single row of posterior stats into predicted mean/std dicts."""
    pred_obj: dict[str, float] = {}
    pred_std: dict[str, float] = {}
    for name in obj_names:
        mean_col = f"{name}_mean"
        std_col = f"{name}_std"
        if mean_col in row.index:
            pred_obj[name] = float(row[mean_col])
        if std_col in row.index:
            pred_std[name] = float(row[std_col])
    return {
        "predicted_objectives": pred_obj or None,
        "predicted_std": pred_std or None,
    }


def _extract_acquisition_values(
    campaign: Campaign,
    rec_df: pd.DataFrame,
    pending_df: pd.DataFrame | None,
) -> tuple[list[float | None], str | None]:
    """Extract per-suggestion acquisition function values from BayBE.

    Pending experiments are forwarded so the acquisition values reflect
    the same conditioning BayBE used during ``recommend`` (TODO 1.62). A
    short warning message is returned alongside the values when
    extraction is impossible in the current recommender phase so the
    backend can surface it via :class:`SuggestionBatch.warnings`.
    """
    try:
        acq_series = campaign.acquisition_values(
            candidates=rec_df,
            pending_experiments=pending_df,
        )
        return [float(v) for v in acq_series.values], None
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("Acquisition value extraction failed: %s", e)
        return [None] * len(rec_df), str(e)


def _extract_model_info(
    campaign: Campaign,
) -> tuple[dict[str, str | list[float] | float | None], str | None]:
    """Extract model hyperparameters from BayBE's fitted surrogate.

    Returns ``(info, warning)`` mirroring :func:`_extract_posterior_stats`.
    Multi-target campaigns produce a ``ModelListGP``; in that case the
    first sub-model's hyperparameters are reported (BayBE composes one
    GP per target). The ``kernel_type`` key is always present so callers
    can distinguish "fitted" (string class name) from "not available"
    (``None``) — surfacing only the latter as a method warning (TODO 1.67).
    """
    info: dict[str, str | list[float] | float | None] = {
        "kernel_type": None,
    }
    try:
        surrogate = campaign.get_surrogate()
        botorch_model = surrogate.to_botorch()
        model_for_kernel = _select_kernel_model(botorch_model)
        covar = getattr(model_for_kernel, "covar_module", None)
        if covar is None:
            return info, "BayBE surrogate did not expose a covariance module"
        kernel = getattr(covar, "base_kernel", covar)
        info["kernel_type"] = type(kernel).__name__

        ls = kernel.lengthscale.detach().squeeze()  # ty: ignore[call-non-callable, unresolved-attribute]
        if ls.numel() == 1:
            info["lengthscales"] = [round(float(ls.item()), 4)]
        else:
            info["lengthscales"] = [round(float(v), 4) for v in ls.tolist()]

        if hasattr(model_for_kernel, "likelihood") and hasattr(
            model_for_kernel.likelihood, "noise"
        ):
            noise = model_for_kernel.likelihood.noise.item()  # ty: ignore[unresolved-attribute]
            info["noise_variance"] = round(float(noise), 6)
        if hasattr(covar, "outputscale"):
            oscale = covar.outputscale.item()  # ty: ignore[unresolved-attribute]
            info["output_scale"] = round(float(oscale), 4)
        return info, None
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("Model info extraction failed: %s", e)
        return info, str(e)


def _select_kernel_model(botorch_model: Any) -> Any:
    """Pick the BoTorch model whose hyperparameters we report.

    ``ModelListGP`` (used by multi-target/multi-objective BayBE campaigns)
    wraps one sub-model per target. We report the first sub-model so the
    metadata is non-empty and consistent across calls; the per-target
    breakdown can be reconstructed from BayBE's posterior_stats output
    when finer granularity is needed.
    """
    sub_models = getattr(botorch_model, "models", None)
    if sub_models is not None and len(sub_models) > 0:
        return sub_models[0]
    return botorch_model


def _extract_feature_importance(campaign: Campaign) -> dict[str, float] | None:
    """Extract SHAP-based feature importance from BayBE (optional)."""
    try:
        from baybe.insights.shap import SHAPInsight

        insight = SHAPInsight.from_campaign(campaign)
        values = insight.explanation.values  # ty: ignore[unresolved-attribute]
        feature_names = insight.explanation.feature_names  # ty: ignore[unresolved-attribute]
        if values is not None and feature_names is not None:
            mean_abs = [float(abs(v).mean()) for v in values.T]
            return dict(zip(feature_names, mean_abs, strict=False))
    except (ImportError, *_BAYBE_SAFE_EXCEPTIONS) as e:
        logger.debug("SHAP feature importance extraction failed: %s", e)
    return None


def _build_fitted_campaign(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> Campaign:
    """Build a Campaign, add measurements, and trigger model fitting.

    Used by diagnostics methods that need a fitted surrogate. A single
    call replaces the duplicate build-add-recommend pattern that was
    previously in both _compute_model_diagnostics and
    _compute_hyperparameter_diagnostics.
    """
    campaign = _build_campaign(spec)
    obs_df = observations_to_dataframe(observations, spec)
    campaign.add_measurements(obs_df)
    campaign.recommend(batch_size=1)  # triggers model fitting
    return campaign


def _active_recommender(campaign: Campaign):
    """Return the active non-meta recommender of the campaign, if exposed.

    BayBE's meta-recommender flow advances state during ``recommend``; the
    method-info path needs the recommender after that switch happened
    rather than the meta-recommender wrapper. Falls back to the raw
    ``campaign.recommender`` attribute when the campaign exposes no
    helper.
    """
    helper = getattr(campaign, "_get_non_meta_recommender", None)
    if callable(helper):
        try:
            return helper()
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("Could not resolve active BayBE recommender: %s", e)
    return getattr(campaign, "recommender", None)


def _active_acquisition_label(recommender: Any) -> str | None:
    """Read the acquisition function class name from a BayBE recommender, if any."""
    acq_fn = getattr(recommender, "acquisition_function", None)
    if acq_fn is None:
        return None
    return type(acq_fn).__name__


def _campaign_searchspace_label(campaign: Campaign) -> str:
    """Return the BayBE searchspace type as a stable string label."""
    searchspace_type = getattr(getattr(campaign, "searchspace", None), "type", None)
    if hasattr(searchspace_type, "value"):
        return str(searchspace_type.value)
    return str(searchspace_type)


def _strategy_and_model(
    rec_name: str | None,
    is_nonpredictive: bool,
    is_multi: bool,
) -> tuple[str, str]:
    """Compose the live ``(strategy, model_type)`` labels for method-info.

    Nonpredictive recommenders (e.g. random warmup) explicitly report
    ``"none (space-filling)"`` for the model type so the audit trail
    cannot claim a GP surrogate when none is fitted.
    """
    if is_nonpredictive:
        strategy = (
            f"{rec_name} (space-filling, no surrogate)"
            if rec_name
            else "RandomRecommender (space-filling initial design)"
        )
        return strategy, "none (space-filling)"
    surrogate = _MODEL_TYPE_MULTI if is_multi else _MODEL_TYPE_SINGLE
    if rec_name is not None:
        return f"{rec_name} (GP-based)", surrogate
    return "BotorchRecommender (GP-based)", surrogate


def _acquisition_label(
    recommender: Any,
    is_nonpredictive: bool,
    is_multi: bool,
) -> str:
    """Choose the acquisition-function label live from the recommender."""
    if is_nonpredictive:
        # Nonpredictive recommenders do not have an acquisition function;
        # claiming qLogNEI here would mislead audit metadata.
        return "none (space-filling)"
    live_acq = _active_acquisition_label(recommender)
    if live_acq is not None:
        return live_acq
    return (
        f"{_FALLBACK_ACQ_MULTI} {_FALLBACK_LABEL}"
        if is_multi
        else f"{_FALLBACK_ACQ_SINGLE} {_FALLBACK_LABEL}"
    )


# ---------------------------------------------------------------------------
# BayBEBackend
# ---------------------------------------------------------------------------


class BayBEBackend(BaseBackend):
    """BayBE-based Bayesian Optimization backend.

    Implements the BOBackend protocol, maximizing use of BayBE-native APIs.
    Inherits duplicate detection, batch diversity, and JSON state-envelope
    helpers from :class:`BaseBackend`; overrides validation and method
    metadata to encode the BayBE-specific capability matrix.
    """

    @property
    def name(self) -> str:
        return "baybe"

    @property
    def supported_features(self) -> frozenset[Feature]:
        return _SUPPORTED_FEATURES

    # -- Spec features that BayBE does NOT support --------------------------
    _UNSUPPORTED_OPTIONS: list[tuple[str, str]] = [
        ("turbo_config", "TuRBO trust-region optimization"),
        ("saasbo_config", "SAASBO high-dimensional optimization"),
        ("fidelity_parameter", "Multi-fidelity optimization"),
        ("transfer_learning", "Transfer learning (RGPE)"),
        ("use_cost_aware", "Cost-aware optimization (EIpu)"),
        ("use_input_warping", "Input warping"),
        ("outcome_constraints", "Outcome constraints"),
    ]

    def validate_capabilities(self, spec: OptimizationSpec) -> BackendValidationResult:
        """Per-feature, per-option capability report for BayBE.

        ``Feature.CONSTRAINTS`` is reported per-constraint so hybrid /
        categorical-arithmetic constraints route ``backend="auto"`` away
        from BayBE instead of failing inside SearchSpace construction
        (TODO 1.64). ``Feature.TRANSFER_LEARNING`` is supported only when
        the campaign uses BayBE-native ``TaskParameter``s (TODO 1.65) —
        the BoTorch RGPE flavour exposed by ``OptimizationSpec.transfer_learning``
        remains an ignored option for BayBE. Misshaped
        ``parameter_options['baybe']`` and ``backend_options['baybe']``
        payloads surface as :class:`CapabilityStatus.UNSUPPORTED` reports
        here rather than crashing the suggestion path.
        """
        feature_reports = self._feature_reports(spec)
        option_reports = self._option_reports(spec)
        return BackendValidationResult(
            backend=self.name,
            feature_reports=tuple(feature_reports),
            option_reports=tuple(option_reports),
        )

    def _feature_reports(self, spec: OptimizationSpec) -> list[CapabilityReport]:
        """Build the feature half of :meth:`validate_capabilities` output.

        Features whose source spec attribute is in
        :data:`_BAYBE_DEGRADABLE_FEATURE_MAP` (``turbo_config``,
        ``saasbo_config``, ``fidelity_parameter``, ``use_cost_aware``,
        ``use_input_warping``, ``outcome_constraints``) are
        semantically load-bearing on BoTorch and would change the
        produced suggestions; BayBE cannot honor them. The report
        emits ``UNSUPPORTED`` by default so create-time capability
        enforcement rejects ``backend="baybe"`` for such specs.
        Listing the source attribute in
        :attr:`OptimizationSpec.acknowledge_degradations` downgrades
        the report to ``IGNORED`` — the caller has opted into the
        degraded run with a prominent warning. Features activated by
        more than one attribute (notably ``HIGH_DIMENSIONAL``, set by
        both ``turbo_config`` and ``saasbo_config``) only downgrade
        when *every* attribute actually set on the spec is
        acknowledged; otherwise the unacknowledged attribute's
        option-level report keeps the spec ``is_compatible == False``
        and the feature report names the specific unacknowledged
        attribute for the diagnostic. Genuinely incompatible items
        (hybrid constraints, RGPE transfer learning, malformed BayBE
        typed options) stay ``UNSUPPORTED`` regardless of
        acknowledgement.
        """
        reports: list[CapabilityReport] = []
        acknowledged_attrs = set(spec.acknowledge_degradations)
        transfer_handled = False
        for feature in sorted(required_features(spec)):
            if feature == Feature.CONSTRAINTS:
                reports.extend(self._constraint_feature_reports(spec))
            elif feature == Feature.TRANSFER_LEARNING:
                reports.append(self._transfer_learning_report(spec))
                transfer_handled = True
            else:
                reports.append(self._classify_feature(spec, feature, acknowledged_attrs))
        # A BayBE-native TaskParameter is the supported transfer-learning path
        # even when the neutral spec did not flag transfer_learning itself.
        if not transfer_handled:
            task_report = self._task_parameter_feature_report(spec)
            if task_report is not None:
                reports.append(task_report)
        return reports

    def _classify_feature(
        self,
        spec: OptimizationSpec,
        feature: Feature,
        acknowledged_attrs: set[str],
    ) -> CapabilityReport:
        """Classify a single non-constraint, non-transfer feature for BayBE."""
        if feature in self.supported_features:
            return CapabilityReport(key=str(feature), status=CapabilityStatus.SUPPORTED)
        if feature not in _BAYBE_DEGRADABLE_FEATURES:
            return CapabilityReport(
                key=str(feature),
                status=CapabilityStatus.UNSUPPORTED,
                reason=f"{feature} is not supported by BayBE",
            )
        active_attrs = _active_attrs_for_feature(spec, feature)
        unacknowledged = tuple(a for a in active_attrs if a not in acknowledged_attrs)
        if not unacknowledged:
            return CapabilityReport(
                key=str(feature),
                status=CapabilityStatus.IGNORED,
                reason=(
                    f"{feature} is not honored by BayBE and will be ignored "
                    "(degradation acknowledged)."
                ),
            )
        attrs_label = ", ".join(f"'{a}'" for a in unacknowledged)
        return CapabilityReport(
            key=str(feature),
            status=CapabilityStatus.UNSUPPORTED,
            reason=(
                f"{feature} is not honored by BayBE. The spec sets "
                f"{attrs_label} which BayBE cannot apply. Either pin "
                "backend='botorch' (or 'auto'), or list "
                f"{attrs_label} in 'acknowledge_degradations' to run "
                "on BayBE with this option silently dropped."
            ),
        )

    def _constraint_feature_reports(self, spec: OptimizationSpec) -> list[CapabilityReport]:
        """One :class:`CapabilityReport` per constraint.

        Delegates to :func:`baybe_constraint_support` so the capability
        report and the converter agree on what BayBE can construct. The
        report key includes the index plus neutral constraint type so an
        agent can match the warning back to the input spec.
        """
        if not spec.constraints:
            return []
        reports: list[CapabilityReport] = []
        for idx, constraint in enumerate(spec.constraints):
            key = f"constraint[{idx}]({constraint.type.value})"
            ok, reason = baybe_constraint_support(constraint, spec.parameters)
            if ok:
                reports.append(CapabilityReport(key=key, status=CapabilityStatus.SUPPORTED))
            else:
                reports.append(
                    CapabilityReport(
                        key=key,
                        status=CapabilityStatus.UNSUPPORTED,
                        reason=reason or "Unsupported constraint",
                    )
                )
        return reports

    def _transfer_learning_report(self, spec: OptimizationSpec) -> CapabilityReport:
        """Classify the spec's ``transfer_learning`` slot.

        BayBE's transfer-learning path is the native ``TaskParameter``
        mechanism — the neutral RGPE-style ``transfer_learning`` config
        targets the BoTorch backend. Reporting RGPE as ``IGNORED`` keeps
        BoTorch-shaped campaigns on BayBE only when the user explicitly
        pins ``backend="baybe"``; otherwise the
        :class:`~bo_engine.backend_base.CapabilityReport` is preserved as
        a method warning. When the spec exposes a TaskParameter via
        ``parameter_options['baybe'].role == 'task'`` the report flips to
        ``SUPPORTED``.
        """
        if any(_parameter_is_task(p.parameter_options) for p in spec.parameters):
            return CapabilityReport(
                key=str(Feature.TRANSFER_LEARNING),
                status=CapabilityStatus.SUPPORTED,
                reason="BayBE-native TaskParameter declared via parameter_options['baybe']",
            )
        return CapabilityReport(
            key=str(Feature.TRANSFER_LEARNING),
            status=CapabilityStatus.UNSUPPORTED,
            reason=(
                "BayBE supports transfer learning via TaskParameter; the neutral "
                "RGPE config targets the BoTorch backend."
            ),
        )

    def _task_parameter_feature_report(
        self,
        spec: OptimizationSpec,
    ) -> CapabilityReport | None:
        """Return a SUPPORTED transfer-learning report when a TaskParameter is present."""
        if not any(_parameter_is_task(p.parameter_options) for p in spec.parameters):
            return None
        return CapabilityReport(
            key=str(Feature.TRANSFER_LEARNING),
            status=CapabilityStatus.SUPPORTED,
            reason="BayBE-native TaskParameter declared via parameter_options['baybe']",
        )

    def _option_reports(self, spec: OptimizationSpec) -> list[CapabilityReport]:
        """Option-half of validate_capabilities: degradable knobs + typed BayBE option validation.

        Each active option that BayBE cannot honor maps to an
        ``UNSUPPORTED`` report by default. Listing the option's
        attribute name in :attr:`OptimizationSpec.acknowledge_degradations`
        downgrades that single entry to ``IGNORED`` — the caller has
        explicitly opted into running on BayBE with that semantic
        knob silently dropped. ``transfer_learning`` is exempt here
        because its feature-level report already routes
        ``TaskParameter``-style usage to ``SUPPORTED`` (option-level
        IGNORED is correct only when the caller has already chosen to
        run the spec on BayBE).
        """
        reports: list[CapabilityReport] = []
        acknowledged_attrs = set(spec.acknowledge_degradations)
        for attr, label in self._UNSUPPORTED_OPTIONS:
            value = getattr(spec, attr, None)
            is_set = bool(value) if not isinstance(value, list) else len(value) > 0
            if not is_set:
                continue
            if attr == "transfer_learning":
                # Transfer learning has its own routing — keep the
                # legacy IGNORED option-level warning to mirror what
                # callers grep for; the feature report decides
                # SUPPORTED/UNSUPPORTED.
                reports.append(
                    CapabilityReport(
                        key=attr,
                        status=CapabilityStatus.IGNORED,
                        reason=f"{label} is not supported by BayBE and will be ignored.",
                    )
                )
                continue
            if attr in acknowledged_attrs:
                reports.append(
                    CapabilityReport(
                        key=attr,
                        status=CapabilityStatus.IGNORED,
                        reason=(
                            f"{label} is not supported by BayBE and will be ignored "
                            "(degradation acknowledged)."
                        ),
                    )
                )
            else:
                reports.append(
                    CapabilityReport(
                        key=attr,
                        status=CapabilityStatus.UNSUPPORTED,
                        reason=(
                            f"{label} is not supported by BayBE; running this spec "
                            "on BayBE would silently drop the option. Pin "
                            "backend='botorch' (or 'auto'), or list "
                            f"'{attr}' in 'acknowledge_degradations' to accept "
                            "the degraded run."
                        ),
                    )
                )
        reports.extend(self._parameter_option_reports(spec))
        reports.extend(self._backend_option_reports(spec))
        return reports

    def _parameter_option_reports(self, spec: OptimizationSpec) -> list[CapabilityReport]:
        """Validate ``parameter_options['baybe']`` and surface schema errors."""
        reports: list[CapabilityReport] = []
        for p in spec.parameters:
            if not p.parameter_options or "baybe" not in p.parameter_options:
                continue
            try:
                opts = extract_baybe_parameter_options(p.parameter_options)
            except pydantic.ValidationError as e:
                reports.append(
                    CapabilityReport(
                        key=f"parameter_options[{p.name}].baybe",
                        status=CapabilityStatus.UNSUPPORTED,
                        reason=f"Invalid BayBE parameter options: {e.errors()[0]['msg']}",
                    )
                )
                continue
            reports.extend(_validate_parameter_role(p, opts))
        return reports

    def _backend_option_reports(self, spec: OptimizationSpec) -> list[CapabilityReport]:
        """Validate ``backend_options['baybe']`` and surface schema errors."""
        if not spec.backend_options or "baybe" not in spec.backend_options:
            return []
        try:
            extract_baybe_backend_options(spec.backend_options)
        except pydantic.ValidationError as e:
            return [
                CapabilityReport(
                    key="backend_options.baybe",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=f"Invalid BayBE backend options: {e.errors()[0]['msg']}",
                )
            ]
        return []

    def validate_spec(self, spec: OptimizationSpec) -> list[str]:
        """Backward-compatible string-warning surface for BayBE.

        Preserves the exact phrasing the previous implementation used so
        callers that grep the warnings (and tests that assert on the
        wording) keep working: ``"<label> is not supported by BayBE and
        will be ignored."``.
        """
        warnings: list[str] = []
        for attr, label in self._UNSUPPORTED_OPTIONS:
            value = getattr(spec, attr, None)
            is_set = bool(value) if not isinstance(value, list) else len(value) > 0
            if is_set:
                warnings.append(f"{label} is not supported by BayBE and will be ignored.")
        return warnings

    def generate_initial_design(
        self,
        spec: OptimizationSpec,
        n_points: int,
    ) -> list[dict[str, Any]]:
        campaign = _build_campaign(spec)
        rec_df = campaign.recommend(batch_size=n_points)
        return dataframe_to_suggestions(rec_df, spec)

    def generate_suggestions(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        batch_size: int,
        iteration: int,
        backend_state: dict[str, Any] | None = None,
        pending_points: list[dict[str, Any]] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> SuggestionBatch:
        # BayBE's campaign loop is internally segmented but does not yet
        # expose intermediate hooks; emit start/done milestones so the
        # MCP client at least sees a heartbeat.
        emit(
            progress_callback,
            ProgressEvent(
                phase="generate_suggestions_start",
                message=(
                    f"BayBE: recommending {batch_size} suggestion(s) on "
                    f"{len(observations)} observation(s)"
                ),
            ),
        )
        inner_state = self.unwrap_state(backend_state)
        campaign = _restore_or_build_campaign(spec, inner_state)

        # Reconcile measurements by stable identity (TODO 1.63). Run the
        # reconciliation even when observations is empty so a restored
        # campaign with stale measurements is rebuilt instead of carrying
        # forward data the storage layer no longer owns.
        campaign, identity_index = _reconcile_measurements(
            campaign, spec, observations, inner_state
        )

        pending_df, pending_warnings = self._convert_pending_points(spec, pending_points)
        rec_df = campaign.recommend(batch_size=batch_size, pending_experiments=pending_df)
        param_dicts = dataframe_to_suggestions(rec_df, spec)

        predictions, posterior_warning = _extract_posterior_stats(campaign, rec_df, spec)
        acq_values, acq_warning = _extract_acquisition_values(campaign, rec_df, pending_df)
        model_info, model_warning = _extract_model_info(campaign)

        method_info = self._method_info_from_campaign(campaign, spec, len(observations))
        method_info.update(model_info)
        acq_label = str(method_info.get("acquisition_function") or "")

        suggestions = _build_suggestion_list(
            param_dicts,
            predictions,
            acq_values,
            method_info,
            observations,
            iteration,
            batch_size,
            acquisition_label=acq_label,
        )

        warnings_out = self.validate_spec(spec)
        warnings_out.extend(pending_warnings)
        for warning in (posterior_warning, acq_warning, model_warning):
            if warning is None:
                continue
            warnings_out.append(f"BayBE introspection incomplete: {warning}")

        emit(
            progress_callback,
            ProgressEvent(
                phase="generate_suggestions_done",
                message=f"BayBE: produced {len(suggestions)} suggestion(s)",
                progress=float(len(suggestions)),
                total=float(batch_size),
            ),
        )

        return SuggestionBatch(
            suggestions=suggestions,
            method_info=method_info,
            backend_state=self.wrap_state(
                _serialize_campaign(campaign, identity_index),
                schema_version=_STATE_SCHEMA_VERSION_IDENTITY,
            ),
            warnings=warnings_out,
        )

    def _convert_pending_points(
        self,
        spec: OptimizationSpec,
        pending_points: list[dict[str, Any]] | None,
    ) -> tuple[pd.DataFrame | None, list[str]]:
        """Translate neutral pending dicts to BayBE's dataframe shape (TODO 1.62).

        Validation failures are downgraded to warnings rather than
        raising — pending points are an optimization hint, not a
        correctness requirement, and dropping them silently would defeat
        the whole feature. ``None`` is returned when no usable pending
        rows remain.
        """
        if not pending_points:
            return None, []
        try:
            df = pending_points_to_dataframe(pending_points, spec)
        except (TypeError, ValueError) as e:
            return None, [f"Pending points dropped: could not build BayBE dataframe ({e})"]
        if df.empty:
            return None, []
        return df, []

    def _method_info_from_campaign(
        self,
        campaign: Campaign,
        spec: OptimizationSpec,
        n_observations: int,
    ) -> dict[str, Any]:
        """Source method metadata from the active campaign (TODO 1.67).

        Falls back to the previous static labels only when BayBE cannot
        provide live introspection (e.g. before the first recommend or
        after a serialization edge case). Static fallbacks are tagged
        with ``"(fallback)"`` so callers can tell live metadata from a
        guess.
        """
        recommender = _active_recommender(campaign)
        is_multi = spec.n_objectives > 1
        is_nonpredictive = isinstance(recommender, NonPredictiveRecommender)
        rec_name = type(recommender).__name__ if recommender is not None else None
        searchspace_str = _campaign_searchspace_label(campaign)
        objective_type = type(getattr(campaign, "objective", None)).__name__
        strategy, model_type = _strategy_and_model(rec_name, is_nonpredictive, is_multi)
        acq_fn = _acquisition_label(recommender, is_nonpredictive, is_multi)

        return {
            "model_type": model_type,
            "acquisition_function": acq_fn,
            "optimization_strategy": strategy,
            "recommender": rec_name,
            "is_nonpredictive": is_nonpredictive,
            "searchspace_type": searchspace_str,
            "objective_type": objective_type,
            "input_transforms": ["BayBE internal encoding"],
            "explanation": (
                f"BayBE backend with {n_observations} observations. "
                f"Using {strategy}."
                + (f" Multi-objective with {spec.n_objectives} targets." if is_multi else "")
            ),
            "confidence": (
                "medium" if n_observations < _MIN_OBSERVATIONS_FOR_CONFIDENCE else "high"
            ),
            "alternatives": [],
            "warnings": [],
            "baybe_version": baybe_version,
            "bo_engine_baybe_version": _BO_ENGINE_BAYBE_VERSION,
        }

    def compute_hypervolume(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> float | None:
        if spec.n_objectives < 2 or len(observations) < 2:
            return None

        y_bo = _observations_to_minimization_tensor(spec, observations)
        pareto_y, _ = engine_compute_pareto_front(y_bo)
        ref_point = _compute_reference_point(y_bo)
        return engine_compute_hypervolume(pareto_y, ref_point)

    def select_methods(
        self,
        spec: OptimizationSpec,
        n_observations: int,
    ) -> dict[str, Any]:
        """Fallback method metadata when no live campaign is available.

        Live metadata is normally read off the running campaign inside
        :meth:`generate_suggestions` via
        :meth:`_method_info_from_campaign`. This entry point is invoked
        by the server's diagnostics pipeline when no campaign exists yet
        (e.g. before the first iteration) and therefore can only return
        static labels marked as ``(fallback)``.
        """
        is_multi = spec.n_objectives > 1
        if n_observations == 0:
            strategy = f"RandomRecommender (space-filling initial design) {_FALLBACK_LABEL}"
        else:
            strategy = f"BotorchRecommender (GP-based) {_FALLBACK_LABEL}"
        acq_fn = (
            f"{_FALLBACK_ACQ_MULTI} {_FALLBACK_LABEL}"
            if is_multi
            else f"{_FALLBACK_ACQ_SINGLE} {_FALLBACK_LABEL}"
        )
        return {
            "model_type": _MODEL_TYPE_MULTI if is_multi else _MODEL_TYPE_SINGLE,
            "acquisition_function": acq_fn,
            "optimization_strategy": strategy,
            "input_transforms": ["BayBE internal encoding"],
            "explanation": (
                f"BayBE backend with {n_observations} observations. "
                f"Using {strategy}."
                + (f" Multi-objective with {spec.n_objectives} targets." if is_multi else "")
            ),
            "confidence": (
                "medium" if n_observations < _MIN_OBSERVATIONS_FOR_CONFIDENCE else "high"
            ),
            "alternatives": [],
            "warnings": [],
            "baybe_version": baybe_version,
            "bo_engine_baybe_version": _BO_ENGINE_BAYBE_VERSION,
        }

    def compute_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        sections: frozenset[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        all_sections = frozenset(["objectives", "model", "outliers", "suggestions_tensor"])
        requested = all_sections if sections is None else sections
        _ = progress_callback  # BayBE diagnostics phases are not granular yet
        result: dict[str, Any] = {}

        if "objectives" in requested:
            result.update(self._compute_objective_diagnostics(spec, observations))

        # Model and hyperparameter diagnostics share a single fitted campaign (E5)
        need_model = "model" in requested or "suggestions_tensor" in requested
        if need_model:
            result.update(self._compute_model_and_hyperparameter_diagnostics(spec, observations))

        if "outliers" in requested:
            result.update(self._compute_outlier_diagnostics(spec, observations))

        return result

    def _compute_objective_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Compute objective metrics (best value, Pareto front, hypervolume)."""
        if spec.n_objectives == 1:
            return self._single_objective_diagnostics(spec, observations)
        return self._multi_objective_diagnostics(spec, observations)

    def _single_objective_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        obj = spec.objectives[0]
        values = [obs.objective_values[obj.name] for obs in observations]
        if not values:
            return {
                "best_value": None,
                "best_parameters": None,
                "improvement_history": [],
                "improvement_rate": 0.0,
                "pareto_front": None,
                "hypervolume": None,
                "n_pareto_points": None,
            }
        best_val, best_idx = compute_best_value(values, minimize=obj.minimize)
        imp_hist = compute_improvement_history(values, minimize=obj.minimize)
        return {
            "best_value": best_val,
            "best_parameters": observations[best_idx].parameter_values,
            "improvement_history": imp_hist,
            "improvement_rate": compute_single_objective_improvement_rate(imp_hist),
            "pareto_front": None,
            "hypervolume": None,
            "n_pareto_points": None,
        }

    def _multi_objective_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        if len(observations) < 2:
            return {
                "best_value": None,
                "best_parameters": None,
                "improvement_history": None,
                "improvement_rate": None,
                "pareto_front": [],
                "hypervolume": 0.0,
                "n_pareto_points": 0,
            }

        y_bo = _observations_to_minimization_tensor(spec, observations)
        minimize_mask = torch.tensor([o.minimize for o in spec.objectives], dtype=torch.bool)
        pareto_y, _ = engine_compute_pareto_front(y_bo)
        pareto_display = pareto_y.clone()
        pareto_display[:, ~minimize_mask] = -pareto_display[:, ~minimize_mask]

        obj_names = [o.name for o in spec.objectives]
        ref_point = get_reference_point(y_bo, minimize_mask)
        hv = engine_compute_hypervolume(pareto_y, ref_point)

        return {
            "best_value": None,
            "best_parameters": None,
            "improvement_history": None,
            "improvement_rate": None,
            "pareto_front": summarize_pareto_front(pareto_display, obj_names),
            "hypervolume": hv,
            "n_pareto_points": len(pareto_y),
        }

    def _compute_model_and_hyperparameter_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Compute model + hyperparameter diagnostics with a single fitted campaign (E5)."""
        empty: dict[str, Any] = {
            "feature_importance": None,
            "loo_cv_metrics": None,
            "model_correlation": None,
            "hyperparameters": None,
        }

        min_data = max(_MIN_DATA_ABSOLUTE, _MIN_DATA_PARAM_MULTIPLIER * len(spec.parameters))
        if len(observations) < min_data:
            return empty

        try:
            campaign = _build_fitted_campaign(spec, observations)
            obs_df = observations_to_dataframe(observations, spec)

            model_info, _ = _extract_model_info(campaign)
            fi = _extract_feature_importance(campaign)
            corr = _baybe_model_correlation(campaign, obs_df, spec)

            hp = {
                "kernel_type": model_info.get("kernel_type"),
                "lengthscales": model_info.get("lengthscales"),
                "noise_variance": model_info.get("noise_variance"),
                "output_scale": model_info.get("output_scale"),
            }

            return {
                "model_correlation": corr,
                "feature_importance": fi,
                "loo_cv_metrics": None,
                "hyperparameters": hp,
            }
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("BayBE model diagnostics failed: %s", e)
            return empty

    def _compute_outlier_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Detect outliers — normalized to BoTorch format (E4)."""
        if len(observations) < 5:
            return {"outliers": None}

        try:
            train_x, train_y, bounds = _prepare_tensors(spec, observations)
            minimize_mask = torch.tensor([o.minimize for o in spec.objectives], dtype=torch.bool)
            train_y_bo = train_y.clone()
            train_y_bo[:, ~minimize_mask] = -train_y_bo[:, ~minimize_mask]
            obj_names = [o.name for o in spec.objectives]

            outliers = detect_outliers(
                train_x=train_x,
                train_y=train_y_bo,
                bounds=bounds,
                objective_names=obj_names,
            )
            # Normalized to match BoTorch backend format (E4)
            outlier_results = (
                [
                    {
                        "result_index": o.index,
                        "standardized_error": round(o.standardized_error, 2),
                        "objective": o.objective_name,
                    }
                    for o in outliers
                ]
                if outliers
                else []
            )

            return {
                "outliers": {
                    "count": len(outlier_results),
                    "outlier_results": outlier_results,
                }
            }
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("Outlier detection failed: %s", e)
            return {"outliers": None}


# ---------------------------------------------------------------------------
# Module-level helpers (no self)
# ---------------------------------------------------------------------------


def _parameter_is_task(parameter_options: dict[str, dict[str, Any]] | None) -> bool:
    """Return True when ``parameter_options['baybe'].role == 'task'``."""
    try:
        opts = extract_baybe_parameter_options(parameter_options)
    except pydantic.ValidationError:
        return False
    return opts.role == BayBEParameterRole.TASK


def _validate_parameter_role(
    p: Any,
    opts: BayBEParameterOptions,
) -> list[CapabilityReport]:
    """Cross-check parameter-spec type against the requested BayBE role.

    Beyond the shape-level Pydantic checks, this also enforces BayBE
    semantic invariants:

    * ``role`` of ``task``/``substance`` requires a categorical base
      parameter.
    * ``active_values`` for ``role=task`` must all be members of the
      declared categories — BayBE's ``TaskParameter`` constructor
      raises a ``ValueError`` otherwise.
    * ``substance_data`` for ``role=substance`` must cover every
      declared category and ``baybe[chem]`` must be installed before
      BayBE's :class:`SubstanceParameter` can build its descriptor table.
    """
    if opts.role not in (BayBEParameterRole.TASK, BayBEParameterRole.SUBSTANCE):
        return []
    if p.type != ParameterType.CATEGORICAL:
        return [
            CapabilityReport(
                key=f"parameter_options[{p.name}].baybe.role",
                status=CapabilityStatus.UNSUPPORTED,
                reason=(
                    f"BayBE {opts.role.value} parameter requires a categorical base, "
                    f"got {p.type.value}"
                ),
            )
        ]

    categories = set(p.categories or [])
    if opts.role == BayBEParameterRole.TASK:
        return _task_role_reports(p.name, opts, categories)
    return _substance_role_reports(p.name, opts, categories)


def _task_role_reports(
    name: str,
    opts: BayBEParameterOptions,
    categories: set[str],
) -> list[CapabilityReport]:
    """Validate the ``role=task`` shape against the declared categories."""
    if not opts.active_values:
        return []
    unknown = sorted(set(opts.active_values) - categories)
    if not unknown:
        return []
    return [
        CapabilityReport(
            key=f"parameter_options[{name}].baybe.active_values",
            status=CapabilityStatus.UNSUPPORTED,
            reason=(
                f"BayBE TaskParameter active_values {unknown} not in declared "
                f"categories {sorted(categories)}"
            ),
        )
    ]


def _substance_role_reports(
    name: str,
    opts: BayBEParameterOptions,
    categories: set[str],
) -> list[CapabilityReport]:
    """Validate the ``role=substance`` shape plus runtime chemistry availability.

    Independent checks: missing ``baybe[chem]`` extras and a malformed
    ``substance_data`` are separate problems, so both can fire on the
    same parameter to give the user a complete punch list.
    """
    reports: list[CapabilityReport] = []
    if not _CHEMISTRY_AVAILABLE:
        reports.append(
            CapabilityReport(
                key=f"parameter_options[{name}].baybe.role",
                status=CapabilityStatus.UNSUPPORTED,
                reason=(
                    "BayBE substance parameters require the 'baybe[chem]' optional "
                    f"dependency, which is not installed ({_CHEMISTRY_UNAVAILABLE_REASON})."
                ),
            )
        )
    if not opts.substance_data:
        reports.append(
            CapabilityReport(
                key=f"parameter_options[{name}].baybe.substance_data",
                status=CapabilityStatus.UNSUPPORTED,
                reason="BayBE substance parameter requires substance_data (SMILES map).",
            )
        )
        return reports
    missing = sorted(categories - set(opts.substance_data.keys()))
    if missing:
        reports.append(
            CapabilityReport(
                key=f"parameter_options[{name}].baybe.substance_data",
                status=CapabilityStatus.UNSUPPORTED,
                reason=f"BayBE substance_data is missing SMILES for categories {missing}",
            )
        )
    return reports


def _baybe_model_correlation(
    campaign: Campaign,
    obs_df: pd.DataFrame,
    spec: OptimizationSpec,
) -> float:
    """Compute rank correlation between BayBE posterior mean and actuals."""
    from scipy import stats as scipy_stats

    try:
        stats_df = campaign.posterior_stats(candidates=obs_df, stats=("mean",))
        obj_name = spec.objectives[0].name
        mean_col = f"{obj_name}_mean"

        # Fall back to first mean column if exact name not found
        if mean_col not in stats_df.columns:
            mean_cols = [c for c in stats_df.columns if "mean" in str(c).lower()]
            if not mean_cols:
                return 0.0
            mean_col = mean_cols[0]

        predicted = stats_df[mean_col].values
        actual = obs_df[obj_name].values
        result = scipy_stats.spearmanr(predicted, actual)
        corr = float(result.statistic)
        return corr if corr == corr else 0.0  # Handle NaN  # noqa: PLR0124
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("BayBE model correlation failed: %s", e)
        return 0.0


def _prepare_tensors(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare train_x, train_y, bounds tensors from observations."""
    obj_names = [o.name for o in spec.objectives]

    x_rows = [encode_categorical(obs.parameter_values, spec) for obs in observations]
    train_x = torch.stack(x_rows)

    y_rows = [[obs.objective_values[n] for n in obj_names] for obs in observations]
    train_y = torch.tensor(y_rows, dtype=get_dtype(), device=get_device())

    bounds = get_bounds_tensor(spec)
    return train_x, train_y, bounds


# ---------------------------------------------------------------------------
# Provenance construction
# ---------------------------------------------------------------------------


def _build_suggestion_list(
    param_dicts: list[dict[str, Any]],
    predictions: list[dict[str, dict[str, float] | None]],
    acq_values: list[float | None],
    method_info: dict[str, Any],
    observations: list[ObservationData],
    iteration: int,
    batch_size: int,
    acquisition_label: str,
) -> list[dict[str, Any]]:
    """Build the suggestion list with full BayBE-sourced provenance."""
    suggestions: list[dict[str, Any]] = []
    for i, params in enumerate(param_dicts):
        pred = predictions[i] if i < len(predictions) else {}
        acq_val = acq_values[i] if i < len(acq_values) else None

        suggestions.append(
            {
                "parameter_values": params,
                "provenance": {
                    "iteration": iteration,
                    "batch_index": i,
                    "generation_method": "baybe_bo",
                    "acquisition_function": acquisition_label,
                    "acquisition_value": acq_val,
                    "model_type": method_info.get("model_type", _MODEL_TYPE_SINGLE),
                    "confidence_level": (
                        "medium" if len(observations) < _MIN_OBSERVATIONS_FOR_CONFIDENCE else "high"
                    ),
                    "explanation": (
                        f"Suggestion {i + 1}/{batch_size} generated by BayBE "
                        f"with {len(observations)} prior observations."
                    ),
                    "predicted_objectives": pred.get("predicted_objectives"),
                    "predicted_std": pred.get("predicted_std"),
                },
            }
        )
    return suggestions
