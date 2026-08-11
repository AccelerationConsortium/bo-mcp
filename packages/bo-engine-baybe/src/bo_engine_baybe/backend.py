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

The implementation has been split across companion modules so each file
owns one concern and stays well under the 1k LOC cognitive-load ceiling:

* :mod:`bo_engine_baybe.state` — campaign construction, state envelope
  serialization, and stable-identity measurement reconciliation.
* :mod:`bo_engine_baybe.introspection` — BayBE-native posterior /
  acquisition / model / SHAP extraction and the diagnostics tensor
  helpers.
* :mod:`bo_engine_baybe.capabilities` — static capability matrix, the
  degradable-knob policy, and the parameter-role validators consumed by
  :meth:`BayBEBackend.validate_capabilities`.

This module retains the :class:`BayBEBackend` class plus the
``method_info`` / provenance helpers that are shaped by the backend's
constants. Private helper symbols (``_observation_fingerprint``,
``_build_observation_identity``, ``_CHEMISTRY_AVAILABLE``) remain
re-exported from this module so existing test imports keep working
without modification.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, ClassVar, cast

import pandas as pd
import pydantic
import torch
from baybe import Campaign
from baybe import __version__ as baybe_version
from baybe.campaign import _EXCLUDED, _MEASURED, _RECOMMENDED
from baybe.exceptions import (
    IncompatibilityError,
    IncompleteMeasurementsError,
    ModelNotTrainedError,
    NoMeasurementsError,
    NotEnoughPointsLeftError,
)
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpaceType
from baybe.settings import Settings

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
    single_objective_family_acquisition_report,
    wrap_backend_exception,
)
from bo_engine.constants import (
    ACQF_FALLBACK_MIN_SAMPLES,
    DUPLICATE_DETECTION_TOLERANCE,
)
from bo_engine.diagnostics import (
    compute_best_value,
    compute_improvement_history,
    compute_observed_hypervolume,
    compute_single_objective_improvement_rate,
    observations_to_minimization_form,
    summarize_pareto_front,
)
from bo_engine.diagnostics import (
    compute_pareto_front as engine_compute_pareto_front,
)
from bo_engine.initial_design import SearchSpaceExhaustedError
from bo_engine.progress import ProgressCallback, ProgressEvent, emit
from bo_engine.reproducibility import GLOBAL_RNG_LOCK, derive_seed, draw_fallback_seed
from bo_engine.result_validation import (
    detect_duplicates as engine_detect_duplicates,
)
from bo_engine.result_validation import detect_outliers, duplicate_row_mask
from bo_engine.types import (
    SINGLE_OBJECTIVE_ONLY_ACQUISITION,
    UCB_FAMILY_ACQUISITION,
    AcquisitionMethod,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterType,
    TargetMode,
)
from bo_engine_baybe.capabilities import (
    _BAYBE_DEGRADABLE_FEATURES,
    _CHEMISTRY_AVAILABLE,
    _CHEMISTRY_UNAVAILABLE_REASON,
    _CONDITIONAL_FEATURES,
    _SUPPORTED_FEATURES,
    _active_attrs_for_feature,
    _parameter_is_task,
    _validate_parameter_fields,
    _validate_parameter_role,
)
from bo_engine_baybe.converters import (
    BAYBE_UNSUPPORTED_ACQUISITION,
    acquisition_output_count,
    baybe_constraint_support,
    dataframe_to_suggestions,
    log_transform_match_reason,
    log_transform_maximize_reason,
    objective_support_issues,
    observations_to_dataframe,
    pending_points_to_dataframe,
    resolve_searchspace_budget,
)
from bo_engine_baybe.introspection import (
    _MODEL_TYPE_SINGLE,
    _acquisition_label,
    _active_recommender,
    _baybe_model_correlation,
    _build_fitted_campaign,
    _campaign_searchspace_label,
    _extract_acquisition_values,
    _extract_feature_importance_report,
    _extract_model_info,
    _extract_posterior_stats,
    _prepare_tensors,
    _strategy_and_model,
)
from bo_engine_baybe.options import (
    BayBEBackendOptions,
    BayBEInitialRecommender,
    BayBEParameterOptions,
    BayBESurrogateKind,
    configured_non_gp_surrogate_kind,
    extract_baybe_backend_options,
    extract_baybe_parameter_options,
)
from bo_engine_baybe.searchspace_budget import subsample_warning
from bo_engine_baybe.state import (
    _BAYBE_SAFE_EXCEPTIONS,
    _INITIAL_RECOMMENDER_FACTORIES,
    _STATE_SCHEMA_VERSION_IDENTITY,
    _build_observation_identity,
    _initial_recommender_choice,
    _observation_fingerprint,
    _reconcile_measurements,
    _resolve_switch_after,
    _restore_or_build_campaign,
    _serialize_campaign,
)
from bo_engine_baybe.surrogates import (
    NOISE_PRIOR_IGNORED_REASON,
    build_baybe_surrogate,
    describe_configured_kernel,
)

logger = logging.getLogger(__name__)

# Version of this backend package. Imported lazily into ``method_info`` so the
# backend module does not need to import ``bo_engine_baybe.__init__`` (which
# would create a circular import via ``__init__`` re-exporting BayBEBackend).
_BO_ENGINE_BAYBE_VERSION = "0.1.0"

_MIN_OBSERVATIONS_FOR_CONFIDENCE = 5

# Minimum observations before fitting a model for diagnostics
_MIN_DATA_ABSOLUTE = 3
_MIN_DATA_PARAM_MULTIPLIER = 2

# Exceptions BayBE raises when acquisition values are *structurally*
# unavailable (nonpredictive recommender phase, untrained or incomplete
# surrogate). Only these may degrade the unseen-fallback ranking to "first
# candidate" — a genuine Bayesian-phase failure must propagate so the
# backend exception wrapper surfaces it instead of silently returning an
# unranked point under a success warning.
_ACQUISITION_UNAVAILABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    IncompatibilityError,
    IncompleteMeasurementsError,
    ModelNotTrainedError,
    NoMeasurementsError,
)


def _fallback_sample_count(campaign: Campaign) -> int:
    """Return the shared minimum candidate-cloud size for duplicate fallbacks."""
    recommender = _active_recommender(campaign)
    return max(
        int(getattr(recommender, "n_raw_samples", 0) or 0),
        ACQF_FALLBACK_MIN_SAMPLES,
    )


def _existing_continuous_rows(
    parameter_names: list[str],
    observations: list[ObservationData],
    pending_df: pd.DataFrame | None,
) -> torch.Tensor:
    """Collect evaluated and pending points as one float64 row tensor."""
    rows = [
        [float(observation.parameter_values[name]) for name in parameter_names]
        for observation in observations
        if all(name in observation.parameter_values for name in parameter_names)
    ]
    if pending_df is not None and not pending_df.empty:
        rows.extend(pending_df[parameter_names].astype(float).to_numpy().tolist())
    if not rows:
        return torch.empty((0, len(parameter_names)), dtype=torch.float64)
    return torch.tensor(rows, dtype=torch.float64)


def _sample_unseen_continuous_candidates(
    campaign: Campaign,
    parameter_names: list[str],
    existing_rows: torch.Tensor,
) -> pd.DataFrame:
    """Draw a q=1-feasible candidate cloud and drop already-seen points.

    Interpoint constraints bind across the rows of one sampled batch, but
    every fallback candidate stands in for an independent q=1
    recommendation — so with interpoint constraints present, candidates are
    drawn as single-point batches, making each one satisfy the constraint
    on its own.
    """
    sample_count = _fallback_sample_count(campaign)
    subspace = campaign.searchspace.continuous
    if subspace.has_interpoint_constraints:
        candidates = pd.concat(
            [subspace.sample_uniform(1) for _ in range(sample_count)],
            ignore_index=True,
        )
    else:
        candidates = subspace.sample_uniform(sample_count)
    candidate_rows = torch.from_numpy(
        candidates[parameter_names].to_numpy(dtype="float64", copy=True)
    )
    seen = duplicate_row_mask(candidate_rows, existing_rows)
    return candidates.loc[~seen.numpy()].reset_index(drop=True)


def _existing_hybrid_records(
    parameter_names: list[str],
    observations: list[ObservationData],
    pending_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Collect evaluated and pending hybrid points in experimental form."""
    records = [
        {name: observation.parameter_values[name] for name in parameter_names}
        for observation in observations
        if all(name in observation.parameter_values for name in parameter_names)
    ]
    if pending_df is not None and not pending_df.empty:
        records.extend(
            cast(
                "list[dict[str, Any]]",
                pending_df[parameter_names].to_dict(orient="records"),
            )
        )
    return records


def _sample_hybrid_candidates(campaign: Campaign, sample_count: int) -> pd.DataFrame:
    """Draw a bounded cloud spanning BayBE's feasible hybrid subspaces.

    BayBE represents a hybrid space as an enumerated, constraint-filtered
    discrete subspace plus a continuous subspace with its own native sampler.
    Every discrete configuration is represented when the bounded cloud is
    large enough; otherwise a reproducible random subset is used. Pairing
    those rows with native continuous samples yields complete experimental-
    representation candidates suitable for ``Campaign.acquisition_values``.
    """
    discrete_pool = campaign.searchspace.discrete.exp_rep
    if discrete_pool.empty:
        message = "BayBE hybrid fallback has no feasible discrete configurations."
        raise RuntimeError(message)

    if len(discrete_pool) >= sample_count:
        discrete_candidates = discrete_pool.sample(n=sample_count, replace=False)
    else:
        repetitions, remainder = divmod(sample_count, len(discrete_pool))
        parts = [discrete_pool] * repetitions
        if remainder:
            parts.append(discrete_pool.sample(n=remainder, replace=False))
        discrete_candidates = pd.concat(parts, ignore_index=True)

    continuous_candidates = campaign.searchspace.continuous.sample_uniform(sample_count)
    return pd.concat(
        [
            discrete_candidates.reset_index(drop=True),
            continuous_candidates.reset_index(drop=True),
        ],
        axis=1,
    )


def _sample_unseen_hybrid_candidates(
    campaign: Campaign,
    parameter_names: list[str],
    categorical_names: list[str],
    existing_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Draw feasible full hybrid candidates and remove already-seen rows."""
    candidates = _sample_hybrid_candidates(campaign, _fallback_sample_count(campaign))
    candidate_records = cast(
        "list[dict[str, Any]]",
        candidates[parameter_names].to_dict(orient="records"),
    )
    seen = [
        _hybrid_record_is_duplicate(
            candidate,
            existing_records,
            categorical_names,
        )
        for candidate in candidate_records
    ]
    return candidates.loc[[not is_seen for is_seen in seen]].reset_index(drop=True)


def _hybrid_record_is_duplicate(
    candidate: dict[str, Any],
    existing_records: list[dict[str, Any]],
    categorical_names: list[str],
) -> bool:
    """Match categorical labels exactly before applying numeric tolerance."""
    category_matches = [
        existing
        for existing in existing_records
        if all(candidate.get(name) == existing.get(name) for name in categorical_names)
    ]
    return bool(
        engine_detect_duplicates(
            candidate,
            category_matches,
            DUPLICATE_DETECTION_TOLERANCE,
        )
    )


def _best_unseen_candidate(
    campaign: Campaign,
    candidates: pd.DataFrame,
    pending_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Pick the highest-acquisition candidate via BayBE's public API.

    ``Campaign.acquisition_values`` transforms candidates by column name,
    so the ranking is independent of both the dataframe's column order and
    BayBE's internal (alphabetical) computational-representation order.
    When acquisition values are structurally unavailable — the
    nonpredictive recommender phase, where any unseen sample is as good as
    another — the first candidate is returned. Any other ranking failure
    propagates so the backend exception wrapper reports it instead of
    silently substituting an unranked point.
    """
    try:
        acq_series = campaign.acquisition_values(
            candidates=candidates,
            pending_experiments=pending_df,
        )
        best_idx = int(acq_series.to_numpy().argmax())
    except _ACQUISITION_UNAVAILABLE_EXCEPTIONS as e:
        logger.debug("Unseen-fallback acquisition ranking unavailable: %s", e)
        best_idx = 0
    return candidates.iloc[[best_idx]].reset_index(drop=True)


def _replace_duplicate_continuous_recommendation(
    campaign: Campaign,
    spec: OptimizationSpec,
    rec_df: pd.DataFrame,
    observations: list[ObservationData],
    pending_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str | None]:
    """Replace a duplicate q=1 continuous recommendation with an unseen point.

    BayBE cannot disable re-recommendation through its campaign flags when a
    search space has a continuous component.  Match the native BoTorch
    backend's q=1 protection at the adapter boundary: normal recommendations
    pass through unchanged, while a duplicate falls back to the highest-
    acquisition feasible unseen point in a BayBE-native random candidate
    cloud (interpoint constraints are honored per candidate).

    Scope: enforced only for purely continuous spaces with
    ``batch_size == 1``, mirroring the native backend's documented q=1
    limitation. Hybrid spaces are handled by the companion adapter-level
    fallback below.

    Raises:
        RuntimeError: If every sampled fallback candidate matches an
            evaluated or pending point. A sampled cloud cannot prove
            exhaustion of a continuous space, so this mirrors the native
            backend's sampled-fallback contract (an honest failure, not
            a ``SearchSpaceExhaustedError``).
    """
    if campaign.searchspace.type is not SearchSpaceType.CONTINUOUS:
        return rec_df, None
    if len(rec_df) != 1:
        logger.debug(
            "Duplicate protection is not enforced for batch_size=%d; "
            "returning BayBE's recommendation unchanged.",
            len(rec_df),
        )
        return rec_df, None

    parameter_names = [parameter.name for parameter in spec.parameters]
    existing_rows = _existing_continuous_rows(parameter_names, observations, pending_df)
    recommended_row = torch.from_numpy(rec_df[parameter_names].to_numpy(dtype="float64", copy=True))
    if not bool(duplicate_row_mask(recommended_row, existing_rows).any()):
        return rec_df, None

    candidates = _sample_unseen_continuous_candidates(campaign, parameter_names, existing_rows)
    if candidates.empty:
        message = (
            "BayBE returned an already evaluated or pending point, and no "
            "unseen fallback candidate was found in the sampled cloud. "
            "The space may still contain unseen points."
        )
        raise RuntimeError(message)

    best = _best_unseen_candidate(campaign, candidates, pending_df)
    campaign.clear_cache()
    warning = (
        "BayBE recommended an already evaluated or pending point; replaced it "
        "with a feasible unseen candidate."
    )
    logger.warning(warning)
    return best, warning


def _replace_duplicate_hybrid_recommendation(
    campaign: Campaign,
    spec: OptimizationSpec,
    rec_df: pd.DataFrame,
    observations: list[ObservationData],
    pending_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str | None]:
    """Replace a duplicate q=1 BayBE hybrid recommendation.

    Numeric coordinates use the engine's shared near-duplicate tolerance;
    categorical coordinates must match exactly. The fallback samples complete
    feasible hybrid candidates, filters evaluated and pending experiments, and
    ranks the remaining rows through BayBE's public acquisition API.
    """
    if campaign.searchspace.type is not SearchSpaceType.HYBRID:
        return rec_df, None
    if len(rec_df) != 1:
        logger.debug(
            "Duplicate protection is not enforced for hybrid batch_size=%d; "
            "returning BayBE's recommendation unchanged.",
            len(rec_df),
        )
        return rec_df, None

    parameter_names = [parameter.name for parameter in spec.parameters]
    categorical_names = [
        parameter.name
        for parameter in spec.parameters
        if parameter.type == ParameterType.CATEGORICAL
    ]
    existing_records = _existing_hybrid_records(parameter_names, observations, pending_df)
    recommendation = cast(
        "dict[str, Any]",
        rec_df.iloc[0][parameter_names].to_dict(),
    )
    if not _hybrid_record_is_duplicate(
        recommendation,
        existing_records,
        categorical_names,
    ):
        return rec_df, None

    candidates = _sample_unseen_hybrid_candidates(
        campaign,
        parameter_names,
        categorical_names,
        existing_records,
    )
    if candidates.empty:
        message = (
            "BayBE returned an already evaluated or pending hybrid point, and no "
            "unseen fallback candidate was found in the sampled cloud. "
            "The space may still contain unseen points."
        )
        raise RuntimeError(message)

    best = _best_unseen_candidate(campaign, candidates, pending_df)
    campaign.clear_cache()
    warning = (
        "BayBE recommended an already evaluated or pending hybrid point; "
        "replaced it with a feasible unseen candidate."
    )
    logger.warning(warning)
    return best, warning


def _replace_duplicate_recommendation(
    campaign: Campaign,
    spec: OptimizationSpec,
    rec_df: pd.DataFrame,
    observations: list[ObservationData],
    pending_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str | None]:
    """Apply the q=1 duplicate safeguard for continuous-component spaces."""
    if campaign.searchspace.type is SearchSpaceType.HYBRID:
        return _replace_duplicate_hybrid_recommendation(
            campaign,
            spec,
            rec_df,
            observations,
            pending_df,
        )
    return _replace_duplicate_continuous_recommendation(
        campaign,
        spec,
        rec_df,
        observations,
        pending_df,
    )


def _named_lengthscales(
    raw_lengthscales: object,
    parameter_names: list[str],
) -> dict[str, float] | None:
    """Normalize BayBE lengthscales to the BOBackend diagnostics contract.

    The shared diagnostics contract exposes ``lengthscales`` as
    ``dict[str, float]`` keyed by user-facing parameter names. BayBE reports
    raw model lengthscales as a positional vector; for encoded categorical or
    molecular features that vector can have a different dimensionality from the
    original search-space parameters. In that case, use stable encoded-dimension
    labels rather than pretending those dimensions are original parameters.
    """
    result: dict[str, float] | None = None
    values: list[float] | None = None

    def as_float(value: object) -> float:
        return float(cast("Any", value))

    if isinstance(raw_lengthscales, Mapping):
        result = {str(key): as_float(value) for key, value in raw_lengthscales.items()}
    elif isinstance(raw_lengthscales, (int, float)):
        values = [float(raw_lengthscales)]
    elif isinstance(raw_lengthscales, Iterable) and not isinstance(raw_lengthscales, (str, bytes)):
        values = [as_float(value) for value in raw_lengthscales]

    if values is not None:
        if not values:
            result = {}
        elif len(values) == 1 and parameter_names:
            value = round(values[0], 4)
            result = {name: value for name in parameter_names}
        elif len(values) == len(parameter_names):
            result = {
                name: round(value, 4) for name, value in zip(parameter_names, values, strict=True)
            }
        else:
            result = {f"encoded_dim_{index}": round(value, 4) for index, value in enumerate(values)}

    return result


# Seed-derivation context tags (see bo_engine.reproducibility.derive_seed).
# Namespaced with "baybe:" so seeds derived for this backend can never
# collide with the BoTorch backend's per-phase tags at the same master seed.
_SEED_CONTEXT_INITIAL_DESIGN = "baybe:initial_design"

# Capability-report key for the typed surrogate selection.
_SURROGATE_OPTION_KEY = "backend_options.baybe.surrogate"
_SEED_CONTEXT_ITERATION_TEMPLATE = "baybe:iter_{iteration}"

# Human-readable labels for the acquisition methods BayBE cannot express,
# keyed by the neutral enum. Used by the capability reports and the
# validate_spec warning surface; the mapping itself lives in
# bo_engine_baybe.converters.BAYBE_UNSUPPORTED_ACQUISITION.
_UNSUPPORTED_ACQUISITION_LABELS: dict[AcquisitionMethod, str] = {
    AcquisitionMethod.COST_WEIGHTED_EI: "Cost-weighted acquisition (cost_weighted_ei)",
    AcquisitionMethod.MULTI_FIDELITY_KG: ("Multi-fidelity knowledge gradient (multi_fidelity_kg)"),
}

# Reason attached to the DEGRADED capability report for log_transform
# objectives: BayBE honors the flag, but with weaker semantics than the
# BoTorch backend's Log → Standardize outcome transform.
_LOG_TRANSFORM_DEGRADED_REASON = (
    "BayBE applies log_transform at the acquisition-objective level "
    "(improvement is measured on the log scale), but its GP surrogate "
    "still fits the raw target scale — unlike BoTorch's Log → Standardize "
    "outcome transform. Pin backend='botorch' if log-scale model fitting "
    "is required."
)


# Re-export private symbols that the test suite imports from this module.
# Moving the implementations to dedicated submodules must not break
# ``from bo_engine_baybe.backend import _observation_fingerprint`` etc.
__all__ = [
    "_BAYBE_SAFE_EXCEPTIONS",
    "_CHEMISTRY_AVAILABLE",
    "_CHEMISTRY_UNAVAILABLE_REASON",
    "BayBEBackend",
    "_build_observation_identity",
    "_observation_fingerprint",
]


class BayBEBackend(BaseBackend):
    """BayBE-based Bayesian Optimization backend.

    Implements the BOBackend protocol, maximizing use of BayBE-native APIs.
    Inherits duplicate detection, batch diversity, and JSON state-envelope
    helpers from :class:`BaseBackend`; overrides validation and method
    metadata to encode the BayBE-specific capability matrix.
    """

    @property
    def name(self) -> str:
        """Backend identifier used by the registry."""
        return "baybe"

    @property
    def supported_features(self) -> frozenset[Feature]:
        """Capability set this backend exposes via :class:`BOBackend`."""
        return _SUPPORTED_FEATURES

    @property
    def conditional_features(self) -> dict[Feature, str]:
        """Features supported only under additional conditions, keyed by capability."""
        return dict(_CONDITIONAL_FEATURES)

    def parameter_options_schema(self) -> dict[str, Any]:
        """Return the JSON schema for ``parameter_options['baybe']``.

        Surfaces the typed :class:`BayBEParameterOptions` shape — the
        categorical ``encoding``, the ``role`` selector, ``active_values``
        for ``role=task``, ``substance_data`` / ``substance_encoding`` for
        ``role=substance``, and ``custom_descriptors`` / ``decorrelate`` for
        ``role=custom`` — so MCP and REST clients discover the molecular
        (substance) and custom-representation recipes from the schema
        instead of reading source. The fragment carries Pydantic's local
        ``$defs`` (nested enums); the schema-extension layer inlines it
        before splicing.
        """
        return BayBEParameterOptions.model_json_schema()

    def backend_options_schema(self) -> dict[str, Any]:
        """Return the JSON schema for ``backend_options['baybe']``.

        Surfaces the typed :class:`BayBEBackendOptions` shape — the
        recommender configuration and campaign-level toggles — so MCP
        and REST clients discover the per-campaign BayBE options from
        the schema instead of reading source. Same ``$defs`` inlining
        contract as :meth:`parameter_options_schema`.
        """
        return BayBEBackendOptions.model_json_schema()

    # -- Spec features that BayBE does NOT support --------------------------
    _UNSUPPORTED_OPTIONS: ClassVar[list[tuple[str, str]]] = [
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
        from BayBE instead of failing inside SearchSpace construction.
        ``Feature.TRANSFER_LEARNING`` is supported only when
        the campaign uses BayBE-native ``TaskParameter``s —
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
            reports.append(self._degradable_option_report(attr, label, acknowledged_attrs))
        acq_opt = getattr(spec, "acquisition_optimization", None)
        if acq_opt is not None and (
            acq_opt.num_restarts is not None or acq_opt.raw_samples is not None
        ):
            # The field is always present (default_factory); only explicit
            # overrides count as a degradation.
            # BoTorch L-BFGS-B restart/raw-sample budget. BayBE optimizes
            # acquisition internally and cannot honor it; IGNORED (not
            # UNSUPPORTED) so an explicit backend="baybe" still runs while
            # backend="auto" prefers a backend that honors the override.
            reports.append(
                CapabilityReport(
                    key="acquisition_optimization",
                    status=CapabilityStatus.IGNORED,
                    reason=(
                        "Acquisition-optimizer overrides (num_restarts / raw_samples) "
                        "target the BoTorch backend and are ignored by BayBE."
                    ),
                )
            )
        reports.extend(self._acquisition_method_reports(spec, acknowledged_attrs))
        reports.extend(self._log_transform_reports(spec))
        reports.extend(self._objective_surface_reports(spec))
        reports.extend(self._parameter_option_reports(spec))
        reports.extend(self._backend_option_reports(spec))
        reports.extend(_searchspace_budget_reports(spec))
        return reports

    @staticmethod
    def _objective_surface_reports(spec: OptimizationSpec) -> list[CapabilityReport]:
        """Report invalid match/transform/desirability objective configurations.

        Delegates to
        :func:`bo_engine_baybe.converters.objective_support_issues` so the
        capability report and ``spec_to_objective`` construction agree —
        the same symmetry contract the constraint surface uses.
        """
        return [
            CapabilityReport(key=key, status=CapabilityStatus.UNSUPPORTED, reason=reason)
            for key, reason in objective_support_issues(spec)
        ]

    @staticmethod
    def _degradable_option_report(
        attr: str,
        label: str,
        acknowledged_attrs: set[str],
    ) -> CapabilityReport:
        """UNSUPPORTED report for a knob BayBE drops; IGNORED when acknowledged."""
        if attr in acknowledged_attrs:
            return CapabilityReport(
                key=attr,
                status=CapabilityStatus.IGNORED,
                reason=(
                    f"{label} is not supported by BayBE and will be ignored "
                    "(degradation acknowledged)."
                ),
            )
        return CapabilityReport(
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

    def _acquisition_method_reports(
        self,
        spec: OptimizationSpec,
        acknowledged_attrs: set[str],
    ) -> list[CapabilityReport]:
        """Report ``spec.acquisition_method`` values BayBE cannot express.

        Mappable methods are wired into ``BotorchRecommender`` by
        :func:`bo_engine_baybe.converters.spec_to_acquisition_function`
        and need no report. The unmappable subset
        (:data:`BAYBE_UNSUPPORTED_ACQUISITION`) follows the standard
        degradable-knob policy: UNSUPPORTED by default, IGNORED once
        ``'acquisition_method'`` is acknowledged — the campaign then runs
        with BayBE's default acquisition function. A single-objective-only
        member requested on a spec whose built objective is genuinely
        multi-output (:func:`acquisition_output_count` — desirability
        scalarizes to one output and *does* honor these members) follows
        the same policy, because the dispatch falls back to the
        hypervolume default. Two combination rules are enforced
        alongside: ``acquisition_beta`` is only valid with a UCB-family
        method (the converter raises the matching ``ValueError`` at
        construction), and Thompson sampling cannot serve
        ``batch_size > 1`` (BayBE's ``qTS.supports_batching`` is False, so
        a batched recommend is guaranteed to fail at suggestion time).
        """
        reports: list[CapabilityReport] = []
        method = spec.acquisition_method
        if method in BAYBE_UNSUPPORTED_ACQUISITION:
            label = _UNSUPPORTED_ACQUISITION_LABELS[method]
            reports.append(
                self._degradable_option_report("acquisition_method", label, acknowledged_attrs)
            )
        elif method in SINGLE_OBJECTIVE_ONLY_ACQUISITION and acquisition_output_count(spec) > 1:
            reports.append(
                single_objective_family_acquisition_report(method, spec.acknowledge_degradations)
            )
        if spec.acquisition_beta is not None and method not in UCB_FAMILY_ACQUISITION:
            reports.append(
                CapabilityReport(
                    key="acquisition_beta",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        "acquisition_beta is only valid for the UCB acquisition "
                        f"family; got acquisition_method='{method.value}'. Remove "
                        "beta or select upper_confidence_bound."
                    ),
                )
            )
        if (
            method == AcquisitionMethod.THOMPSON_SAMPLING
            and acquisition_output_count(spec) == 1
            and spec.batch_size > 1
        ):
            reports.append(
                CapabilityReport(
                    key="acquisition_method",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        "Thompson sampling on BayBE (qTS) does not support batched "
                        f"recommendations; got batch_size={spec.batch_size}. Use "
                        "batch_size=1 or a different acquisition method."
                    ),
                )
            )
        return reports

    @staticmethod
    def _log_transform_reports(spec: OptimizationSpec) -> list[CapabilityReport]:
        """Per-objective report for ``log_transform`` objectives.

        Minimize objectives report DEGRADED: BayBE honors the flag (the
        converter chains a logarithmic target transformation), but with
        weaker semantics than BoTorch — the log enters the acquisition
        objective only, while the GP surrogate still fits the raw target
        scale. DEGRADED keeps the spec compatible (no acknowledgement
        gate) while ``backend="auto"`` tiering and the warning surface
        make the difference visible.

        Maximize objectives report UNSUPPORTED with the same reason the
        converter raises at construction time
        (:func:`log_transform_maximize_reason`) — otherwise
        ``is_compatible`` would accept a spec that
        ``generate_suggestions`` is guaranteed to reject. Direction
        resolves via ``ObjectiveSpec.effective_mode`` (the same read the
        target builder uses), so a ``target_mode`` override can never
        make this report and construction disagree; the
        ``log_transform`` × ``target_mode='match'`` combination is
        rejected by the shared ``objective_support_issues`` surface and
        skipped here to avoid a duplicate report.
        """
        reports: list[CapabilityReport] = []
        for idx, obj in enumerate(spec.objectives):
            if not obj.log_transform or obj.effective_mode == TargetMode.MATCH:
                continue
            key = f"objectives[{idx}].log_transform"
            if obj.effective_mode == TargetMode.MAXIMIZE:
                reports.append(
                    CapabilityReport(
                        key=key,
                        status=CapabilityStatus.UNSUPPORTED,
                        reason=log_transform_maximize_reason(obj.name),
                    )
                )
                continue
            reports.append(
                CapabilityReport(
                    key=key,
                    status=CapabilityStatus.DEGRADED,
                    reason=_LOG_TRANSFORM_DEGRADED_REASON,
                )
            )
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
            reports.extend(_validate_parameter_fields(p, opts))
        return reports

    def _backend_option_reports(self, spec: OptimizationSpec) -> list[CapabilityReport]:
        """Validate ``backend_options['baybe']`` and surface schema/build errors.

        Beyond the Pydantic schema check, this validates cross-field rules
        that only the backend can decide: non-random initial recommenders
        need a purely discrete search space (their BayBE ``compatibility``
        is ``SearchSpaceType.DISCRETE``), and the configured surrogate must
        actually construct (build-and-catch, the custom-descriptor-table
        precedent) so a bad kernel/preset is an intake-time report instead
        of a deferred fit crash. The neutral ``noise_prior_params`` is
        reported IGNORED — see
        :data:`bo_engine_baybe.surrogates.NOISE_PRIOR_IGNORED_REASON`.
        """
        reports: list[CapabilityReport] = []
        if spec.noise_prior_params is not None:
            reports.append(
                CapabilityReport(
                    key="noise_prior_params",
                    status=CapabilityStatus.IGNORED,
                    reason=NOISE_PRIOR_IGNORED_REASON,
                )
            )
        if not spec.backend_options or "baybe" not in spec.backend_options:
            return reports
        try:
            options = extract_baybe_backend_options(spec.backend_options)
        except pydantic.ValidationError as e:
            reports.append(
                CapabilityReport(
                    key="backend_options.baybe",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=f"Invalid BayBE backend options: {e.errors()[0]['msg']}",
                )
            )
            return reports
        reports.extend(self._recommender_option_reports(spec, options))
        reports.extend(self._surrogate_option_reports(spec, options))
        reports.extend(self._campaign_toggle_reports(spec, options))
        return reports

    @staticmethod
    def _campaign_toggle_reports(
        spec: OptimizationSpec,
        options: BayBEBackendOptions,
    ) -> list[CapabilityReport]:
        """Space-shape compatibility check for the ``allow_recommending_*`` toggles.

        BayBE forbids ``allow_recommending_already_measured=False`` /
        ``allow_recommending_already_recommended=False`` /
        ``allow_recommending_pending_experiments=False`` on campaigns
        whose search space has a continuous part ("for algorithmic
        reasons" — the per-candidate metadata ledger needs an enumerable
        space), raising ``IncompatibilityError`` at campaign build.
        Reporting the combination here turns the deferred suggestion-time
        crash into an intake-time report, mirroring the recommender
        space-shape checks.
        """
        has_continuous = any(p.type == ParameterType.CONTINUOUS for p in spec.parameters)
        if not has_continuous:
            return []
        reports: list[CapabilityReport] = []
        toggles = (
            ("allow_recommending_already_measured", options.allow_recommending_already_measured),
            (
                "allow_recommending_already_recommended",
                options.allow_recommending_already_recommended,
            ),
            (
                "allow_recommending_pending_experiments",
                options.allow_recommending_pending_experiments,
            ),
        )
        reports.extend(
            CapabilityReport(
                key=f"backend_options.baybe.{name}",
                status=CapabilityStatus.UNSUPPORTED,
                reason=(
                    f"{name}=False requires a purely discrete search space on "
                    "BayBE (the per-candidate metadata ledger cannot cover a "
                    "continuous subspace); this spec declares continuous "
                    "parameters. Drop the toggle or discretize the parameters."
                ),
            )
            for name, value in toggles
            if value is False
        )
        return reports

    @staticmethod
    def _recommender_option_reports(
        spec: OptimizationSpec,
        options: BayBEBackendOptions,
    ) -> list[CapabilityReport]:
        """Space-compatibility check for the initial-recommender selection."""
        if options.recommender is None:
            return []
        choice = options.recommender.initial_recommender
        if choice == BayBEInitialRecommender.RANDOM:
            return []
        has_continuous = any(str(p.type) == str(ParameterType.CONTINUOUS) for p in spec.parameters)
        if not has_continuous:
            return []
        return [
            CapabilityReport(
                key="backend_options.baybe.recommender.initial_recommender",
                status=CapabilityStatus.UNSUPPORTED,
                reason=(
                    f"initial_recommender='{choice.value}' requires a purely "
                    "discrete search space (BayBE clustering/FPS recommenders "
                    "are SearchSpaceType.DISCRETE); this spec declares "
                    "continuous parameters. Use initial_recommender='random'."
                ),
            )
        ]

    @staticmethod
    def _surrogate_option_reports(
        spec: OptimizationSpec,
        options: BayBEBackendOptions,
    ) -> list[CapabilityReport]:
        """Space-compatibility + build-and-catch guard for the surrogate selection.

        Non-GP surrogates (random forest, NGBoost, Bayesian linear, mean
        prediction) expose a non-differentiable posterior; BayBE's
        continuous-space acquisition optimization is gradient-based and
        fails at recommend time ("does not require grad"). They are
        therefore restricted to purely discrete search spaces, where the
        acquisition is scored by enumeration. The NGBoost surrogate needs
        an explicit module-availability probe (the chem-availability
        precedent): BayBE imports ``ngboost`` lazily at *fit* time, so
        the surrogate constructs successfully without the package and the
        build-and-catch guard below would let the missing dependency
        surface as a deferred ``Campaign.recommend`` crash.
        """
        if options.surrogate is None:
            return []
        has_continuous = any(str(p.type) == str(ParameterType.CONTINUOUS) for p in spec.parameters)
        if options.surrogate.kind != BayBESurrogateKind.GP and has_continuous:
            return [
                CapabilityReport(
                    key=_SURROGATE_OPTION_KEY,
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        f"Surrogate '{options.surrogate.kind.value}' has a "
                        "non-differentiable posterior and requires a purely "
                        "discrete search space on BayBE (continuous acquisition "
                        "optimization is gradient-based). Use kind='gp' or "
                        "declare a discrete space."
                    ),
                )
            ]
        if (
            options.surrogate.kind == BayBESurrogateKind.NGBOOST
            and importlib.util.find_spec("ngboost") is None
        ):
            return [
                CapabilityReport(
                    key=_SURROGATE_OPTION_KEY,
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        "Surrogate 'ngboost' needs the optional 'ngboost' "
                        "package, which is not installed (BayBE imports it "
                        "lazily at fit time, so the failure would otherwise "
                        "surface at suggestion generation)."
                    ),
                )
            ]
        try:
            build_baybe_surrogate(spec, options)
        except ImportError as e:
            return [
                CapabilityReport(
                    key=_SURROGATE_OPTION_KEY,
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        f"Surrogate '{options.surrogate.kind.value}' needs an "
                        f"optional dependency that is not installed: {e}"
                    ),
                )
            ]
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            return [
                CapabilityReport(
                    key=_SURROGATE_OPTION_KEY,
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=f"Surrogate construction failed: {e}",
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
        if spec.acquisition_method in BAYBE_UNSUPPORTED_ACQUISITION:
            label = _UNSUPPORTED_ACQUISITION_LABELS[spec.acquisition_method]
            warnings.append(f"{label} is not supported by BayBE and will be ignored.")
        warnings.extend(
            _log_transform_spec_warning(obj) for obj in spec.objectives if obj.log_transform
        )
        return warnings

    def generate_initial_design(
        self,
        spec: OptimizationSpec,
        n_points: int,
    ) -> list[dict[str, Any]]:
        """Recommend an initial batch of suggestions before any observations exist.

        Honors ``spec.random_seed``: when set, the recommendation runs
        inside :func:`_baybe_rng_scope` so repeated calls return the same
        design without leaking RNG state into the rest of the process.
        """
        from bo_engine_baybe.state import _build_campaign

        with _baybe_rng_scope(spec, _SEED_CONTEXT_INITIAL_DESIGN):
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
        initial_design_history: list[dict[str, Any]] | None = None,
    ) -> SuggestionBatch:
        """Recommend the next batch and translate BayBE/BoTorch errors to backend exceptions.

        The whole call runs inside :func:`_baybe_rng_scope` so a set
        ``spec.random_seed`` makes the batch reproducible without leaking
        seeded RNG state into the rest of the process.

        ``initial_design_history`` is intentionally ignored: BayBE owns its
        warm-up continuation through the persisted campaign in ``backend_state``
        and decides its own initial-vs-model phase (``switch_after``). Feeding
        it the caller's Sobol-issuance history would be meaningless (BayBE does
        not consume a BoTorch Sobol cursor), and routing those non-actionable
        points through ``pending_experiments`` would distort acquisition. Only
        ``pending_points`` — kept strictly actionable by the caller — reaches
        BayBE's recommender as pending experiments.
        """
        # Explicitly discard: see docstring. Named for signature/contract
        # parity with BOBackend; BayBE tracks its own continuation.
        del initial_design_history
        try:
            with _baybe_rng_scope(
                spec, _SEED_CONTEXT_ITERATION_TEMPLATE.format(iteration=iteration)
            ) as random_seed:
                return self._generate_suggestions_unwrapped(
                    spec=spec,
                    observations=observations,
                    batch_size=batch_size,
                    iteration=iteration,
                    backend_state=backend_state,
                    pending_points=pending_points,
                    progress_callback=progress_callback,
                    random_seed=random_seed,
                )
        except SearchSpaceExhaustedError:
            # Already engine-neutral (mapped from NotEnoughPointsLeftError);
            # the server has a dedicated SEARCH_SPACE_EXHAUSTED envelope.
            raise
        except (*_BAYBE_SAFE_EXCEPTIONS,) as exc:
            # Translate every library-level exception that escapes
            # BayBE/BoTorch/GPyTorch into the typed backend hierarchy
            # so the server layer dispatches on a stable type rather
            # than parsing exception strings.
            raise wrap_backend_exception(exc, backend_name=self.name) from exc

    def _generate_suggestions_unwrapped(
        self,
        *,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        batch_size: int,
        iteration: int,
        backend_state: dict[str, Any] | None,
        pending_points: list[dict[str, Any]] | None,
        progress_callback: ProgressCallback | None,
        random_seed: int | None,
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
        campaign = _restore_or_build_campaign(
            spec, inner_state, observations=observations, pending_points=pending_points
        )

        # Reconcile measurements by stable identity. Run the
        # reconciliation even when observations is empty so a restored
        # campaign with stale measurements is rebuilt instead of carrying
        # forward data the storage layer no longer owns.
        campaign, identity_index = _reconcile_measurements(
            campaign, spec, observations, inner_state
        )

        pending_df, pending_warnings = self._convert_pending_points(spec, pending_points)
        try:
            rec_df = campaign.recommend(batch_size=batch_size, pending_experiments=pending_df)
        except NotEnoughPointsLeftError as exc:
            # Translate to the engine-neutral exhaustion error so the server
            # emits the same structured SEARCH_SPACE_EXHAUSTED envelope
            # (counts + terminate recommendation) as the BoTorch path.
            raise _search_space_exhausted_error(spec, campaign, batch_size, pending_df) from exc
        rec_df, duplicate_warning = _replace_duplicate_recommendation(
            campaign,
            spec,
            rec_df,
            observations,
            pending_df,
        )
        param_dicts = dataframe_to_suggestions(rec_df, spec)

        predictions, posterior_warning = _extract_posterior_stats(campaign, rec_df, spec)
        acq_values, acq_warning = _extract_acquisition_values(campaign, rec_df, pending_df)
        non_gp_surrogate = configured_non_gp_surrogate_kind(
            extract_baybe_backend_options(spec.backend_options)
        )
        model_info, model_warning = _extract_model_info(campaign, non_gp_surrogate)

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
            random_seed=random_seed,
        )

        warnings_out = self.validate_spec(spec)
        warnings_out.extend(pending_warnings)
        if duplicate_warning is not None:
            warnings_out.append(duplicate_warning)
        subsample_note = _subsample_warning_for_spec(spec)
        if subsample_note is not None:
            warnings_out.append(subsample_note)
            method_info["searchspace_subsampled"] = True
            method_info["searchspace_n_candidates"] = len(campaign.searchspace.discrete.exp_rep)
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
        """Translate neutral pending dicts to BayBE's dataframe shape.

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
        """Source method metadata from the active campaign.

        Falls back to the previous static labels only when BayBE cannot
        provide live introspection (e.g. before the first recommend or
        after a serialization edge case). Static fallbacks are tagged
        with ``"(fallback)"`` so callers can tell live metadata from a
        guess.
        """
        recommender = _active_recommender(campaign)
        # Desirability declares several objectives but scalarizes to one
        # model output, so the model/acquisition labels key on the
        # acquisition output count, not the declared objective count.
        output_multi = acquisition_output_count(spec) > 1
        is_nonpredictive = isinstance(recommender, NonPredictiveRecommender)
        rec_name = type(recommender).__name__ if recommender is not None else None
        searchspace_str = _campaign_searchspace_label(campaign)
        objective_type = type(getattr(campaign, "objective", None)).__name__
        backend_opts = extract_baybe_backend_options(spec.backend_options)
        strategy, model_type = _strategy_and_model(
            rec_name,
            is_nonpredictive,
            output_multi,
            configured_non_gp_surrogate_kind(backend_opts),
        )
        acq_fn, acq_inferred = _acquisition_label(recommender, is_nonpredictive, output_multi)

        return {
            "model_type": model_type,
            "acquisition_function": acq_fn,
            # ``acquisition_function_inferred`` replaces the legacy
            # ``(fallback)`` suffix on the label with a structured signal:
            # True means BayBE did not expose a live ``acquisition_function``
            # attribute on the recommender and the label is the static
            # default. Downstream consumers (UIs, LLM tools) should branch
            # on this flag instead of string-matching the label.
            "acquisition_function_inferred": acq_inferred,
            "optimization_strategy": strategy,
            "kernel": describe_configured_kernel(backend_opts),
            "recommender": rec_name,
            "is_nonpredictive": is_nonpredictive,
            "searchspace_type": searchspace_str,
            "objective_type": objective_type,
            "input_transforms": ["BayBE internal encoding"],
            "explanation": (
                f"BayBE backend with {n_observations} observations. "
                f"Using {strategy}." + _objective_shape_sentence(spec, output_multi)
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
        """Return the hypervolume of observed Pareto front, or ``None`` if not applicable.

        Delegates to the shared :func:`compute_observed_hypervolume` so this
        backend reports the *same* ``None``/``0.0``/value contract and the
        *same* reference point as the BoTorch backend — the HV history that
        drives convergence detection is therefore comparable across backends.
        :meth:`_multi_objective_diagnostics` delegates to the same helper,
        so the live diagnostics ``hypervolume`` field agrees with this one.
        """
        return compute_observed_hypervolume(spec, observations)

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
        (e.g. before the first iteration). The legacy ``(fallback)``
        string suffix on the returned labels is replaced by the
        structured ``is_fallback: True`` field so programmatic consumers
        (UIs, LLM tools) can branch on a typed signal instead of
        string-matching a label.
        """
        # Same output-count rule as the live path: a desirability spec
        # runs a single-output model, so the fallback labels must not
        # advertise the hypervolume family.
        output_multi = acquisition_output_count(spec) > 1
        backend_opts = extract_baybe_backend_options(spec.backend_options)
        non_gp_surrogate = configured_non_gp_surrogate_kind(backend_opts)
        # Reuse the live-path label helpers so fallback and live metadata
        # agree phase-for-phase (no label churn when a consumer's next
        # call hits the live path). The phase is resolved exactly like
        # campaign construction: before the switch point the configured
        # initial recommender is active and no surrogate or acquisition
        # function exists.
        in_warmup = n_observations < _resolve_switch_after(spec)
        initial_name = (
            _INITIAL_RECOMMENDER_FACTORIES[_initial_recommender_choice(backend_opts)].__name__
            if in_warmup
            else None
        )
        strategy, model_type = _strategy_and_model(
            initial_name, in_warmup, output_multi, non_gp_surrogate
        )
        # With no live recommender to read (recommender=None), the warm-up
        # label "none (space-filling)" is definitive (inferred=False, as on
        # the live path) while the GP-phase label is the static-table guess
        # (inferred=True).
        acq_fn, acq_inferred = _acquisition_label(None, in_warmup, output_multi)
        return {
            "model_type": model_type,
            "acquisition_function": acq_fn,
            "acquisition_function_inferred": acq_inferred,
            "optimization_strategy": strategy,
            "kernel": describe_configured_kernel(backend_opts),
            "is_fallback": True,
            "input_transforms": ["BayBE internal encoding"],
            "explanation": (
                f"BayBE backend with {n_observations} observations. "
                f"Using {strategy}." + _objective_shape_sentence(spec, output_multi)
            ),
            # The static fallback can only guess at confidence; ``low``
            # is the honest signal until a live campaign is available.
            "confidence": "low",
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
        """Compute diagnostic sections (objectives, model, outliers, suggestions tensor)."""
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
        if obj.effective_mode == TargetMode.MATCH and obj.target_value is not None:
            # Match-a-target objectives: "best" is the observation closest
            # to the target value, and improvement is measured on the
            # distance-to-target trajectory (min/max of the raw values
            # would mislabel the extremes as best).
            distances = [abs(v - obj.target_value) for v in values]
            _, best_idx = compute_best_value(distances, minimize=True)
            best_val = values[best_idx]
            imp_hist = compute_improvement_history(distances, minimize=True)
        else:
            # Direction resolves via effective_mode so a target_mode
            # override wins over the raw boolean — the same read the
            # optimization itself honors.
            is_min = obj.effective_mode == TargetMode.MINIMIZE
            best_val, best_idx = compute_best_value(values, minimize=is_min)
            imp_hist = compute_improvement_history(values, minimize=is_min)
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
        """Compute Pareto front and hypervolume for multi-objective diagnostics.

        ``hypervolume`` is delegated to :func:`compute_observed_hypervolume`
        so this live diagnostics figure uses the same pinned reference
        point as :meth:`compute_hypervolume` / ``campaign.hypervolume_history``
        (see M75) — a moving, recomputed-per-call reference point would let
        a dominated, worse-in-one-objective observation inflate this field
        even though the Pareto front (and the history) did not improve.
        """
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

        y_bo, minimize_mask = observations_to_minimization_form(spec, observations)
        pareto_y, _ = engine_compute_pareto_front(y_bo)
        pareto_display = pareto_y.clone()
        pareto_display[:, ~minimize_mask] = -pareto_display[:, ~minimize_mask]

        obj_names = [o.name for o in spec.objectives]

        return {
            "best_value": None,
            "best_parameters": None,
            "improvement_history": None,
            "improvement_rate": None,
            "pareto_front": summarize_pareto_front(pareto_display, obj_names),
            "hypervolume": compute_observed_hypervolume(spec, observations),
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

            non_gp_surrogate = configured_non_gp_surrogate_kind(
                extract_baybe_backend_options(spec.backend_options)
            )
            model_info, _ = _extract_model_info(campaign, non_gp_surrogate)
            fi_report = _extract_feature_importance_report(campaign, spec) or {}
            corr = _baybe_model_correlation(campaign, obs_df, spec)
            parameter_names = [parameter.name for parameter in spec.parameters]

            hp = {
                "kernel_type": model_info.get("kernel_type"),
                "lengthscales": _named_lengthscales(
                    model_info.get("lengthscales"),
                    parameter_names,
                ),
                "noise_variance": model_info.get("noise_variance"),
                "output_scale": model_info.get("output_scale"),
                # Kernel-specific learned parameters — present when the
                # fitted kernel exposes them (offset for linear/polynomial,
                # period length for periodic, alpha for rational quadratic).
                "kernel_offset": model_info.get("kernel_offset"),
                "kernel_period_length": model_info.get("kernel_period_length"),
                "kernel_alpha": model_info.get("kernel_alpha"),
            }
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("BayBE model diagnostics failed: %s", e)
            return empty
        result: dict[str, Any] = {
            "model_correlation": corr,
            "feature_importance": fi_report.get("feature_importance"),
            "loo_cv_metrics": None,
            "hyperparameters": hp,
        }
        # Optional richer insights blocks (per-target breakdown, bounded
        # row-level attributions) — present only when computed.
        for key in ("feature_importance_per_target", "feature_importance_rows"):
            if key in fi_report:
                result[key] = fi_report[key]
        return result

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
            minimize_mask = torch.tensor(
                [o.effective_mode == TargetMode.MINIMIZE for o in spec.objectives],
                dtype=torch.bool,
            )
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


def _log_transform_spec_warning(obj: ObjectiveSpec) -> str:
    """Legacy string warning for one log-transformed objective.

    The maximize and match combinations are rejected by the capability
    layer; their warning strings reuse the shared rejection reasons so
    this surface can never contradict ``validate_capabilities``. The
    remaining (minimize) case keeps the historical acquisition-level
    phrasing.
    """
    if obj.effective_mode == TargetMode.MAXIMIZE:
        return log_transform_maximize_reason(obj.name)
    if obj.effective_mode == TargetMode.MATCH:
        return log_transform_match_reason(obj.name)
    return (
        f"log_transform for objective '{obj.name}' is applied at the "
        "acquisition-objective level on BayBE; the GP surrogate still "
        "fits the raw target scale."
    )


def _objective_shape_sentence(spec: OptimizationSpec, output_multi: bool) -> str:
    """Trailing explanation fragment describing the objective shape.

    Desirability declares several objectives but scalarizes them into a
    single model output, so the two counts diverge and the sentence must
    say which shape actually drives the model.
    """
    if spec.n_objectives <= 1:
        return ""
    if output_multi:
        return f" Multi-objective with {spec.n_objectives} targets."
    return f" Desirability scalarization over {spec.n_objectives} targets."


def _subsample_warning_for_spec(spec: OptimizationSpec) -> str | None:
    """Prominent warning when the spec's discrete space runs subsampled.

    Uses the same budget resolution as the construction branch in
    ``spec_to_searchspace`` so the warning can never disagree with what
    was built. Estimation failures degrade to "no warning" — the
    construction path surfaces its own error in that case.
    """
    try:
        budget = resolve_searchspace_budget(spec)
    except Exception as e:  # noqa: BLE001 — probe must never mask the real error surface
        # Broad by design: the precise estimate constructs BayBE parameter
        # objects, whose validators raise arbitrary library exceptions
        # (attrs ExceptionGroup for invalid SMILES, OptionalImportError for
        # missing chem extras, ...). Those failures are owned by the
        # parameter-role capability validators / the construction path —
        # the size probe only ever adds a warning on top.
        logger.debug("Search-space budget resolution failed: %s", e)
        return None
    if not budget.exceeds_budget:
        return None
    return subsample_warning(budget)


def _searchspace_budget_reports(spec: OptimizationSpec) -> list[CapabilityReport]:
    """DEGRADED capability report for above-budget discrete search spaces.

    DEGRADED (not UNSUPPORTED) keeps an explicit ``backend="baybe"``
    working — the campaign runs on a deterministically subsampled
    candidate set with a prominent warning — while ``backend="auto"``'s
    tier logic prefers a FULL backend (BoTorch) that does not enumerate
    the space at all.
    """
    try:
        budget = resolve_searchspace_budget(spec)
    except Exception as e:  # noqa: BLE001 — see _subsample_warning_for_spec
        logger.debug("Search-space budget resolution failed: %s", e)
        return []
    if not budget.exceeds_budget:
        return []
    return [
        CapabilityReport(
            key="searchspace_size",
            status=CapabilityStatus.DEGRADED,
            reason=subsample_warning(budget),
        )
    ]


def _excluded_candidate_mask(
    campaign: Campaign,
    pending_df: pd.DataFrame | None,
) -> pd.Series:
    """Boolean mask over the discrete grid of rows the recommender may not pick.

    Mirrors the candidate filter ``baybe.Campaign.recommend`` applies before
    raising ``NotEnoughPointsLeftError``: explicitly excluded rows, plus
    already-recommended / already-measured / pending rows whenever the
    corresponding ``allow_recommending_*`` flag does not permit them. Keeping
    the expression identical to BayBE's own is what makes the derived
    ``n_available`` the ground truth for what the recommender saw.
    """
    metadata = campaign._searchspace_metadata
    mask = metadata[_EXCLUDED].astype(bool)
    if not campaign.allow_recommending_already_recommended:
        mask |= metadata[_RECOMMENDED]
    if not campaign.allow_recommending_already_measured:
        mask |= metadata[_MEASURED]
    if pending_df is not None and not campaign.allow_recommending_pending_experiments:
        exp_rep = campaign.searchspace.discrete.exp_rep
        # Deduplicate before the left merge: a duplicated pending row fans
        # the merge out to more rows than exp_rep has, and an index-aligned
        # |= would then mark the wrong candidates. Re-anchoring the result
        # on exp_rep's index keeps the mask aligned with the metadata
        # ledger regardless of the merge's fresh RangeIndex.
        merged = exp_rep.merge(pending_df.drop_duplicates(), indicator=True, how="left")
        mask |= pd.Series(merged["_merge"].eq("both").to_numpy(), index=exp_rep.index)
    return mask


def _search_space_exhausted_error(
    spec: OptimizationSpec,
    campaign: Campaign,
    batch_size: int,
    pending_df: pd.DataFrame | None = None,
) -> SearchSpaceExhaustedError:
    """Build the engine-neutral exhaustion error from a BayBE campaign.

    Counts come from the campaign itself: the discrete subspace's
    experimental representation is the enumerated candidate grid, and
    ``n_available`` counts the rows left after the same measured /
    recommended / pending exclusions the failing ``recommend`` call
    applied (see :func:`_excluded_candidate_mask`) — deriving it from
    measurements alone would report unseen combinations while zero
    candidates actually remain. On the subsampled path the grid is a
    bounded candidate sample, not the full space, so the error carries
    the ``subsampled`` marker plus the full combination count and the
    active row budget — the server's exhaustion envelope then recommends
    raising ``backend_options['baybe'].max_candidates`` instead of
    terminating a campaign whose real space is far from exhausted.
    """
    n_total: int | None = None
    n_available = 0
    try:
        n_total = len(campaign.searchspace.discrete.exp_rep)
        n_available = int((~_excluded_candidate_mask(campaign, pending_df)).sum())
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:  # pragma: no cover - defensive
        logger.debug("Could not derive exhaustion counts: %s", e)
    subsampled = False
    n_full_combinations: int | None = None
    max_candidates: int | None = None
    try:
        budget = resolve_searchspace_budget(spec)
        if budget.exceeds_budget:
            subsampled = True
            n_full_combinations = budget.n_combinations
            max_candidates = budget.max_candidates
    except Exception as e:  # noqa: BLE001 — see _subsample_warning_for_spec
        logger.debug("Search-space budget resolution failed: %s", e)
    return SearchSpaceExhaustedError(
        n_requested=batch_size,
        n_available=n_available,
        n_total_combinations=n_total,
        subsampled=subsampled,
        n_full_combinations=n_full_combinations,
        max_candidates=max_candidates,
    )


@contextlib.contextmanager
def _baybe_rng_scope(spec: OptimizationSpec, context: str) -> Iterator[int | None]:
    """Scope the RNG-consuming section of a BayBE recommendation call.

    Every path holds the engine-wide
    :data:`bo_engine.reproducibility.GLOBAL_RNG_LOCK` — the same lock
    the BoTorch suggestion and Thompson-sampling paths hold around their
    ``fork_rng`` sections — so all global-RNG snapshot/restore consumers
    in the process are serialized consistently across worker threads
    (the server offloads generation via ``asyncio.to_thread``). When
    ``spec.random_seed`` is set, a per-phase seed is derived via
    :func:`bo_engine.reproducibility.derive_seed` and applied through
    BayBE's own settings system (``baybe.settings.Settings`` with
    ``random_seed``; Python, NumPy, and Torch), which snapshots the
    process-wide RNG states on entry and restores them on exit —
    re-running the same spec with the same ``random_seed`` and phase
    context reproduces the batch. Unseeded specs draw a fresh seed via
    :func:`bo_engine.reproducibility.draw_fallback_seed` (OS entropy —
    never the process-global stream a concurrent seeded scope may own)
    and apply it the same way; that path is deliberately
    non-reproducible, mirroring the BoTorch backend's
    ``_resolve_acquisition_seed`` fallback. The applied seed is yielded
    so callers can stamp it into suggestion provenance for auditability;
    no public spec field accepts a raw applied seed, so a recorded
    fallback seed cannot be replayed through ``spec.random_seed``.
    (Residual limitation: RNG consumers outside the lock — e.g.
    diagnostics model fits — can still interleave with a seeded scope.)
    """
    if spec.random_seed is None:
        seed = draw_fallback_seed()
    else:
        seed = derive_seed(spec.random_seed, context)
    # ty cannot see the attrs-generated __init__ on baybe's Settings.
    with GLOBAL_RNG_LOCK, Settings(random_seed=seed):  # ty: ignore[unknown-argument]
        yield seed


def _build_suggestion_list(
    param_dicts: list[dict[str, Any]],
    predictions: list[dict[str, dict[str, float] | None]],
    acq_values: list[float | None],
    method_info: dict[str, Any],
    observations: list[ObservationData],
    iteration: int,
    batch_size: int,
    acquisition_label: str,
    random_seed: int | None,
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
                    # Neutral phase labels shared with the BoTorch backend:
                    # nonpredictive recommender phase = initial design, GP
                    # phase = model-based BO. The backend identity lives in
                    # ``_metadata.backend`` / ``model_type``, not here.
                    "generation_method": (
                        "initial_design" if method_info.get("is_nonpredictive") else "bo"
                    ),
                    "acquisition_function": acquisition_label,
                    "acquisition_value": acq_val,
                    "random_seed": random_seed,
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
