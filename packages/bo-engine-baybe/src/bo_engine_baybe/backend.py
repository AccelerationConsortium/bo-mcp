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
import logging
from collections.abc import Iterator
from typing import Any, ClassVar

import pandas as pd
import pydantic
import torch
from baybe import Campaign
from baybe import __version__ as baybe_version
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.utils.random import temporary_seed
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
    wrap_backend_exception,
)
from bo_engine.diagnostics import (
    compute_best_value,
    compute_improvement_history,
    compute_observed_hypervolume,
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
from bo_engine.reproducibility import GLOBAL_RNG_LOCK, derive_seed
from bo_engine.result_validation import (
    detect_outliers,
)
from bo_engine.types import (
    AcquisitionMethod,
    ObservationData,
    OptimizationSpec,
)

from bo_engine_baybe.capabilities import (
    _BAYBE_DEGRADABLE_FEATURES,
    _CHEMISTRY_AVAILABLE,
    _CHEMISTRY_UNAVAILABLE_REASON,
    _CONDITIONAL_FEATURES,
    _SUPPORTED_FEATURES,
    _active_attrs_for_feature,
    _parameter_is_task,
    _validate_parameter_role,
)
from bo_engine_baybe.converters import (
    BAYBE_UNSUPPORTED_ACQUISITION,
    baybe_constraint_support,
    dataframe_to_suggestions,
    log_transform_maximize_reason,
    observations_to_dataframe,
    pending_points_to_dataframe,
)
from bo_engine_baybe.introspection import (
    _FALLBACK_ACQ_MULTI,
    _FALLBACK_ACQ_SINGLE,
    _MODEL_TYPE_MULTI,
    _MODEL_TYPE_SINGLE,
    _acquisition_label,
    _active_recommender,
    _baybe_model_correlation,
    _build_fitted_campaign,
    _campaign_searchspace_label,
    _extract_acquisition_values,
    _extract_feature_importance,
    _extract_model_info,
    _extract_posterior_stats,
    _observations_to_minimization_tensor,
    _prepare_tensors,
    _strategy_and_model,
)
from bo_engine_baybe.options import (
    BayBEParameterOptions,
    extract_baybe_backend_options,
    extract_baybe_parameter_options,
)
from bo_engine_baybe.state import (
    _BAYBE_SAFE_EXCEPTIONS,
    _STATE_SCHEMA_VERSION_IDENTITY,
    _build_observation_identity,
    _observation_fingerprint,
    _reconcile_measurements,
    _restore_or_build_campaign,
    _serialize_campaign,
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

# Seed-derivation context tags (see bo_engine.reproducibility.derive_seed).
# Namespaced with "baybe:" so seeds derived for this backend can never
# collide with the BoTorch backend's per-phase tags at the same master seed.
_SEED_CONTEXT_INITIAL_DESIGN = "baybe:initial_design"
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
        for ``role=task``, and ``substance_data`` / ``substance_encoding``
        for ``role=substance`` — so MCP and REST clients discover the
        molecular (substance) recipe from the schema instead of reading
        source. The fragment carries Pydantic's local ``$defs`` (nested
        enums); the schema-extension layer inlines it before splicing.
        """
        return BayBEParameterOptions.model_json_schema()

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
        reports.extend(self._acquisition_method_reports(spec, acknowledged_attrs))
        reports.extend(self._log_transform_reports(spec))
        reports.extend(self._parameter_option_reports(spec))
        reports.extend(self._backend_option_reports(spec))
        return reports

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

        Mappable methods (expected_improvement, noisy EI, hypervolume,
        scalarized) are wired into ``BotorchRecommender`` by
        :func:`bo_engine_baybe.converters.spec_to_acquisition_function`
        and need no report. The unmappable subset
        (:data:`BAYBE_UNSUPPORTED_ACQUISITION`) follows the standard
        degradable-knob policy: UNSUPPORTED by default, IGNORED once
        ``'acquisition_method'`` is acknowledged — the campaign then runs
        with BayBE's default acquisition function.
        """
        method = spec.acquisition_method
        if method not in BAYBE_UNSUPPORTED_ACQUISITION:
            return []
        label = _UNSUPPORTED_ACQUISITION_LABELS[method]
        return [self._degradable_option_report("acquisition_method", label, acknowledged_attrs)]

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
        ``generate_suggestions`` is guaranteed to reject.
        """
        reports: list[CapabilityReport] = []
        for idx, obj in enumerate(spec.objectives):
            if not obj.log_transform:
                continue
            key = f"objectives[{idx}].log_transform"
            if not obj.minimize:
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
        if spec.acquisition_method in BAYBE_UNSUPPORTED_ACQUISITION:
            label = _UNSUPPORTED_ACQUISITION_LABELS[spec.acquisition_method]
            warnings.append(f"{label} is not supported by BayBE and will be ignored.")
        warnings.extend(
            log_transform_maximize_reason(obj.name)
            if not obj.minimize
            else (
                f"log_transform for objective '{obj.name}' is applied at the "
                "acquisition-objective level on BayBE; the GP surrogate still "
                "fits the raw target scale."
            )
            for obj in spec.objectives
            if obj.log_transform
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
    ) -> SuggestionBatch:
        """Recommend the next batch and translate BayBE/BoTorch errors to backend exceptions.

        The whole call runs inside :func:`_baybe_rng_scope` so a set
        ``spec.random_seed`` makes the batch reproducible without leaking
        seeded RNG state into the rest of the process.
        """
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
        campaign = _restore_or_build_campaign(spec, inner_state)

        # Reconcile measurements by stable identity. Run the
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
            random_seed=random_seed,
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
        is_multi = spec.n_objectives > 1
        is_nonpredictive = isinstance(recommender, NonPredictiveRecommender)
        rec_name = type(recommender).__name__ if recommender is not None else None
        searchspace_str = _campaign_searchspace_label(campaign)
        objective_type = type(getattr(campaign, "objective", None)).__name__
        strategy, model_type = _strategy_and_model(rec_name, is_nonpredictive, is_multi)
        acq_fn, acq_inferred = _acquisition_label(recommender, is_nonpredictive, is_multi)

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
        """Return the hypervolume of observed Pareto front, or ``None`` if not applicable.

        Delegates to the shared :func:`compute_observed_hypervolume` so this
        backend reports the *same* ``None``/``0.0``/value contract and the
        *same* reference point as the BoTorch backend — the HV history that
        drives convergence detection is therefore comparable across backends,
        and consistent with this backend's own diagnostics surface (which
        already scores HV via ``get_reference_point``).
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
        is_multi = spec.n_objectives > 1
        if n_observations == 0:
            strategy = "RandomRecommender (space-filling initial design)"
        else:
            strategy = "BotorchRecommender (GP-based)"
        acq_fn = _FALLBACK_ACQ_MULTI if is_multi else _FALLBACK_ACQ_SINGLE
        return {
            "model_type": _MODEL_TYPE_MULTI if is_multi else _MODEL_TYPE_SINGLE,
            "acquisition_function": acq_fn,
            "acquisition_function_inferred": True,
            "optimization_strategy": strategy,
            "is_fallback": True,
            "input_transforms": ["BayBE internal encoding"],
            "explanation": (
                f"BayBE backend with {n_observations} observations. "
                f"Using {strategy}."
                + (f" Multi-objective with {spec.n_objectives} targets." if is_multi else "")
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
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("BayBE model diagnostics failed: %s", e)
            return empty
        return {
            "model_correlation": corr,
            "feature_importance": fi,
            "loo_cv_metrics": None,
            "hyperparameters": hp,
        }

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
    BayBE's own :func:`baybe.utils.random.temporary_seed` (Python, NumPy,
    and Torch), which snapshots the process-wide RNG states on entry and
    restores them on exit. Unseeded calls take the lock without seeding:
    they consume the ambient streams as-is (the documented
    non-reproducible path), and serializing them keeps their draws out
    of any concurrent seeded scope's window. (Residual limitation: RNG
    consumers outside the lock — e.g. diagnostics model fits — can still
    interleave with a seeded scope.) Yields the derived seed so callers
    can stamp it into suggestion provenance, or ``None`` when the spec
    carries no seed.
    """
    if spec.random_seed is None:
        with GLOBAL_RNG_LOCK:
            yield None
        return
    seed = derive_seed(spec.random_seed, context)
    with GLOBAL_RNG_LOCK, temporary_seed(seed):
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
                    "generation_method": "baybe_bo",
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
