"""Shared base classes and types for BO backends.

This module is the single source of truth for the typed surfaces that
new backends need to opt into instead of reimplementing:

* :class:`BackendValidationResult` — spec-aware capability report. Backends
  classify each feature/option as supported, degraded, ignored, or
  unsupported for a concrete :class:`~bo_engine.types.OptimizationSpec`.
* :class:`NormalizedProblem` — backend-neutral intermediate representation
  (parameters, objectives, bounds, ordered views). New backends author
  ``OptimizationSpec -> NormalizedProblem -> native backend`` and skip
  duplicating shared normalization rules.
* :class:`BackendStateEnvelope` — versioned wrapper for opaque backend
  state. The server persists this as JSON; the envelope tags the payload
  with the originating backend and a schema version so restoration is
  unambiguous.
* :class:`BaseBackend` — abstract class with sensible defaults for the
  optional members of :class:`~bo_engine.backend.BOBackend`. Concrete
  backends only override what differs from the default behavior.

The :class:`BOBackend` protocol stays public so third-party plugins can
implement it without inheriting :class:`BaseBackend`. The contract test
suite in :mod:`bo_engine.testing.backend_contract` checks both paths.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch

from bo_engine.backend import (
    BatchDiversityMetrics,
    DuplicateInfo,
    Feature,
    SuggestionBatch,
)
from bo_engine.batch_diversity import compute_batch_diversity
from bo_engine.device import get_device, get_dtype
from bo_engine.progress import ProgressCallback
from bo_engine.result_validation import detect_duplicates as engine_detect_duplicates
from bo_engine.suggestions import generate_initial_design as engine_generate_initial_design
from bo_engine.transforms import get_bounds_tensor
from bo_engine.types import ObservationData, OptimizationSpec

CURRENT_STATE_ENVELOPE_VERSION = 1
"""Schema version embedded in :class:`BackendStateEnvelope` payloads."""


class CapabilityStatus(StrEnum):
    """Per-feature/per-option capability status reported by a backend.

    The four levels are deliberately ordered from "fully usable" to "must
    reject"; ``resolve_backend_name("auto", ...)`` and other selectors can
    treat ``UNSUPPORTED`` as a hard veto while letting ``IGNORED`` and
    ``DEGRADED`` proceed with warnings surfaced to users.
    """

    SUPPORTED = "supported"
    DEGRADED = "degraded"
    IGNORED = "ignored"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CapabilityReport:
    """One entry in :class:`BackendValidationResult`.

    ``key`` identifies the feature (``Feature`` value) or option name
    (``"turbo_config"``, ``"use_input_warping"``, …). ``status`` records the
    spec-aware capability classification, and ``reason`` is a short
    human-readable string surfaced to clients.
    """

    key: str
    status: CapabilityStatus
    reason: str = ""


@dataclass(frozen=True)
class BackendValidationResult:
    """Spec-aware capability descriptor produced by ``validate_capabilities``.

    Replaces the flat ``supported_features`` boolean-set check with a
    per-feature, per-option breakdown. Selectors call ``is_compatible`` for
    a hard yes/no answer and ``warnings`` to surface caveats; UI/API layers
    can drill into the typed reports for per-option detail.
    """

    backend: str
    feature_reports: tuple[CapabilityReport, ...] = ()
    option_reports: tuple[CapabilityReport, ...] = ()

    @property
    def is_compatible(self) -> bool:
        """``True`` when no feature or option is reported as unsupported."""
        for report in (*self.feature_reports, *self.option_reports):
            if report.status == CapabilityStatus.UNSUPPORTED:
                return False
        return True

    @property
    def supported_features(self) -> frozenset[Feature]:
        """Subset of features classified ``SUPPORTED`` for this spec."""
        result: set[Feature] = set()
        for report in self.feature_reports:
            if report.status != CapabilityStatus.SUPPORTED:
                continue
            try:
                result.add(Feature(report.key))
            except ValueError:
                continue
        return frozenset(result)

    @property
    def warnings(self) -> list[str]:
        """Human-readable warnings for degraded/ignored items."""
        msgs: list[str] = []
        for report in (*self.feature_reports, *self.option_reports):
            if report.status in (CapabilityStatus.DEGRADED, CapabilityStatus.IGNORED):
                prefix = "degraded" if report.status == CapabilityStatus.DEGRADED else "ignored"
                msgs.append(f"[{prefix}] {report.key}: {report.reason}".rstrip(": "))
        return msgs

    @property
    def unsupported(self) -> list[CapabilityReport]:
        """Reports whose status is :attr:`CapabilityStatus.UNSUPPORTED`."""
        return [
            r
            for r in (*self.feature_reports, *self.option_reports)
            if r.status == CapabilityStatus.UNSUPPORTED
        ]


@dataclass(frozen=True)
class NormalizedParameter:
    """Backend-neutral parameter view used inside :class:`NormalizedProblem`.

    Cross-backend semantics live here; backend-specific knobs live in
    ``options`` (consumed by 1.65). ``bounds`` is canonical for continuous
    and numerical-discrete parameters, ``values`` for explicit discrete
    grids, ``categories`` for categorical labels.
    """

    name: str
    type: str
    bounds: tuple[float, float] | None
    values: tuple[float, ...] | None
    categories: tuple[str, ...] | None
    options: dict[str, Any]


@dataclass(frozen=True)
class NormalizedObjective:
    """Backend-neutral objective view (name and minimize/maximize direction)."""

    name: str
    minimize: bool


@dataclass(frozen=True)
class NormalizedProblem:
    """Backend-neutral intermediate representation of an ``OptimizationSpec``.

    New backends consume ``NormalizedProblem`` instead of re-deriving
    parameter ordering, bounds tensors, and direction masks for every
    spec. The constructor :func:`build_normalized_problem` materializes
    the IR; backends call helpers on it (``parameter_index``,
    ``bounds_tensor``, ``minimize_mask``) rather than rolling their own.
    """

    spec: OptimizationSpec
    parameters: tuple[NormalizedParameter, ...]
    objectives: tuple[NormalizedObjective, ...]
    parameter_index: dict[str, int]
    objective_index: dict[str, int]

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def objective_names(self) -> list[str]:
        return [o.name for o in self.objectives]

    def minimize_mask(self) -> torch.Tensor:
        """Boolean tensor: ``True`` where the objective is minimized."""
        return torch.tensor([o.minimize for o in self.objectives], dtype=torch.bool)

    def bounds_tensor(self) -> torch.Tensor:
        """Delegate to ``get_bounds_tensor`` for the underlying spec."""
        return get_bounds_tensor(self.spec)


def build_normalized_problem(spec: OptimizationSpec) -> NormalizedProblem:
    """Build a :class:`NormalizedProblem` from an :class:`OptimizationSpec`.

    Preserves parameter and objective ordering. ``parameter_options`` on
    each :class:`~bo_engine.types.ParameterSpec` flows into the IR's
    ``options`` dict so per-backend parameter metadata survives the
    normalization step.
    """
    norm_params: list[NormalizedParameter] = []
    for p in spec.parameters:
        norm_params.append(
            NormalizedParameter(
                name=p.name,
                type=str(p.type),
                bounds=p.bounds,
                values=tuple(float(v) for v in p.values) if p.values is not None else None,
                categories=tuple(p.categories) if p.categories is not None else None,
                options=dict(p.parameter_options or {}),
            )
        )
    norm_objs = tuple(
        NormalizedObjective(name=o.name, minimize=o.minimize) for o in spec.objectives
    )
    parameter_index = {p.name: i for i, p in enumerate(norm_params)}
    objective_index = {o.name: i for i, o in enumerate(norm_objs)}
    return NormalizedProblem(
        spec=spec,
        parameters=tuple(norm_params),
        objectives=norm_objs,
        parameter_index=parameter_index,
        objective_index=objective_index,
    )


@dataclass(frozen=True)
class BackendStateEnvelope:
    """Versioned wrapper for a backend's opaque persistence payload.

    The MCP server stores backend state as JSON. Wrapping the payload in
    an envelope means restoration knows which backend produced the state
    and which schema version to expect, so cross-backend migrations and
    incompatible upgrades fail loudly instead of silently corrupting a
    campaign.
    """

    backend: str
    schema_version: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "schema_version": int(self.schema_version),
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackendStateEnvelope:
        if not isinstance(data, dict):
            msg = "BackendStateEnvelope expects a dict payload"
            raise TypeError(msg)
        try:
            backend = str(data["backend"])
            schema_version = int(data["schema_version"])
            payload = data["payload"]
        except KeyError as e:
            msg = f"BackendStateEnvelope is missing required key: {e.args[0]}"
            raise ValueError(msg) from None
        if not isinstance(payload, dict):
            msg = "BackendStateEnvelope.payload must be a dict"
            raise TypeError(msg)
        return cls(backend=backend, schema_version=schema_version, payload=payload)


def is_state_envelope(data: dict[str, Any] | None) -> bool:
    """Return ``True`` when ``data`` looks like a :class:`BackendStateEnvelope`.

    Used by backends that need to peek at a stored state to decide whether
    it has already been wrapped in the envelope or is a bare legacy
    payload that should still be honored.
    """
    if not isinstance(data, dict):
        return False
    return {"backend", "schema_version", "payload"}.issubset(data.keys())


def unwrap_state(
    backend_name: str,
    data: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the inner payload for an envelope produced by ``backend_name``.

    Legacy bare payloads (pre-envelope) are returned unchanged so existing
    persisted state keeps working. Envelopes produced by a *different*
    backend raise ``ValueError`` — silently consuming foreign state would
    risk feeding a BayBE JSON blob into a BoTorch TurboState parser, etc.
    """
    if data is None:
        return None
    if not is_state_envelope(data):
        return data
    envelope = BackendStateEnvelope.from_dict(data)
    if envelope.backend != backend_name:
        msg = (
            f"Backend state belongs to '{envelope.backend}', cannot be restored "
            f"by backend '{backend_name}'"
        )
        raise ValueError(msg)
    return envelope.payload


def wrap_state(
    backend_name: str,
    payload: dict[str, Any] | None,
    schema_version: int = CURRENT_STATE_ENVELOPE_VERSION,
) -> dict[str, Any] | None:
    """Wrap ``payload`` in a JSON-serializable :class:`BackendStateEnvelope`.

    Returns ``None`` when ``payload`` is ``None`` so backends without
    persistent state stay no-op. The result is validated via
    ``json.dumps`` to catch non-serializable contents at the boundary.
    """
    if payload is None:
        return None
    envelope = BackendStateEnvelope(
        backend=backend_name,
        schema_version=schema_version,
        payload=payload,
    )
    encoded = envelope.to_dict()
    json.dumps(encoded)  # fail fast on non-JSON-safe payloads
    return encoded


class BaseBackend(ABC):
    """Abstract base class implementing :class:`~bo_engine.backend.BOBackend` defaults.

    Concrete backends only have to override:

    * :attr:`name`
    * :meth:`generate_suggestions`

    Everything else (initial design via Sobol, duplicate detection,
    diversity metrics, no-op state updates, generic method metadata,
    spec-aware capability validation, JSON-state envelope handling)
    has a default implementation here.

    Backends that need richer behavior can override any of the optional
    members; subclasses do not have to call ``super`` for the defaults.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier (``"botorch"``, ``"baybe"``, …)."""

    @property
    def supported_features(self) -> frozenset[Feature]:
        """Default: empty set. Concrete backends override to advertise features."""
        return frozenset()

    # -- Capability validation --------------------------------------------

    def validate_capabilities(self, spec: OptimizationSpec) -> BackendValidationResult:
        """Return a per-feature, per-option :class:`BackendValidationResult`.

        Default implementation reports every member of
        :class:`Feature` that the spec actually exercises against
        ``supported_features`` and emits an :class:`CapabilityReport` per
        option attribute set on the spec. Subclasses may override to
        encode finer rules (e.g. BayBE rejecting hybrid constraints).
        """
        feature_reports: list[CapabilityReport] = []
        required = required_features(spec)
        supported = self.supported_features
        for feature in required:
            if feature in supported:
                feature_reports.append(
                    CapabilityReport(key=str(feature), status=CapabilityStatus.SUPPORTED)
                )
            else:
                feature_reports.append(
                    CapabilityReport(
                        key=str(feature),
                        status=CapabilityStatus.UNSUPPORTED,
                        reason=f"{feature} is not supported by backend '{self.name}'",
                    )
                )
        option_reports = self._classify_options(spec)
        return BackendValidationResult(
            backend=self.name,
            feature_reports=tuple(feature_reports),
            option_reports=tuple(option_reports),
        )

    def _classify_options(self, spec: OptimizationSpec) -> list[CapabilityReport]:
        """Per-option default: all spec-level options are reported as supported.

        Backends override this to mark unsupported knobs (e.g. BayBE's
        ``turbo_config``-is-ignored behavior).
        """
        reports: list[CapabilityReport] = []
        for option_name in _SPEC_OPTION_KEYS:
            if option_is_active(spec, option_name):
                reports.append(CapabilityReport(key=option_name, status=CapabilityStatus.SUPPORTED))
        return reports

    def validate_spec(self, spec: OptimizationSpec) -> list[str]:
        """Backward-compatible string-warning surface backed by capabilities.

        Returns the warning strings derived from
        :meth:`validate_capabilities`. Existing callers see the same
        list-of-strings shape, while new callers can opt into the richer
        :class:`BackendValidationResult`.
        """
        result = self.validate_capabilities(spec)
        return result.warnings

    # -- Suggestion generation --------------------------------------------

    def generate_initial_design(
        self,
        spec: OptimizationSpec,
        n_points: int,
    ) -> list[dict[str, Any]]:
        """Sobol-based initial design via :func:`bo_engine.suggestions.generate_initial_design`."""
        return engine_generate_initial_design(spec, n_points)

    @abstractmethod
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
        """Concrete backends must implement model-guided suggestion generation."""

    # -- Metrics & diagnostics --------------------------------------------

    def compute_hypervolume(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> float | None:
        """Default implementation: ``None`` for single-objective, 0.0 for <2 obs.

        Subclasses with native hypervolume implementations override this.
        Returning ``None`` for single-objective matches the BoTorch
        backend's contract — hypervolume is only meaningful for
        multi-objective campaigns.
        """
        if spec.n_objectives < 2:
            return None
        if len(observations) < 2:
            return 0.0
        # Backends with no native HV implementation should override this.
        return None

    def detect_duplicates(
        self,
        new_params: dict[str, Any],
        existing_params: list[dict[str, Any]],
        tolerance: float,
    ) -> list[DuplicateInfo]:
        """Reuse :func:`bo_engine.result_validation.detect_duplicates`."""
        raw = engine_detect_duplicates(new_params, existing_params, tolerance)
        return [
            DuplicateInfo(
                index=d.index,
                is_exact=d.is_exact,
                parameter_distance=d.parameter_distance,
            )
            for d in raw
        ]

    def update_state_after_results(
        self,
        spec: OptimizationSpec,
        new_observations: list[ObservationData],
        backend_state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Default: pass the state through unchanged.

        Backends that own evolving state (TuRBO trust region, BayBE
        campaign JSON) override this.
        """
        _ = spec, new_observations
        return backend_state

    def compute_batch_diversity(
        self,
        spec: OptimizationSpec,
        candidates: list[dict[str, Any]],
    ) -> BatchDiversityMetrics | None:
        """Reuse :func:`bo_engine.batch_diversity.compute_batch_diversity`."""
        if len(candidates) < 2:
            return None
        try:
            bounds = get_bounds_tensor(spec)
            param_names = [p.name for p in spec.parameters]
            values = [[float(c.get(name, 0.0)) for name in param_names] for c in candidates]
            tensor = torch.tensor(values, device=get_device(), dtype=get_dtype())
            m = compute_batch_diversity(tensor, bounds)
            return BatchDiversityMetrics(
                min_pairwise_distance=m.min_pairwise_distance,
                mean_pairwise_distance=m.mean_pairwise_distance,
                diversity_score=m.diversity_score,
                is_diverse=m.is_diverse,
            )
        except (RuntimeError, ValueError, TypeError):
            return None

    def select_methods(
        self,
        spec: OptimizationSpec,
        n_observations: int,
    ) -> dict[str, Any]:
        """Generic method metadata. Backends with richer info override."""
        is_multi = spec.n_objectives > 1
        return {
            "model_type": self.name,
            "acquisition_function": "auto",
            "optimization_strategy": ("initial_design" if n_observations == 0 else "model_based"),
            "input_transforms": [],
            "explanation": (
                f"Default '{self.name}' strategy with {n_observations} observations."
                + (f" Multi-objective with {spec.n_objectives} targets." if is_multi else "")
            ),
            "confidence": "low" if n_observations < 5 else "medium",
            "alternatives": [],
            "warnings": [],
        }

    def compute_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        sections: frozenset[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Partial default: empty model/outlier/hyperparameter sections.

        Backends with native model fitting override this. The default
        keeps callers from crashing when a custom backend has no model
        diagnostics yet — they see ``None`` for the missing sections.
        """
        _ = spec, observations, sections, progress_callback
        return {
            "feature_importance": None,
            "loo_cv_metrics": None,
            "model_correlation": None,
            "hyperparameters": None,
            "outliers": None,
        }

    # -- State envelope helpers ------------------------------------------

    def wrap_state(
        self,
        payload: dict[str, Any] | None,
        schema_version: int = CURRENT_STATE_ENVELOPE_VERSION,
    ) -> dict[str, Any] | None:
        """Wrap a state payload in this backend's envelope."""
        return wrap_state(self.name, payload, schema_version=schema_version)

    def unwrap_state(self, data: dict[str, Any] | None) -> dict[str, Any] | None:
        """Unwrap an incoming state dict, accepting legacy bare payloads."""
        return unwrap_state(self.name, data)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Spec attribute names that map to "advanced option" bookkeeping.
# Kept in one place so the default ``_classify_options`` stays in sync with
# the option-capability tests and the BoTorch/BayBE overrides.
_SPEC_OPTION_KEYS: tuple[str, ...] = (
    "turbo_config",
    "saasbo_config",
    "fidelity_parameter",
    "transfer_learning",
    "use_cost_aware",
    "use_input_warping",
    "outcome_constraints",
)


def option_is_active(spec: OptimizationSpec, option_name: str) -> bool:
    """Return True when ``option_name`` is set to a non-default value on ``spec``."""
    value = getattr(spec, option_name, None)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return True


def required_features(spec: OptimizationSpec) -> frozenset[Feature]:
    """Detect which :class:`Feature` values a concrete spec requires.

    Used by :meth:`BaseBackend.validate_capabilities` and
    ``resolve_backend_name`` as the single source of truth for
    spec → feature inference. Centralizing the mapping makes new backends
    easier to keep in sync with the selection logic.
    """
    features: set[Feature] = set()
    if spec.n_objectives > 1:
        features.add(Feature.MULTI_OBJECTIVE)
    features.update(_parameter_type_features(spec))
    features.update(_attribute_features(spec))
    return frozenset(features)


def _parameter_type_features(spec: OptimizationSpec) -> set[Feature]:
    """Features implied by the spec's parameter type mix."""
    param_types = {str(p.type) for p in spec.parameters}
    features: set[Feature] = set()
    if "categorical" in param_types:
        features.add(Feature.CATEGORICAL)
    if "categorical" in param_types and param_types & {"continuous", "discrete"}:
        features.add(Feature.MIXED_SEARCH_SPACE)
    return features


# Attribute → feature map for the option-like spec knobs.
# ``HIGH_DIMENSIONAL`` is implied by either ``turbo_config`` or
# ``saasbo_config`` and is added separately.
_ATTR_FEATURE_MAP: tuple[tuple[str, Feature], ...] = (
    ("constraints", Feature.CONSTRAINTS),
    ("outcome_constraints", Feature.OUTCOME_CONSTRAINTS),
    ("use_cost_aware", Feature.COST_AWARE),
    ("fidelity_parameter", Feature.MULTI_FIDELITY),
    ("transfer_learning", Feature.TRANSFER_LEARNING),
    ("use_input_warping", Feature.INPUT_WARPING),
)


def _attribute_features(spec: OptimizationSpec) -> set[Feature]:
    """Features implied by truthy spec attributes (constraints, knobs)."""
    features: set[Feature] = set()
    for attr, feature in _ATTR_FEATURE_MAP:
        if option_is_active(spec, attr):
            features.add(feature)
    if spec.turbo_config is not None or spec.saasbo_config is not None:
        features.add(Feature.HIGH_DIMENSIONAL)
    return features
