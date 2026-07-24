"""BoTorch backend — wraps existing bo-engine functions behind BOBackend.

This is the default (and currently only) backend. It delegates to the
existing bo-engine modules without duplicating logic.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

# ``_SIMPLE_MC_METHODS`` is the acquisition module's own curation of the
# plain Monte-Carlo family; the capability reports below must mirror its
# constrained-run reroute exactly, so the set is shared rather than
# re-derived here.
from bo_engine.acquisition import _SIMPLE_MC_METHODS, BOTORCH_UNSUPPORTED_ACQUISITION
from bo_engine.backend import (
    Feature,
    SuggestionBatch,
)
from bo_engine.backend_base import (
    BackendInputError,
    BackendValidationResult,
    BaseBackend,
    CapabilityReport,
    CapabilityStatus,
    option_is_active,
    required_features,
    single_objective_family_acquisition_report,
    wrap_backend_exception,
)
from bo_engine.constants import (
    MIN_DATA_ABSOLUTE,
    MIN_DATA_PARAM_MULTIPLIER,
    MIN_OBSERVATIONS_FOR_LOO_CV,
    OUTLIER_DETECTION_MIN_OBSERVATIONS,
)
from bo_engine.diagnostics import (
    LOOCVMetrics,
    compute_best_value,
    compute_improvement_history,
    compute_loo_cv_for_model,
    compute_observed_hypervolume,
    compute_pareto_front,
    compute_rank_correlation,
    compute_single_objective_improvement_rate,
    extract_hyperparameters,
    observations_to_minimization_form,
    summarize_pareto_front,
)
from bo_engine.feature_importance import compute_feature_importance
from bo_engine.initial_design import SearchSpaceExhaustedError
from bo_engine.interop import (
    BAYBE_BACKEND_NAME,
    BAYBE_CUSTOM_ROLE,
    BAYBE_PARAMETER_ROLE_KEY,
    BAYBE_SUBSTANCE_ROLE,
)
from bo_engine.method_selector import select_methods
from bo_engine.models import create_and_fit_model, create_and_fit_single_task_model
from bo_engine.progress import ProgressCallback, ProgressEvent, emit
from bo_engine.result_validation import detect_outliers
from bo_engine.suggestions import (
    generate_next_batch,
    update_turbo_after_evaluation,
)
from bo_engine.suggestions_training import resolve_model_options
from bo_engine.transforms import (
    encode_categorical,
    get_bounds_tensor,
    mixed_space_combo_limit_message,
    mixed_space_combo_overflow,
)
from bo_engine.turbo import TurboState, should_use_turbo
from bo_engine.types import (
    SINGLE_OBJECTIVE_ONLY_ACQUISITION,
    UCB_FAMILY_ACQUISITION,
    AcquisitionMethod,
    ConstraintType,
    ObservationData,
    OptimizationSpec,
    ScalarizationMode,
    SuggestionResult,
    TargetMode,
)

logger = logging.getLogger(__name__)


# Covariance module of the suggestion path's GP surrogates: SingleTaskGP's
# stock kernel (BoTorch >= 0.12) is a dimension-scaled ARD ``RBFKernel``; the
# mixed-space kernel built by ``bo_engine.models.build_mixed_kernel`` adds
# per-parameter Hamming kernels on top of the same dimension-scaled RBF block.
# Reported by ``select_methods`` for diagnostics; a fitted surrogate's
# ``kernel_type`` wins when available.
_DEFAULT_KERNEL_DESCRIPTION = "RBF with automatic relevance determination (ARD)"


def _dict_to_turbo_state(data: dict[str, Any]) -> TurboState:
    """Deserialize dict to TurboState."""
    return TurboState(
        dim=data["dim"],
        batch_size=data["batch_size"],
        length=data["length"],
        length_min=data["length_min"],
        length_max=data["length_max"],
        failure_counter=data["failure_counter"],
        failure_tolerance=data["failure_tolerance"],
        success_counter=data["success_counter"],
        success_tolerance=data["success_tolerance"],
        best_value=data["best_value"],
        restart_triggered=data["restart_triggered"],
    )


def _corrupted_state_error(exc: Exception, backend_name: str) -> BackendInputError:
    """Wrap a state-restoration failure in the typed backend hierarchy.

    Persisted-state restoration (``unwrap_state`` + TurboState
    deserialization) raises raw ``KeyError``/``ValueError``/``TypeError``
    on a corrupted or foreign envelope. Those must not cross the backend
    boundary untyped — the base-class contract is that only
    :class:`BackendError` subclasses leak, so the MCP error mapper can
    build a structured, non-retryable envelope.
    """
    msg = (
        f"Backend '{backend_name}' could not restore the persisted backend "
        f"state ({type(exc).__name__}: {exc}). The stored state is corrupted "
        "or from an incompatible schema; clear it or recreate the campaign."
    )
    return BackendInputError(msg, cause=exc)


def _turbo_state_to_dict(state: TurboState) -> dict[str, Any]:
    """Serialize TurboState to dict."""
    return {
        "dim": state.dim,
        "batch_size": state.batch_size,
        "length": state.length,
        "length_min": state.length_min,
        "length_max": state.length_max,
        "failure_counter": state.failure_counter,
        "failure_tolerance": state.failure_tolerance,
        "success_counter": state.success_counter,
        "success_tolerance": state.success_tolerance,
        "best_value": state.best_value,
        "restart_triggered": state.restart_triggered,
    }


# State-payload keys. New writes keep legacy TurboState fields at the top level
# and add ``sobol_cursor`` alongside them. Once TuRBO state exists, that additive
# shape lets an older worker continue deserializing it during a rolling deployment
# because ``_dict_to_turbo_state`` reads only its known keys. A cursor-only warm-up
# payload still requires draining old workers before enabling this version for an
# explicit/high-dimensional TuRBO campaign: an old worker mistakes any non-empty
# payload for a complete TurboState. ``turbo_state`` is retained as a read-only
# compatibility key for the short-lived nested development format.
_TURBO_STATE_KEY = "turbo_state"
_SOBOL_CURSOR_KEY = "sobol_cursor"


def _validate_sobol_cursor(value: object) -> int | None:
    """Return a non-negative integer cursor or reject corrupted state."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = "sobol_cursor must be a non-negative integer or null"
        raise ValueError(msg)
    return value


def _unpack_backend_payload(
    inner_state: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, int | None]:
    """Return ``(turbo_dict, sobol_cursor)`` from an unwrapped backend payload.

    New writes are additive: ``{<legacy TurboState fields>, "sobol_cursor": int}``,
    or cursor-only when TuRBO has not started. Legacy payloads are the bare
    TurboState dict, so a missing cursor degrades gracefully to ``None`` and the
    engine falls back to the issue-count offset. The former nested development
    shape is still accepted so no locally persisted state is stranded.
    """
    if inner_state is None:
        return None, None
    if _TURBO_STATE_KEY in inner_state:
        turbo_dict = inner_state.get(_TURBO_STATE_KEY)
        if turbo_dict is not None and not isinstance(turbo_dict, dict):
            msg = "turbo_state must be an object or null"
            raise TypeError(msg)
        return turbo_dict, _validate_sobol_cursor(inner_state.get(_SOBOL_CURSOR_KEY))

    has_cursor = _SOBOL_CURSOR_KEY in inner_state
    cursor = _validate_sobol_cursor(inner_state.get(_SOBOL_CURSOR_KEY))
    turbo_dict = {key: value for key, value in inner_state.items() if key != _SOBOL_CURSOR_KEY}
    if turbo_dict:
        return turbo_dict, cursor
    # Cursor-only is a valid warm-up payload with no TuRBO state. A genuinely
    # empty legacy payload remains ``{}`` so a TuRBO-enabled campaign rejects it
    # as corrupted instead of silently treating it as absent state.
    return (None if has_cursor else inner_state), cursor


def _pack_backend_payload(
    turbo_dict: dict[str, Any] | None,
    sobol_cursor: int | None,
) -> dict[str, Any] | None:
    """Add a validated cursor to the legacy-compatible flat TuRBO payload."""
    cursor = _validate_sobol_cursor(sobol_cursor)
    if turbo_dict is None and sobol_cursor is None:
        return None
    payload = dict(turbo_dict or {})
    if cursor is not None:
        payload[_SOBOL_CURSOR_KEY] = cursor
    return payload


def _next_sobol_cursor(results: list[SuggestionResult], previous: int | None) -> int | None:
    """Advance the persisted Sobol cursor past the positions this batch consumed.

    Uses ``max(sobol_index) + 1`` over the batch's initial-design suggestions so
    the next warm-up call resumes just after the last accepted candidate (never
    skipping unused Sobol points, never rescanning). A model-guided batch stamps
    no ``sobol_index`` and leaves the cursor unchanged.
    """
    indices = [sr.sobol_index for sr in results if sr.sobol_index is not None]
    if not indices:
        return previous
    next_cursor = max(indices) + 1
    return next_cursor if previous is None else max(previous, next_cursor)


# BayBE roles encoding a representation BoTorch cannot reproduce: substance
# (SMILES → cheminformatics descriptors) and custom (labels → user-supplied
# numeric representation). Both would be one-hotted into opaque categories by
# BoTorch, silently dropping the representation — so both are hard-vetoed.
_BAYBE_UNREPRODUCIBLE_ROLES: dict[str, str] = {
    BAYBE_SUBSTANCE_ROLE: (
        "encode molecular SMILES as cheminformatics descriptors; BoTorch has no chemistry kernel"
    ),
    BAYBE_CUSTOM_ROLE: (
        "carry a user-supplied numeric representation (e.g. quantum-chemistry "
        "descriptors); BoTorch has no path to load a custom encoding"
    ),
}


def _baybe_unreproducible_role(
    parameter_options: dict[str, dict[str, Any]] | None,
) -> str | None:
    """Return the BayBE role marker BoTorch cannot reproduce, else ``None``.

    Reads the documented cross-backend interop markers (see
    :mod:`bo_engine.interop`) instead of importing ``bo_engine_baybe`` —
    bo-engine must not depend on a backend package. Accepts any mapping
    (plain ``dict`` from the server converter or a frozen view) so the
    check is robust to how the spec was built. Returns the role string
    (``substance`` / ``custom``) when it is one BoTorch must veto.
    """
    if not parameter_options:
        return None
    baybe_opts = parameter_options.get(BAYBE_BACKEND_NAME)
    if not isinstance(baybe_opts, Mapping):
        return None
    role = baybe_opts.get(BAYBE_PARAMETER_ROLE_KEY)
    return role if role in _BAYBE_UNREPRODUCIBLE_ROLES else None


def _substance_parameter_reports(spec: OptimizationSpec) -> list[CapabilityReport]:
    """Flag BayBE ``role=substance``/``role=custom`` parameters ``UNSUPPORTED`` on BoTorch.

    Both roles carry a representation BoTorch cannot reproduce — a molecular
    ``SubstanceParameter`` (SMILES → descriptor encoding) needs a chemistry
    kernel, and a ``CustomDiscreteParameter`` carries a user-supplied numeric
    encoding BoTorch has no path to load. Either way BoTorch would treat the
    labels as opaque one-hot categories, silently dropping the representation.
    Reporting them ``UNSUPPORTED`` keeps ``backend="auto"`` from mis-optimizing
    them and fails a pinned ``backend="botorch"`` loudly at intake. Ordinary
    categorical / ``role=task`` BayBE options carry no such marker and are
    untouched here, so existing categorical and task routing is unchanged.
    """
    reports: list[CapabilityReport] = []
    for p in spec.parameters:
        role = _baybe_unreproducible_role(p.parameter_options)
        if role is None:
            continue
        reports.append(
            CapabilityReport(
                key=(
                    f"parameter_options[{p.name}].{BAYBE_BACKEND_NAME}.{BAYBE_PARAMETER_ROLE_KEY}"
                ),
                status=CapabilityStatus.UNSUPPORTED,
                reason=(
                    f"BayBE-native '{role}' parameters {_BAYBE_UNREPRODUCIBLE_ROLES[role]} "
                    "and would be treated as opaque categories by BoTorch, silently "
                    "dropping the representation. Use backend='baybe' (or 'auto'). This "
                    "is a hard incompatibility and cannot be bypassed via "
                    "acknowledge_degradations."
                ),
            )
        )
    return reports


# Constraint types the BoTorch suggestion pipeline can express (natively as
# linear (in)equalities or via post-hoc projection). The extended families
# (products, cardinality, set-based label constraints) are BayBE-only.
_BOTORCH_SUPPORTED_CONSTRAINT_TYPES: frozenset[ConstraintType] = frozenset(
    {
        ConstraintType.SUM_EQUALS,
        ConstraintType.SUM_LESS_THAN,
        ConstraintType.SUM_GREATER_THAN,
        ConstraintType.LINEAR,
    }
)


def _constraint_surface_reports(spec: OptimizationSpec) -> list[CapabilityReport]:
    """Report constraint types the BoTorch engine cannot express.

    Product/cardinality/set-based constraints (and interpoint semantics)
    are honored by BayBE only; reporting them UNSUPPORTED keeps
    ``backend="auto"`` routing to BayBE and fails a pinned
    ``backend="botorch"`` loudly at intake — the substance-guardrail
    routing-safety pattern applied to constraints.
    """
    reports: list[CapabilityReport] = []
    for idx, constraint in enumerate(spec.constraints):
        key = f"constraint[{idx}]({constraint.type.value})"
        if constraint.type not in _BOTORCH_SUPPORTED_CONSTRAINT_TYPES:
            reports.append(
                CapabilityReport(
                    key=key,
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        f"Constraint type '{constraint.type.value}' is not supported "
                        "by the BoTorch engine. Use backend='baybe' (or 'auto')."
                    ),
                )
            )
        elif constraint.is_interpoint:
            reports.append(
                CapabilityReport(
                    key=key,
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        "Interpoint (across-batch) constraint semantics are not "
                        "supported by the BoTorch engine. Use backend='baybe' "
                        "(or 'auto')."
                    ),
                )
            )
    return reports


def _objective_surface_reports(spec: OptimizationSpec) -> list[CapabilityReport]:
    """Report extended-objective features the BoTorch engine cannot express.

    Match-a-target objectives, the typed target-transform union, and
    desirability scalarization are BayBE-honored neutral-spec features;
    the BoTorch suggestion pipeline only implements the boolean
    ``log_transform`` outcome stack. A ``target_mode`` of
    minimize/maximize that disagrees with the boolean ``minimize`` field
    is likewise vetoed: every direction read in this pipeline consumes
    the boolean, so honoring the spec would optimize the opposite
    direction from what BayBE resolves via ``effective_mode`` for the
    identical spec. Reporting them UNSUPPORTED keeps ``backend="auto"``
    routing to BayBE and fails a pinned ``backend="botorch"`` loudly at
    intake instead of silently optimizing the wrong objective.
    """
    reports: list[CapabilityReport] = []
    for idx, o in enumerate(spec.objectives):
        key = f"objectives[{idx}]"
        if o.target_mode == TargetMode.MATCH:
            reports.append(
                CapabilityReport(
                    key=f"{key}.target_mode",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        f"Match-a-target objectives ('{o.name}') are not supported by "
                        "the BoTorch engine. Use backend='baybe' (or 'auto')."
                    ),
                )
            )
        elif o.target_mode is not None and (o.target_mode == TargetMode.MINIMIZE) != o.minimize:
            reports.append(
                CapabilityReport(
                    key=f"{key}.target_mode",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        f"Objective '{o.name}' sets target_mode="
                        f"'{o.target_mode.value}' while its boolean minimize field "
                        f"is {o.minimize}. The BoTorch engine resolves direction "
                        "from the boolean only, so this spec would be optimized in "
                        "the opposite direction from the target_mode intent. Set "
                        "minimize consistently with target_mode, or use "
                        "backend='baybe' (or 'auto')."
                    ),
                )
            )
        if o.transform is not None:
            reports.append(
                CapabilityReport(
                    key=f"{key}.transform",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        f"Typed target transformations ('{o.name}') are not supported "
                        "by the BoTorch engine; only the boolean log_transform is. "
                        "Use backend='baybe' (or 'auto')."
                    ),
                )
            )
    if spec.scalarization == ScalarizationMode.DESIRABILITY:
        reports.append(
            CapabilityReport(
                key="scalarization",
                status=CapabilityStatus.UNSUPPORTED,
                reason=(
                    "Desirability scalarization is not supported by the BoTorch "
                    "engine (its scalarized path is qLogNParEGO with random "
                    "weights, which has different semantics). Use backend='baybe' "
                    "(or 'auto')."
                ),
            )
        )
    return reports


def _acquisition_method_reports(spec: OptimizationSpec) -> list[CapabilityReport]:
    """Report acquisition selections the BoTorch engine cannot express.

    Members of :data:`bo_engine.acquisition.BOTORCH_UNSUPPORTED_ACQUISITION`
    follow the standard degradable-knob policy (the BayBE
    ``BAYBE_UNSUPPORTED_ACQUISITION`` pattern): UNSUPPORTED by default so
    ``backend="auto"`` routes to a backend that honors the request and a
    pinned ``backend="botorch"`` fails loudly; IGNORED once
    ``'acquisition_method'`` is acknowledged — the run then falls back to
    the objective-family default acquisition. The same policy classifies
    a single-objective-only member requested on a multi-objective spec
    (the dispatch would silently discard the request — and any
    ``acquisition_beta`` riding on it — in favor of the hypervolume
    default). A simple-MC member combined with outcome constraints is
    DEGRADED: the engine reroutes it through NOISY_EI's feasibility
    weighting at construction, which changes the acquisition semantics
    while still honoring the constraints. A stray ``acquisition_beta``
    on a non-UCB method is a hard UNSUPPORTED (the engine factory raises
    the matching ``ValueError`` at construction, so intake must agree).
    """
    reports: list[CapabilityReport] = []
    method = spec.acquisition_method
    if method in BOTORCH_UNSUPPORTED_ACQUISITION:
        if "acquisition_method" in spec.acknowledge_degradations:
            reports.append(
                CapabilityReport(
                    key="acquisition_method",
                    status=CapabilityStatus.IGNORED,
                    reason=(
                        f"Acquisition method '{method.value}' is not supported by the "
                        "BoTorch engine; the objective-family default acquisition is "
                        "used instead (degradation acknowledged)."
                    ),
                )
            )
        else:
            reports.append(
                CapabilityReport(
                    key="acquisition_method",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=(
                        f"Acquisition method '{method.value}' is not supported by the "
                        "BoTorch engine. Pin backend='baybe' (or 'auto'), or list "
                        "'acquisition_method' in 'acknowledge_degradations' to run "
                        "with the objective-family default acquisition instead."
                    ),
                )
            )
    elif len(spec.objectives) > 1 and method in SINGLE_OBJECTIVE_ONLY_ACQUISITION:
        reports.append(
            single_objective_family_acquisition_report(method, spec.acknowledge_degradations)
        )
    if len(spec.objectives) == 1 and spec.outcome_constraints and method in _SIMPLE_MC_METHODS:
        reports.append(
            CapabilityReport(
                key="acquisition_method",
                status=CapabilityStatus.DEGRADED,
                reason=(
                    f"Acquisition method '{method.value}' has no "
                    "feasibility-weighted wiring in the BoTorch engine; with "
                    "outcome constraints active it is rerouted through NOISY_EI "
                    "so the constraints are honored, which changes the "
                    "acquisition semantics. Remove the outcome constraints or "
                    "select noisy_expected_improvement to silence this report."
                ),
            )
        )
    if spec.acquisition_beta is not None and method not in UCB_FAMILY_ACQUISITION:
        reports.append(
            CapabilityReport(
                key="acquisition_beta",
                status=CapabilityStatus.UNSUPPORTED,
                reason=(
                    "acquisition_beta is only valid for the UCB acquisition family; "
                    f"got acquisition_method='{method.value}'. Remove beta or select "
                    "upper_confidence_bound."
                ),
            )
        )
    return reports


# Features whose campaign routing is not wired end to end: the standalone
# modules exist, but ``generate_next_batch`` raises the matching typed
# error instead of dispatching. Reported ``UNSUPPORTED`` (hard veto) by
# ``validate_capabilities`` so intake rejects at create time rather than
# failing at suggestion time — mirroring the static ``supported_features``
# exclusions so the two surfaces cannot drift apart.
_UNROUTED_FEATURES: dict[Feature, str] = {
    Feature.MULTI_FIDELITY: (
        "Multi-fidelity (qMFKG / SingleTaskMultiFidelityGP) is not routed "
        "through the BoTorch suggestion pipeline: generate_next_batch "
        "rejects fidelity_parameter / MULTI_FIDELITY_KG with a typed "
        "MultiFidelityNotSupportedError. Remove the fidelity settings, or "
        "drive the standalone bo_engine.multifidelity helpers directly."
    ),
    Feature.TRANSFER_LEARNING: (
        "RGPE transfer learning is not routed through the BoTorch "
        "suggestion pipeline: generate_next_batch rejects "
        "spec.transfer_learning with a typed "
        "TransferLearningNotSupportedError instead of silently running a "
        "plain GP without the requested prior-campaign transfer. Remove "
        "transfer_learning, use BayBE's native TaskParameter mechanism, or "
        "drive the standalone bo_engine.transfer_learning helpers directly."
    ),
}

# Option-key view of the same un-routed paths. Spec options and inferred
# features are reported separately, so each un-routed path must be vetoed
# on BOTH surfaces — otherwise a spec with e.g. ``transfer_learning``
# reports the feature UNSUPPORTED while the option claims SUPPORTED, and
# clients inspecting option reports get contradictory signals.
_UNROUTED_OPTION_REASONS: dict[str, str] = {
    "saasbo_config": (
        "SAASBO is not routed through the BoTorch suggestion pipeline: "
        "generate_next_batch rejects saasbo_config with a typed "
        "SAASBONotSupportedError instead of silently fitting a dense-ARD "
        "GP under a sparse-prior advertisement. Remove saasbo_config, or "
        "drive the standalone bo_engine.saasbo helpers directly."
    ),
    "fidelity_parameter": _UNROUTED_FEATURES[Feature.MULTI_FIDELITY],
    "transfer_learning": _UNROUTED_FEATURES[Feature.TRANSFER_LEARNING],
}


def _log_transform_direction_reports(spec: OptimizationSpec) -> list[CapabilityReport]:
    """UNSUPPORTED report per ``log_transform=True`` + ``minimize=False`` objective.

    The suggestion pipeline rejects this combination at generation time
    (``ObjectiveSpec.log_transform`` is contractually restricted to
    minimize objectives — see the guards in
    :mod:`bo_engine.suggestions_single_objective` and
    :mod:`bo_engine.suggestions_multi_objective`). Reporting it here
    keeps ``validate_capabilities`` aligned with that runtime contract,
    so a pinned ``backend="botorch"`` campaign is rejected at intake
    instead of failing on its first suggestion batch.
    """
    return [
        CapabilityReport(
            key=f"objectives[{idx}].log_transform",
            status=CapabilityStatus.UNSUPPORTED,
            reason=(
                f"ObjectiveSpec.log_transform=True is only supported for "
                f"minimize=True objectives (objective[{idx}] '{obj.name}' is "
                "maximize). Flip the objective definition or pre-transform "
                "the data."
            ),
        )
        for idx, obj in enumerate(spec.objectives)
        if obj.log_transform and not obj.minimize
    ]


@dataclass(frozen=True)
class _DiagnosticModelFit:
    """A GP fit on the observations, shared across diagnostics sections.

    Fitting the GP dominates the cost of a diagnostics call. The ``model`` and
    ``suggestions_tensor`` sections both consume the same fit, so it is computed
    once per :meth:`BoTorchBackend.compute_diagnostics` invocation and reused
    rather than refit per section.
    """

    model: SingleTaskGP | ModelListGP
    train_x: torch.Tensor
    train_y: torch.Tensor
    bounds: torch.Tensor
    param_names: list[str]
    obj_names: list[str]
    is_single: bool


class BoTorchBackend(BaseBackend):
    """BoTorch-based Bayesian Optimization backend.

    Wraps existing bo-engine functions behind the BOBackend protocol.
    Inherits Sobol initial design, duplicate detection, batch diversity,
    and JSON state-envelope helpers from :class:`BaseBackend`; overrides
    metric and diagnostic helpers that need BoTorch-specific GP work.
    """

    @property
    def name(self) -> str:
        """Backend identifier used by the registry."""
        return "botorch"

    @property
    def supported_features(self) -> frozenset[Feature]:
        """Features this backend supports **unconditionally**.

        ``Feature.MULTI_FIDELITY`` and ``Feature.TRANSFER_LEARNING`` are
        deliberately excluded — ``bo_engine.multifidelity`` and
        ``bo_engine.transfer_learning`` provide standalone helpers
        (qMFKG, RGPE), but the active ``generate_next_batch`` pipeline
        dispatches neither: it raises the typed
        ``MultiFidelityNotSupportedError`` / ``TransferLearningNotSupportedError``
        instead of silently downgrading to single-fidelity / no-transfer
        behaviour. Until the suggestion pipeline routes them end to end,
        advertising the capabilities would lie to callers.
        """
        # Note: ``frozenset - dict.keys()`` falls back to dict_keys'
        # __rsub__ and returns a MUTABLE set — build explicitly instead
        # to honor the protocol's frozenset contract.
        return frozenset(f for f in Feature if f not in _UNROUTED_FEATURES)

    def validate_capabilities(self, spec: OptimizationSpec) -> BackendValidationResult:
        """BoTorch supports every neutral feature and most options in the spec.

        The exceptions are reported ``UNSUPPORTED`` so ``backend="auto"``
        treats them as a hard veto and a pinned ``backend="botorch"`` is
        rejected loudly at intake:

        * A BayBE-native ``role=substance`` parameter (see
          :mod:`bo_engine.interop`). It encodes a molecular SMILES →
          cheminformatics-descriptor representation that BoTorch has no
          kernel for; routing it here would treat the SMILES labels as
          opaque one-hot categories and silently drop the chemistry —
          the worst class of BO bug (a wrong answer that looks fine).
          This veto is deliberately *not* routed through
          ``acknowledge_degradations``: that field downgrades
          silently-dropped *option* knobs, not a misrepresented
          molecular parameter, so it cannot bypass the substance gate.
        * Un-routed pipeline paths, mirroring the static
          ``supported_features`` exclusions (``_UNROUTED_FEATURES``):
          multi-fidelity (``fidelity_parameter`` /
          ``MULTI_FIDELITY_KG``), RGPE ``transfer_learning``, and
          ``saasbo_config``. ``generate_next_batch`` rejects each with a
          typed error, so advertising them ``SUPPORTED`` would accept the
          campaign at intake only to fail at suggestion time. The
          standalone ``bo_engine.multifidelity`` /
          ``bo_engine.transfer_learning`` / ``bo_engine.saasbo`` helpers
          remain available for direct workflows.
        * ``log_transform=True`` on a ``minimize=False`` objective
          (``_log_transform_direction_reports``): the suggestion
          pipeline rejects the combination at generation time, so the
          capability surface must veto it at intake.
        """
        feature_reports = []
        for feature in sorted(required_features(spec)):
            if feature in _UNROUTED_FEATURES:
                feature_reports.append(
                    CapabilityReport(
                        key=str(feature),
                        status=CapabilityStatus.UNSUPPORTED,
                        reason=_UNROUTED_FEATURES[feature],
                    )
                )
            else:
                feature_reports.append(
                    CapabilityReport(key=str(feature), status=CapabilityStatus.SUPPORTED)
                )

        from bo_engine.backend_base import _SPEC_OPTION_KEYS

        option_reports = [
            CapabilityReport(key=key, status=CapabilityStatus.SUPPORTED)
            for key in _SPEC_OPTION_KEYS
            if option_is_active(spec, key) and key not in _UNROUTED_OPTION_REASONS
        ]
        option_reports.extend(
            CapabilityReport(key=key, status=CapabilityStatus.UNSUPPORTED, reason=reason)
            for key, reason in _UNROUTED_OPTION_REASONS.items()
            if option_is_active(spec, key)
        )
        # ``required_features`` infers MULTI_FIDELITY from
        # ``fidelity_parameter`` only — a spec selecting the qMFKG
        # acquisition without a fidelity parameter slips past the feature
        # report, yet generate_next_batch rejects it identically.
        if spec.acquisition_method == AcquisitionMethod.MULTI_FIDELITY_KG:
            option_reports.append(
                CapabilityReport(
                    key="acquisition_method",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=_UNROUTED_FEATURES[Feature.MULTI_FIDELITY],
                )
            )
        option_reports.extend(_acquisition_method_reports(spec))
        option_reports.extend(_objective_surface_reports(spec))
        option_reports.extend(_constraint_surface_reports(spec))
        option_reports.extend(_log_transform_direction_reports(spec))
        option_reports.extend(_substance_parameter_reports(spec))
        combo_overflow = mixed_space_combo_overflow(spec)
        if combo_overflow is not None:
            option_reports.append(
                CapabilityReport(
                    key="parameters",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason=mixed_space_combo_limit_message(combo_overflow),
                )
            )
        return BackendValidationResult(
            backend=self.name,
            feature_reports=tuple(feature_reports),
            option_reports=tuple(option_reports),
        )

    # ----- Suggestion Generation -----

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
        """Build a GP, optimize the acquisition function and return the next batch.

        ``initial_design_history`` (when supplied) is the authoritative set of
        Sobol positions already consumed by this campaign's warm-up; it drives
        the deterministic continuation so the sequence cannot rewind, while
        ``pending_points`` stays strictly actionable and is forwarded as
        ``X_pending``.
        """
        # Accept either the new envelope or a legacy bare payload. State
        # restoration runs inside the typed-error boundary: a corrupted
        # persisted envelope (foreign backend, missing TurboState keys)
        # must surface as a typed BackendError, not a raw KeyError.
        try:
            inner_state = self.unwrap_state(backend_state)
            turbo_dict, sobol_cursor = _unpack_backend_payload(inner_state)
            turbo_state = None
            use_turbo = spec.use_turbo or should_use_turbo(spec.n_parameters)
            if use_turbo and spec.n_objectives == 1 and turbo_dict is not None:
                turbo_state = _dict_to_turbo_state(turbo_dict)
        except (KeyError, TypeError, ValueError) as exc:
            raise _corrupted_state_error(exc, self.name) from exc

        emit(
            progress_callback,
            ProgressEvent(
                phase="generate_suggestions_start",
                message=(
                    f"Fitting GP and optimizing acquisition for {batch_size} "
                    f"suggestion(s) on {len(observations)} observation(s)"
                ),
            ),
        )

        # SearchSpaceExhaustedError is a domain signal interpreted by the
        # operations layer; it is not a backend bug, so let it propagate
        # untouched. All other library-level exceptions are wrapped via
        # ``wrap_backend_exception`` so the MCP error mapper sees a
        # typed :class:`BackendError` rather than a leaky torch/gpytorch
        # exception class.
        try:
            results, new_turbo = generate_next_batch(
                spec=spec,
                observations=observations,
                batch_size=batch_size,
                iteration=iteration,
                turbo_state=turbo_state,
                pending_points=pending_points,
                initial_design_history=initial_design_history,
                sobol_cursor=sobol_cursor,
            )
        except SearchSpaceExhaustedError:
            raise
        except Exception as exc:
            raise wrap_backend_exception(exc, backend_name=self.name) from exc

        emit(
            progress_callback,
            ProgressEvent(
                phase="generate_suggestions_done",
                message=f"Produced {len(results)} suggestion(s)",
                progress=float(len(results)),
                total=float(batch_size),
            ),
        )

        suggestions = [
            {
                "parameter_values": sr.parameter_values,
                "provenance": {
                    "iteration": sr.iteration,
                    "batch_index": sr.batch_index,
                    "generation_method": sr.generation_method,
                    "acquisition_value": sr.acquisition_value,
                    "model_uncertainty": sr.model_uncertainty,
                    "acquisition_function": sr.acquisition_function,
                    "model_type": sr.model_type,
                    "random_seed": sr.random_seed,
                    "model_version": sr.model_version,
                    "confidence_level": sr.confidence_level,
                    "explanation": sr.explanation,
                    "predicted_objectives": sr.predicted_objectives,
                    "predicted_std": sr.predicted_std,
                },
            }
            for sr in results
        ]

        new_turbo_dict = _turbo_state_to_dict(new_turbo) if new_turbo else None
        new_cursor = _next_sobol_cursor(results, sobol_cursor)
        new_state = self.wrap_state(_pack_backend_payload(new_turbo_dict, new_cursor))
        method_info = self.select_methods(spec, len(observations))

        batch_warnings: list[str] = []
        if spec.use_cost_aware and not any(o.cost is not None for o in observations):
            batch_warnings.append(
                "Cost-aware optimization was requested but no observations have "
                "cost data in metadata. Falling back to standard optimization. "
                "Add 'cost' to result metadata to enable cost weighting."
            )

        # ``model_warnings`` is stamped onto every SuggestionResult in the
        # batch — collapse duplicates before lifting them to the SuggestionBatch.
        seen: set[str] = set()
        for sr in results:
            for warning in sr.model_warnings:
                if warning in seen:
                    continue
                seen.add(warning)
                batch_warnings.append(warning)

        return SuggestionBatch(
            suggestions=suggestions,
            method_info=method_info,
            backend_state=new_state,
            warnings=batch_warnings,
        )

    # ----- Metrics -----

    def compute_hypervolume(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> float | None:
        """Return the hypervolume of the observed Pareto front, or ``None`` if not applicable.

        Delegates to the shared :func:`compute_observed_hypervolume` so the
        ``None``/``0.0``/value contract and the reference point are identical
        across backends (the BayBE backend delegates to the same helper).
        """
        return compute_observed_hypervolume(spec, observations)

    def update_state_after_results(
        self,
        spec: OptimizationSpec,
        new_observations: list[ObservationData],
        backend_state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Advance the TuRBO trust-region state given newly observed results."""
        if spec.n_objectives != 1 or backend_state is None:
            return None

        # Same typed-error boundary as generate_suggestions: corrupted
        # persisted state must not leak raw KeyError/ValueError.
        try:
            inner = self.unwrap_state(backend_state)
            turbo_dict, sobol_cursor = _unpack_backend_payload(inner)
            if turbo_dict is None:
                # No TuRBO state to advance (e.g. still in warm-up). Return None so
                # the caller keeps the existing persisted state — which may carry
                # the Sobol continuation cursor — rather than wiping it.
                return None
            turbo_state = _dict_to_turbo_state(turbo_dict)
        except (KeyError, TypeError, ValueError) as exc:
            raise _corrupted_state_error(exc, self.name) from exc
        new_turbo = update_turbo_after_evaluation(
            turbo_state=turbo_state,
            new_observations=new_observations,
            spec=spec,
        )
        # Preserve the Sobol cursor alongside the advanced TuRBO state.
        return self.wrap_state(_pack_backend_payload(_turbo_state_to_dict(new_turbo), sobol_cursor))

    def select_methods(
        self,
        spec: OptimizationSpec,
        n_observations: int,
    ) -> dict[str, Any]:
        """Return method metadata (model type, acquisition, transforms) for diagnostics."""
        ms = select_methods(spec, n_observations)
        return {
            "model_type": ms.model_type,
            "acquisition_function": ms.acquisition_function,
            "optimization_strategy": ms.optimization_strategy,
            "kernel": _DEFAULT_KERNEL_DESCRIPTION,
            "input_transforms": ms.input_transforms,
            "explanation": ms.explanation,
            "confidence": ms.confidence,
            "alternatives": ms.alternatives,
            "warnings": ms.warnings,
        }

    # ----- Diagnostics -----

    def compute_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        sections: frozenset[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Compute model-based diagnostics using BoTorch/GPyTorch."""
        all_sections = frozenset(["objectives", "model", "outliers", "suggestions_tensor"])
        requested = all_sections if sections is None else sections
        result: dict[str, Any] = {}
        is_single = spec.n_objectives == 1
        completed = 0
        total = float(len(requested))

        def announce(phase: str, message: str) -> None:
            nonlocal completed
            emit(
                progress_callback,
                ProgressEvent(
                    phase=phase,
                    message=message,
                    progress=float(completed),
                    total=total,
                ),
            )

        announce("diagnostics_start", f"Computing diagnostics: {sorted(requested)}")

        # The ``model`` and ``suggestions_tensor`` sections both consume a GP
        # fit on the same data. Fit it at most once per call and share the
        # result instead of refitting per section.
        shared_fit: _DiagnosticModelFit | None = None
        fit_computed = False

        def diagnostic_fit() -> _DiagnosticModelFit | None:
            nonlocal shared_fit, fit_computed
            if not fit_computed:
                shared_fit = self._fit_diagnostic_model(spec, observations, is_single)
                fit_computed = True
            return shared_fit

        if "objectives" in requested:
            result.update(self._compute_objective_diagnostics(spec, observations))
            completed += 1
            announce("diagnostics_objectives_done", "Objective metrics complete")

        if "model" in requested:
            result.update(self._compute_model_diagnostics(diagnostic_fit()))
            completed += 1
            announce("diagnostics_model_done", "Model fitting and LOO-CV complete")

        if "outliers" in requested:
            result.update(self._compute_outlier_diagnostics(spec, observations))
            completed += 1
            announce("diagnostics_outliers_done", "Outlier detection complete")

        if "suggestions_tensor" in requested:
            result.update(self._compute_hyperparameters(diagnostic_fit()))
            completed += 1
            announce("diagnostics_hyperparameters_done", "Hyperparameter extraction complete")

        return result

    def _compute_objective_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Compute single/multi-objective metrics (best value, Pareto, HV)."""
        if spec.n_objectives == 1:
            return self._compute_single_objective(spec, observations)
        return self._compute_multi_objective(spec, observations)

    def _compute_single_objective(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Compute best value and improvement history for single-objective."""
        obj = spec.objectives[0]
        values = [obs.objective_values[obj.name] for obs in observations]
        result: dict[str, Any] = {
            "pareto_front": None,
            "hypervolume": None,
            "n_pareto_points": None,
        }
        if values:
            best_val, best_idx = compute_best_value(values, minimize=obj.minimize)
            imp_hist = compute_improvement_history(values, minimize=obj.minimize)
            result["best_value"] = best_val
            result["best_parameters"] = observations[best_idx].parameter_values
            result["improvement_history"] = imp_hist
            result["improvement_rate"] = compute_single_objective_improvement_rate(imp_hist)
        else:
            result["best_value"] = None
            result["best_parameters"] = None
            result["improvement_history"] = []
            result["improvement_rate"] = 0.0
        return result

    def _compute_multi_objective(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Compute Pareto front and hypervolume for multi-objective.

        ``hypervolume`` is delegated to :func:`compute_observed_hypervolume`
        so this live diagnostics figure uses the same pinned reference
        point as ``campaign.hypervolume_history`` (see M75) — a moving,
        recomputed-per-call reference point would let a dominated,
        worse-in-one-objective observation inflate this field even though
        the Pareto front (and the history) did not improve.
        """
        obj_names = [o.name for o in spec.objectives]
        result: dict[str, Any] = {
            "best_value": None,
            "best_parameters": None,
            "improvement_history": None,
            "improvement_rate": None,
        }
        if len(observations) < 2:
            result["pareto_front"] = []
            result["hypervolume"] = 0.0
            result["n_pareto_points"] = 0
            return result

        y_bo, minimize_mask = observations_to_minimization_form(spec, observations)

        pareto_y, _ = compute_pareto_front(y_bo)
        pareto_display = pareto_y.clone()
        pareto_display[:, ~minimize_mask] = -pareto_display[:, ~minimize_mask]

        result["pareto_front"] = summarize_pareto_front(pareto_display, obj_names)
        result["hypervolume"] = compute_observed_hypervolume(spec, observations)
        result["n_pareto_points"] = len(result["pareto_front"])
        return result

    def _fit_diagnostic_model(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        is_single: bool,
    ) -> _DiagnosticModelFit | None:
        """Fit one GP on the observations for the diagnostics sections.

        The model options (fixed noise, log transform, categorical kernel,
        noise prior, warping) come from the same
        :func:`~bo_engine.suggestions_training.resolve_model_options` block
        the suggestion builders use, so the reported noise variance / LOO
        metrics / kernel type describe the production surrogate rather than
        a plain trainable-noise RBF GP. ``target_negated`` stays ``False``
        because ``_prepare_training_data`` returns raw (un-negated) targets.

        Returns ``None`` when there is insufficient data to fit (the same gate
        the model and hyperparameter sections previously applied individually)
        or when the fit raises, so each section can fall back to its own empty
        result.
        """
        n_params = len(spec.parameters)
        min_data = max(MIN_DATA_ABSOLUTE, MIN_DATA_PARAM_MULTIPLIER * n_params)
        if len(observations) < min_data:
            return None

        try:
            train_x, train_y, param_names, obj_names = self._prepare_training_data(
                spec, observations
            )
            bounds = get_bounds_tensor(spec)
            options = resolve_model_options(spec, observations)

            if is_single:
                model: SingleTaskGP | ModelListGP = create_and_fit_single_task_model(
                    train_x,
                    train_y,
                    bounds,
                    use_input_warping=options.use_input_warping,
                    train_yvar=options.train_yvar,
                    log_transform=options.log_flags[0],
                    categorical_blocks=options.categorical_blocks,
                    auto_shift_for_log=options.auto_shift_for_log,
                    noise_prior=options.noise_prior,
                    target_negated=False,
                )
            else:
                model = create_and_fit_model(
                    train_x,
                    train_y,
                    bounds,
                    use_input_warping=options.use_input_warping,
                    train_yvar=options.train_yvar,
                    log_transform=options.log_flags,
                    categorical_blocks=options.categorical_blocks,
                    noise_prior=options.noise_prior,
                    target_negated=False,
                )
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Diagnostic model fit failed: %s", e)
            return None

        return _DiagnosticModelFit(
            model=model,
            train_x=train_x,
            train_y=train_y,
            bounds=bounds,
            param_names=param_names,
            obj_names=obj_names,
            is_single=is_single,
        )

    def _compute_model_diagnostics(
        self,
        fit: _DiagnosticModelFit | None,
    ) -> dict[str, Any]:
        """Compute correlation, feature importance and LOO-CV from a shared GP fit."""
        empty: dict[str, Any] = {
            "feature_importance": None,
            "loo_cv_metrics": None,
            "model_correlation": None,
        }
        if fit is None:
            return empty

        try:
            corr = self._model_correlation(fit.model, fit.train_x, fit.train_y, fit.is_single)
            # "Not measurable" crosses the envelope boundary as None so
            # transports never serialize NaN and callers keep their
            # None-aware handling (a measured 0.0 stays 0.0).
            model_correlation = None if math.isnan(corr) else corr
            fi = compute_feature_importance(
                fit.model,
                fit.train_x,
                fit.param_names,
                include_shap=False,
            )
            loo = self._loo_cv(
                fit.model,
                fit.train_x,
                fit.train_y,
                fit.bounds,
                fit.obj_names,
                fit.train_x.shape[0],
            )
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Model diagnostics failed: %s", e)
            return empty
        return {
            "model_correlation": model_correlation,
            "feature_importance": fi,
            "loo_cv_metrics": loo,
        }

    def _model_correlation(
        self,
        model: SingleTaskGP | ModelListGP,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        is_single: bool,
    ) -> float:
        """Compute rank correlation between model predictions and actuals.

        Multi-objective campaigns average the per-objective correlations over
        the objectives where the correlation is measurable; ``NaN`` (nothing
        measurable) means "unknown", not zero.
        """
        model.eval()
        with torch.no_grad():
            predictions = model.posterior(train_x).mean
        if is_single:
            return compute_rank_correlation(predictions.squeeze(), train_y.squeeze())
        corrs = [
            compute_rank_correlation(predictions[:, i], train_y[:, i])
            for i in range(train_y.shape[1])
        ]
        finite = [c for c in corrs if not math.isnan(c)]
        if not finite:
            return float("nan")
        return sum(finite) / len(finite)

    def _loo_cv(
        self,
        model: SingleTaskGP | ModelListGP,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        bounds: torch.Tensor,
        obj_names: list[str],
        n_obs: int,
    ) -> dict[str, dict[str, float]] | None:
        """Compute LOO-CV metrics per objective."""
        if n_obs < MIN_OBSERVATIONS_FOR_LOO_CV:
            return None
        try:
            loo = compute_loo_cv_for_model(model, train_x, train_y, bounds)
            loo_by_obj: dict[str, dict[str, float]] = {}
            if isinstance(loo, dict):
                loo_dict = cast("dict[int, LOOCVMetrics]", loo)
                for idx, name in enumerate(obj_names):
                    if idx in loo_dict:
                        loo_by_obj[name] = self._loo_to_dict(loo_dict[idx])
            elif obj_names:
                loo_by_obj[obj_names[0]] = self._loo_to_dict(loo)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("LOO-CV failed: %s", e)
            return None
        return loo_by_obj

    def _compute_outlier_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Detect outliers via LOO residuals."""
        if len(observations) < OUTLIER_DETECTION_MIN_OBSERVATIONS:
            return {"outliers": None}

        try:
            # ``detect_outliers`` reports actual/predicted values back to the
            # user, so it consumes raw user-scale targets (standardized
            # residuals are sign-invariant).
            train_x, train_y_raw, _, obj_names = self._prepare_training_data(spec, observations)
            bounds = get_bounds_tensor(spec)

            outliers = detect_outliers(
                train_x=train_x,
                train_y=train_y_raw,
                bounds=bounds,
                objective_names=obj_names,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Outlier detection failed: %s", e)
            return {"outliers": None}

        if outliers:
            info = [
                {
                    "result_index": o.index,
                    "standardized_error": round(o.standardized_error, 2),
                    "actual_value": round(o.actual_value, 4),
                    "predicted_value": round(o.predicted_value, 4),
                    "objective": o.objective_name,
                    "parameter_values": observations[o.index].parameter_values,
                }
                for o in outliers
            ]
            return {
                "outliers": {
                    "count": len(outliers),
                    "outlier_results": info,
                }
            }
        return {"outliers": {"count": 0, "outlier_results": []}}

    def _compute_hyperparameters(
        self,
        fit: _DiagnosticModelFit | None,
    ) -> dict[str, Any]:
        """Extract GP hyperparameters from a shared diagnostic GP fit."""
        if fit is None:
            return {"hyperparameters": None}

        try:
            hp = extract_hyperparameters(fit.model, fit.param_names)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Hyperparameter extraction failed: %s", e)
            return {"hyperparameters": None}
        return {
            "hyperparameters": {
                "lengthscales": hp.lengthscales,
                "noise_variance": hp.noise_variance,
                "output_scale": hp.output_scale,
                "kernel_type": hp.kernel_type,
                "model_type": hp.model_type,
            }
        }

    def _prepare_training_data(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
        """Build train_x and train_y tensors from observations.

        ``train_y`` is returned on the **raw user scale** (no direction
        negation). Every consumer in the diagnostics layer is
        direction-agnostic — LOO RMSE/MAE/R², rank correlation and feature
        importance are invariant under a joint sign flip of targets and
        predictions — and the outlier report must surface actual/predicted
        values on the user's scale.
        """
        param_names = [p.name for p in spec.parameters]
        obj_names = [o.name for o in spec.objectives]

        x_list = [encode_categorical(obs.parameter_values, spec) for obs in observations]
        train_x = torch.stack(x_list)

        y_list = [
            torch.tensor([obs.objective_values[n] for n in obj_names], dtype=torch.double)
            for obs in observations
        ]
        train_y = torch.stack(y_list)

        return train_x, train_y, param_names, obj_names

    @staticmethod
    def _loo_to_dict(m: LOOCVMetrics) -> dict[str, float]:
        return {"rmse": m.rmse, "mae": m.mae, "r_squared": m.r_squared}
