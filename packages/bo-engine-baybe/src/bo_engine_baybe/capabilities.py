"""BayBE capability reporting helpers.

Split from :mod:`bo_engine_baybe.backend` so the static capability matrix
(supported / conditional / degradable features and the parameter-role
validators) lives in one place. The :class:`~bo_engine_baybe.backend.BayBEBackend`
class consumes these constants and helpers when building its
:class:`BackendValidationResult`.
"""

from __future__ import annotations

from typing import Any

import pydantic

from bo_engine.backend import Feature
from bo_engine.backend_base import (
    CapabilityReport,
    CapabilityStatus,
)
from bo_engine.types import OptimizationSpec, ParameterSpec, ParameterType
from bo_engine_baybe.options import (
    BayBEParameterOptions,
    BayBEParameterRole,
    extract_baybe_parameter_options,
)


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

# Features BayBE supports *unconditionally* — independent of spec
# shape. ``Feature.TRANSFER_LEARNING`` is intentionally absent because
# BayBE only honours transfer learning when the spec declares a
# ``TaskParameter`` via ``parameter_options['baybe'].role == 'task'``;
# advertising it here would lie to ``list_capabilities`` callers that
# do not know the precondition. ``_feature_reports`` and
# ``_task_parameter_feature_report`` continue to flip TRANSFER_LEARNING
# to ``SUPPORTED`` when a TaskParameter is actually present, so a spec
# that exercises the feature still resolves correctly through
# :meth:`validate_capabilities`.
_SUPPORTED_FEATURES = frozenset(
    {
        Feature.MULTI_OBJECTIVE,
        Feature.CONSTRAINTS,
        Feature.CATEGORICAL,
        Feature.MIXED_SEARCH_SPACE,
    }
)

# Conditional features BayBE can support — only on specs that satisfy
# the documented precondition. Used by ``list_capabilities`` to annotate
# the static surface so LLM clients and humans can see what activates
# each conditional feature.
_CONDITIONAL_FEATURES: dict[Feature, str] = {
    Feature.TRANSFER_LEARNING: (
        "Requires a parameter with parameter_options['baybe'].role == 'task' "
        "(BayBE-native TaskParameter)."
    ),
}


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


def _parameter_is_task(parameter_options: dict[str, dict[str, Any]] | None) -> bool:
    """Return True when ``parameter_options['baybe'].role == 'task'``."""
    try:
        opts = extract_baybe_parameter_options(parameter_options)
    except pydantic.ValidationError:
        return False
    return opts.role == BayBEParameterRole.TASK


def _validate_parameter_role(
    p: ParameterSpec,
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
    substance_labels = set(opts.substance_data)
    missing = sorted(categories - substance_labels)
    extra = sorted(substance_labels - categories)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing SMILES for categories {missing}")
        if extra:
            parts.append(f"extra SMILES for undeclared categories {extra}")
        reports.append(
            CapabilityReport(
                key=f"parameter_options[{name}].baybe.substance_data",
                status=CapabilityStatus.UNSUPPORTED,
                reason="BayBE substance_data has category mismatch: " + "; ".join(parts),
            )
        )
        return reports
    # SMILES parse-check runs last and ONLY when chemistry extras are
    # installed: a malformed SMILES passes the dict[str, str] schema but
    # crashes inside RDKit at SubstanceParameter construction, so we fail
    # loud at intake instead. Ordering is load-bearing — the chem-missing
    # path above (real or monkeypatched) must never reach the lazy RDKit
    # import, so it stays behind this guard.
    if _CHEMISTRY_AVAILABLE:
        reports.extend(_invalid_smiles_reports(name, dict(opts.substance_data)))
    return reports


def _invalid_smiles_reports(name: str, substance_data: dict[str, str]) -> list[CapabilityReport]:
    """Report invalid or duplicate SMILES as ``UNSUPPORTED``.

    Lazy-imports RDKit so the import only happens when chemistry extras are
    present. The sole caller already gates on ``_CHEMISTRY_AVAILABLE``; a
    chem-missing run never reaches here. ``rdBase.BlockLogs`` suppresses
    parse-error spam while canonical SMILES let us catch duplicate molecules
    before BayBE's ``SubstanceParameter`` constructor raises later.
    """
    from collections import defaultdict

    from rdkit import Chem, rdBase

    canonical_by_category: dict[str, str] = {}
    invalid: list[str] = []
    with rdBase.BlockLogs():
        for category, smiles in substance_data.items():
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                invalid.append(category)
                continue
            canonical_by_category[category] = Chem.MolToSmiles(molecule, canonical=True)

    reports: list[CapabilityReport] = []
    if invalid:
        reports.append(
            CapabilityReport(
                key=f"parameter_options[{name}].baybe.substance_data",
                status=CapabilityStatus.UNSUPPORTED,
                reason=(
                    f"BayBE substance_data contains SMILES that RDKit cannot parse for "
                    f"categories {sorted(invalid)}; fix the SMILES strings."
                ),
            )
        )

    categories_by_smiles: dict[str, list[str]] = defaultdict(list)
    for category, canonical in canonical_by_category.items():
        categories_by_smiles[canonical].append(category)
    duplicates = [
        sorted(categories) for categories in categories_by_smiles.values() if len(categories) > 1
    ]
    if duplicates:
        reports.append(
            CapabilityReport(
                key=f"parameter_options[{name}].baybe.substance_data",
                status=CapabilityStatus.UNSUPPORTED,
                reason=(
                    "BayBE substance_data contains labels that resolve to the same "
                    f"substance: {sorted(duplicates)}; keep one label per molecule."
                ),
            )
        )
    return reports
