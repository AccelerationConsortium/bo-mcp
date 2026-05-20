"""BayBE capability classification: semantic knobs are UNSUPPORTED by default.

Semantically load-bearing options (``use_input_warping``,
``use_cost_aware``, ``outcome_constraints``, ``turbo_config``,
``saasbo_config``, ``fidelity_parameter``) are silently dropped at
BayBE runtime. Reporting them as ``IGNORED`` lets the campaign accept
and run with the wrong semantics — the worst class of BO bug, wrong
answers that look fine.

These tests pin down the tristate classification:

* By default, every degradable option emits ``UNSUPPORTED`` at both
  the feature and option level, so ``backend="baybe"`` is rejected
  at create-time with a structured error.
* ``backend="auto"`` continues to route around BayBE for such specs
  (BoTorch wins the FULL tier).
* Listing the field name in
  :attr:`OptimizationSpec.acknowledge_degradations` downgrades the
  report to ``IGNORED`` so the caller opts into a degraded BayBE run
  with a prominent warning.

Genuinely unsupported cases (RGPE transfer learning, hybrid
constraints, malformed BayBE typed options) stay ``UNSUPPORTED``
regardless of acknowledgement.
"""

from __future__ import annotations

from typing import Any

import pytest
from bo_engine.backend_base import CapabilityStatus
from bo_engine.saasbo import SAASBOConfig
from bo_engine.types import (
    FidelityParameterSpec,
    ObjectiveSpec,
    OptimizationSpec,
    OutcomeConstraintSpec,
    ParameterSpec,
    ParameterType,
    TransferLearningSpec,
    TurboConfig,
)

from bo_engine_baybe.backend import BayBEBackend


def _scalar_spec(**spec_overrides: Any) -> OptimizationSpec:
    """Single-objective continuous spec, overridable per-test."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        **spec_overrides,
    )


_DEGRADABLE_CASES = [
    ({"use_input_warping": True}, "input_warping", "use_input_warping"),
    ({"use_cost_aware": True}, "cost_aware", "use_cost_aware"),
    ({"turbo_config": TurboConfig()}, "high_dimensional", "turbo_config"),
    (
        {"saasbo_config": SAASBOConfig(warmup_steps=8, num_samples=8, thinning=2)},
        "high_dimensional",
        "saasbo_config",
    ),
    (
        {
            "fidelity_parameter": FidelityParameterSpec(name="fid", bounds=(0.1, 1.0), target=1.0),
        },
        "multi_fidelity",
        "fidelity_parameter",
    ),
    (
        {"outcome_constraints": [OutcomeConstraintSpec(objective_name="y", threshold=0.5)]},
        "outcome_constraints",
        "outcome_constraints",
    ),
]


@pytest.mark.parametrize(("kwargs", "feature_key", "option_key"), _DEGRADABLE_CASES)
def test_baybe_rejects_degradable_knobs_by_default(
    kwargs: dict[str, Any],
    feature_key: str,
    option_key: str,
) -> None:
    """Each semantic knob is UNSUPPORTED by default — the spec is rejected."""
    spec = _scalar_spec(**kwargs)
    result = BayBEBackend().validate_capabilities(spec)

    assert not result.is_compatible, (
        f"Semantic knob {option_key!r} must reject backend='baybe' by default"
    )
    feature_statuses = {r.key: r.status for r in result.feature_reports}
    option_statuses = {r.key: r.status for r in result.option_reports}
    assert feature_statuses.get(feature_key) == CapabilityStatus.UNSUPPORTED
    assert option_statuses.get(option_key) == CapabilityStatus.UNSUPPORTED


@pytest.mark.parametrize(("kwargs", "feature_key", "option_key"), _DEGRADABLE_CASES)
def test_acknowledged_degradation_downgrades_to_ignored(
    kwargs: dict[str, Any],
    feature_key: str,
    option_key: str,
) -> None:
    """Naming the field in ``acknowledge_degradations`` downgrades UNSUPPORTED → IGNORED."""
    spec = _scalar_spec(acknowledge_degradations=(option_key,), **kwargs)
    result = BayBEBackend().validate_capabilities(spec)

    assert result.is_compatible
    feature_statuses = {r.key: r.status for r in result.feature_reports}
    option_statuses = {r.key: r.status for r in result.option_reports}
    assert feature_statuses.get(feature_key) == CapabilityStatus.IGNORED
    assert option_statuses.get(option_key) == CapabilityStatus.IGNORED


def test_high_dimensional_diagnostic_names_only_active_attrs() -> None:
    """``HIGH_DIMENSIONAL`` is activated by ``turbo_config`` AND ``saasbo_config``.

    A naive reverse-lookup names ``turbo_config`` first even when only
    ``saasbo_config`` is set, sending the caller to acknowledge the
    wrong field. The classifier must filter to attributes actually set
    on the spec.
    """
    spec = _scalar_spec(saasbo_config=SAASBOConfig(warmup_steps=8, num_samples=8, thinning=2))
    result = BayBEBackend().validate_capabilities(spec)
    high_dim = [r for r in result.feature_reports if r.key == "high_dimensional"]
    assert high_dim
    reason = high_dim[0].reason or ""
    assert "saasbo_config" in reason, reason
    assert "turbo_config" not in reason, reason


def test_high_dimensional_partial_acknowledgement_remains_incompatible() -> None:
    """Both TuRBO and SAASBO set, only one acknowledged → spec stays incompatible.

    The option-level report for the unacknowledged attribute keeps
    ``is_compatible == False`` even though the feature report could in
    principle be downgraded; the test pins that the feature report
    also stays ``UNSUPPORTED`` and names the unacknowledged attribute
    so the caller can fix the spec without guessing.
    """
    spec = _scalar_spec(
        turbo_config=TurboConfig(),
        saasbo_config=SAASBOConfig(warmup_steps=8, num_samples=8, thinning=2),
        acknowledge_degradations=("saasbo_config",),
    )
    result = BayBEBackend().validate_capabilities(spec)
    assert not result.is_compatible
    high_dim = [r for r in result.feature_reports if r.key == "high_dimensional"]
    assert high_dim
    assert high_dim[0].status == CapabilityStatus.UNSUPPORTED
    reason = high_dim[0].reason or ""
    assert "turbo_config" in reason, reason
    option_statuses = {r.key: r.status for r in result.option_reports}
    assert option_statuses.get("turbo_config") == CapabilityStatus.UNSUPPORTED
    assert option_statuses.get("saasbo_config") == CapabilityStatus.IGNORED


def test_rgpe_transfer_learning_still_unsupported_for_baybe() -> None:
    """RGPE transfer learning on BayBE remains UNSUPPORTED.

    BayBE supports transfer learning only via a ``TaskParameter``; the
    neutral ``transfer_learning`` slot targets BoTorch's RGPE backend.
    The IGNORED downgrade must not weaken this gate.
    """
    spec = _scalar_spec(transfer_learning=TransferLearningSpec(prior_campaign_ids=["abc"]))
    result = BayBEBackend().validate_capabilities(spec)

    assert not result.is_compatible
    tr_reports = [r for r in result.feature_reports if r.key == "transfer_learning"]
    assert tr_reports
    assert tr_reports[0].status == CapabilityStatus.UNSUPPORTED


def test_validate_spec_still_emits_legacy_warnings() -> None:
    """The legacy string-warning surface still lists degraded knobs.

    ``backend.validate_spec`` is used by callers that grep warning
    strings; the UNSUPPORTED classification at the structured layer
    must not drop the human-readable warning.
    """
    backend = BayBEBackend()
    spec = _scalar_spec(use_input_warping=True, use_cost_aware=True)
    warnings = backend.validate_spec(spec)
    assert any("Input warping" in w for w in warnings)
    assert any("Cost-aware" in w for w in warnings)
