"""Round-3 regression tests: BayBE-ignored knobs must not block campaign creation.

Pre-fix, ``required_features`` translated active option fields like
``use_input_warping``, ``use_cost_aware``, ``outcome_constraints``,
``turbo_config``, ``saasbo_config``, ``fidelity_parameter`` into
abstract :class:`Feature` values. BayBE then reported these as
``UNSUPPORTED`` *feature* reports while simultaneously reporting them
as ``IGNORED`` *option* reports — a contradiction that, combined with
the new create-time capability enforcement, rejected
``backend="baybe"`` specs that used to be accepted with a warning.

These tests pin down that BayBE-ignored knobs:

* keep the per-option IGNORED warning so users still see "BayBE will
  not honor X";
* downgrade the corresponding feature report from ``UNSUPPORTED`` to
  ``IGNORED`` so :attr:`BackendValidationResult.is_compatible` is
  ``True`` and ``create_campaign_operation`` does **not** reject the
  spec.

Genuinely unsupported cases (RGPE transfer learning, hybrid
constraints, malformed BayBE typed options) still produce
``UNSUPPORTED`` reports and continue to reject campaigns.
"""

from __future__ import annotations

from typing import Any

import pytest
from bo_engine.backend_base import CapabilityStatus
from bo_engine.types import (
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


@pytest.mark.parametrize(
    ("kwargs", "feature_key", "option_key"),
    [
        ({"use_input_warping": True}, "input_warping", "use_input_warping"),
        ({"use_cost_aware": True}, "cost_aware", "use_cost_aware"),
        ({"turbo_config": TurboConfig()}, "high_dimensional", "turbo_config"),
        (
            {"outcome_constraints": [OutcomeConstraintSpec(objective_name="y", threshold=0.5)]},
            "outcome_constraints",
            "outcome_constraints",
        ),
    ],
)
def test_baybe_ignored_knobs_do_not_block_compatibility(
    kwargs: dict[str, Any],
    feature_key: str,
    option_key: str,
) -> None:
    """Each BayBE-ignored knob keeps the spec compatible (campaign accepted)."""
    spec = _scalar_spec(**kwargs)
    result = BayBEBackend().validate_capabilities(spec)

    assert result.is_compatible, (
        f"BayBE-ignored knob {option_key!r} should not block campaign creation"
    )

    feature_statuses = {r.key: r.status for r in result.feature_reports}
    option_statuses = {r.key: r.status for r in result.option_reports}

    # The feature-level report must be IGNORED, not UNSUPPORTED.
    assert feature_statuses.get(feature_key) == CapabilityStatus.IGNORED
    # The option-level IGNORED report still surfaces the warning to the user.
    assert option_statuses.get(option_key) == CapabilityStatus.IGNORED


def test_rgpe_transfer_learning_still_unsupported_for_baybe() -> None:
    """RGPE transfer learning on BayBE remains UNSUPPORTED.

    BayBE supports transfer learning only via a ``TaskParameter``; the
    neutral ``transfer_learning`` slot targets BoTorch's RGPE backend.
    The friend's review explicitly allowed this to remain rejected, so
    the round-3 IGNORED-downgrade must not weaken this gate.
    """
    spec = _scalar_spec(transfer_learning=TransferLearningSpec(prior_campaign_ids=["abc"]))
    result = BayBEBackend().validate_capabilities(spec)

    assert not result.is_compatible
    tr_reports = [r for r in result.feature_reports if r.key == "transfer_learning"]
    assert tr_reports
    assert tr_reports[0].status == CapabilityStatus.UNSUPPORTED


def test_validate_spec_still_emits_legacy_warnings() -> None:
    """The legacy string-warning surface still lists ignored knobs.

    ``backend.validate_spec`` is used by callers that grep warning
    strings; the IGNORED downgrade must not silently drop those.
    """
    backend = BayBEBackend()
    spec = _scalar_spec(use_input_warping=True, use_cost_aware=True)
    warnings = backend.validate_spec(spec)
    assert any("Input warping" in w for w in warnings)
    assert any("Cost-aware" in w for w in warnings)
