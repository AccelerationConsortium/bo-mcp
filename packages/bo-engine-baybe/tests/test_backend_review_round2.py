"""Round-2 regression tests for BayBE-backend review findings.

Each test maps to one finding from the second round of review:

1. Constraints with a mix of known and unknown parameter names must be
   classified as ``unknown``/``UNSUPPORTED`` instead of being silently
   downgraded based on the known names alone.
2. BayBE substance parameters cannot be constructed without the
   ``baybe[chem]`` extras; capabilities must report this clearly when
   chem isn't installed.
3. The explicit-backend creation path must enforce typed-option
   validation, not just the auto-selection path.
"""

from __future__ import annotations

import pytest
from bo_engine.backend_base import CapabilityStatus
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

from bo_engine_baybe.backend import (
    _CHEMISTRY_AVAILABLE,
    BayBEBackend,
)


class TestFinding1MixedKnownUnknownConstraintParams:
    """Constraints over a mix of declared and undeclared parameter names.

    Pre-fix: ``classify_constraint_target`` only returned ``unknown``
    when every name was missing. With one known continuous param and
    one undeclared name, it returned ``continuous`` and the converter
    handed the constraint to BayBE, which then crashed inside
    ``SearchSpace.from_product``.
    """

    def _spec_with_mixed_names(self) -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            constraints=[
                ConstraintSpec(
                    type=ConstraintType.SUM_EQUALS,
                    parameters=["a", "missing"],
                    value=1.0,
                ),
            ],
        )

    def test_validate_capabilities_reports_unknown(self) -> None:
        result = BayBEBackend().validate_capabilities(self._spec_with_mixed_names())
        assert not result.is_compatible
        bad = [
            r
            for r in result.feature_reports
            if r.key.startswith("constraint[") and r.status == CapabilityStatus.UNSUPPORTED
        ]
        assert bad
        assert "['missing']" in bad[0].reason

    def test_converter_rejects_mixed_names(self) -> None:
        from bo_engine_baybe.converters import spec_to_constraints

        spec = self._spec_with_mixed_names()
        with pytest.raises(ValueError, match="unknown parameters"):
            spec_to_constraints(spec.constraints, spec.parameters)


class TestFinding2SubstanceWithoutChemExtras:
    """SubstanceParameter requires ``baybe[chem]`` to actually construct.

    When chem extras are missing, the capability report flags the role as
    ``UNSUPPORTED`` so the spec is rejected at intake rather than
    crashing inside the BayBE constructor at suggestion time.
    """

    @pytest.mark.skipif(
        _CHEMISTRY_AVAILABLE,
        reason="baybe[chem] is installed; this regression check only applies when it isn't",
    )
    def test_substance_role_reports_unavailable(self) -> None:
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="solvent",
                    type=ParameterType.CATEGORICAL,
                    categories=["water", "ethanol"],
                    parameter_options={
                        "baybe": {
                            "role": "substance",
                            "substance_data": {"water": "O", "ethanol": "CCO"},
                        }
                    },
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        result = BayBEBackend().validate_capabilities(spec)
        assert not result.is_compatible
        bad = [
            r
            for r in result.option_reports
            if r.status == CapabilityStatus.UNSUPPORTED and "baybe[chem]" in (r.reason or "")
        ]
        assert bad


class TestFinding3ExplicitBackendValidation:
    """``backend="baybe"`` must enforce ``validate_capabilities`` at creation.

    Pre-fix: only ``validate_spec`` ran for explicit backends, so the new
    typed-option ``UNSUPPORTED`` reports got bypassed. A campaign with
    invalid BayBE ``parameter_options`` was accepted and only failed
    during the first suggestion call. This package-level test confirms
    the BayBE side reports the spec as incompatible; the end-to-end
    ``create_campaign_operation`` rejection lives in the server tests
    (the operation depends on the DB session machinery).
    """

    def test_invalid_active_values_makes_spec_incompatible(self) -> None:
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="lab",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B"],
                    parameter_options={
                        "baybe": {"role": "task", "active_values": ["A", "Z"]},
                    },
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        result = BayBEBackend().validate_capabilities(spec)
        assert not result.is_compatible
        assert any(
            r.status == CapabilityStatus.UNSUPPORTED and "active_values" in r.key
            for r in result.option_reports
        )
