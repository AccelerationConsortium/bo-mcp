"""TODO 1.64 / 1.65 — capability classification & typed option validation tests.

The BayBE backend's ``validate_capabilities`` must produce a per-constraint
classification (continuous / discrete / hybrid / categorical) so that
``backend_name="auto"`` does not pick BayBE for specs BayBE cannot
construct. Misshaped ``parameter_options["baybe"]`` /
``backend_options["baybe"]`` payloads also surface as
``CapabilityStatus.UNSUPPORTED`` reports here instead of crashing the
suggestion path.

Reference: TODO 1.64 (constraint capability), 1.65 (BayBE-native parameter
metadata typed schema).
"""

from __future__ import annotations

from bo_engine.backend_base import CapabilityStatus
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

from bo_engine_baybe.backend import BayBEBackend


def _make_spec(
    parameters: list[ParameterSpec],
    constraints: list[ConstraintSpec] | None = None,
    backend_options: dict | None = None,
) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=parameters,
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        constraints=constraints or [],
        backend_options=backend_options,
    )


class TestConstraintCapability:
    def test_continuous_only_sum_is_supported(self) -> None:
        spec = _make_spec(
            parameters=[
                ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            constraints=[
                ConstraintSpec(
                    type=ConstraintType.SUM_EQUALS,
                    parameters=["a", "b"],
                    value=1.0,
                ),
            ],
        )
        result = BayBEBackend().validate_capabilities(spec)
        assert result.is_compatible
        constraint_reports = [r for r in result.feature_reports if r.key.startswith("constraint[")]
        assert constraint_reports
        assert all(r.status == CapabilityStatus.SUPPORTED for r in constraint_reports)

    def test_hybrid_constraint_blocks_auto_selection(self) -> None:
        spec = _make_spec(
            parameters=[
                ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="b", type=ParameterType.DISCRETE, values=[0.0, 0.5, 1.0]),
            ],
            constraints=[
                ConstraintSpec(
                    type=ConstraintType.SUM_EQUALS,
                    parameters=["a", "b"],
                    value=1.0,
                ),
            ],
        )
        result = BayBEBackend().validate_capabilities(spec)
        assert not result.is_compatible
        unsupported = [
            r for r in result.feature_reports if r.status == CapabilityStatus.UNSUPPORTED
        ]
        assert any("mixed continuous/discrete" in r.reason for r in unsupported)

    def test_discrete_only_constraint_is_supported(self) -> None:
        spec = _make_spec(
            parameters=[
                ParameterSpec(
                    name="a", type=ParameterType.DISCRETE, values=[0.0, 0.25, 0.5, 0.75, 1.0]
                ),
                ParameterSpec(
                    name="b", type=ParameterType.DISCRETE, values=[0.0, 0.25, 0.5, 0.75, 1.0]
                ),
            ],
            constraints=[
                ConstraintSpec(
                    type=ConstraintType.SUM_EQUALS,
                    parameters=["a", "b"],
                    value=1.0,
                ),
            ],
        )
        result = BayBEBackend().validate_capabilities(spec)
        constraint_reports = [r for r in result.feature_reports if r.key.startswith("constraint[")]
        assert all(r.status == CapabilityStatus.SUPPORTED for r in constraint_reports)


class TestTransferLearningCapability:
    def test_task_parameter_makes_transfer_supported(self) -> None:
        spec = _make_spec(
            parameters=[
                ParameterSpec(
                    name="lab",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B"],
                    parameter_options={"baybe": {"role": "task", "active_values": ["A"]}},
                ),
            ],
        )
        result = BayBEBackend().validate_capabilities(spec)
        tr_reports = [r for r in result.feature_reports if r.key == "transfer_learning"]
        assert tr_reports
        assert tr_reports[0].status == CapabilityStatus.SUPPORTED

    def test_rgpe_style_transfer_unsupported(self) -> None:
        """Neutral ``transfer_learning`` config targets BoTorch RGPE, not BayBE."""
        from bo_engine.types import TransferLearningSpec

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            transfer_learning=TransferLearningSpec(prior_campaign_ids=["abc"]),
        )
        result = BayBEBackend().validate_capabilities(spec)
        tr_reports = [r for r in result.feature_reports if r.key == "transfer_learning"]
        assert tr_reports
        assert tr_reports[0].status == CapabilityStatus.UNSUPPORTED
        assert "TaskParameter" in tr_reports[0].reason


class TestTypedOptionValidation:
    def test_invalid_parameter_options_reported(self) -> None:
        """Misspelled BayBE parameter option keys produce an UNSUPPORTED report."""
        spec = _make_spec(
            parameters=[
                ParameterSpec(
                    name="solvent",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B"],
                    parameter_options={"baybe": {"role": "task", "bogus_key": 42}},
                ),
            ],
        )
        result = BayBEBackend().validate_capabilities(spec)
        bad = [r for r in result.option_reports if "parameter_options" in r.key]
        assert bad
        assert bad[0].status == CapabilityStatus.UNSUPPORTED

    def test_invalid_backend_options_reported(self) -> None:
        """Misspelled BayBE backend option keys produce an UNSUPPORTED report."""
        spec = _make_spec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            backend_options={"baybe": {"not_a_real_key": True}},
        )
        result = BayBEBackend().validate_capabilities(spec)
        bad = [r for r in result.option_reports if r.key == "backend_options.baybe"]
        assert bad
        assert bad[0].status == CapabilityStatus.UNSUPPORTED

    def test_substance_role_requires_data(self) -> None:
        """A SubstanceParameter role without SMILES data is reported as UNSUPPORTED."""
        spec = _make_spec(
            parameters=[
                ParameterSpec(
                    name="solvent",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B"],
                    parameter_options={"baybe": {"role": "substance"}},
                ),
            ],
        )
        result = BayBEBackend().validate_capabilities(spec)
        bad = [r for r in result.option_reports if "substance_data" in r.key]
        assert bad
        assert bad[0].status == CapabilityStatus.UNSUPPORTED

    def test_task_role_on_continuous_parameter_reported(self) -> None:
        """``role=task`` on a non-categorical parameter is UNSUPPORTED."""
        spec = _make_spec(
            parameters=[
                ParameterSpec(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),
                    parameter_options={"baybe": {"role": "task"}},
                ),
            ],
        )
        result = BayBEBackend().validate_capabilities(spec)
        bad = [r for r in result.option_reports if r.key.startswith("parameter_options[x]")]
        assert bad
        assert bad[0].status == CapabilityStatus.UNSUPPORTED
        assert "categorical base" in bad[0].reason
