"""Capability classification & typed option validation tests.

The BayBE backend's ``validate_capabilities`` must produce a per-constraint
classification (continuous / discrete / hybrid / categorical) so that
``backend_name="auto"`` does not pick BayBE for specs BayBE cannot
construct. Misshaped ``parameter_options["baybe"]`` /
``backend_options["baybe"]`` payloads also surface as
``CapabilityStatus.UNSUPPORTED`` reports here instead of crashing the
suggestion path.
"""

from __future__ import annotations

from bo_engine.backend_base import CapabilityStatus
from bo_engine.types import (
    AcquisitionMethod,
    AcquisitionOptimizationConfig,
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    TargetMode,
)
from bo_engine_baybe.backend import BayBEBackend


def _make_spec(
    parameters: list[ParameterSpec],
    constraints: list[ConstraintSpec] | None = None,
    backend_options: dict | None = None,
    objectives: list[ObjectiveSpec] | None = None,
    **spec_kwargs: object,
) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=parameters,
        objectives=objectives or [ObjectiveSpec(name="y", minimize=True)],
        constraints=constraints or [],
        backend_options=backend_options,
        **spec_kwargs,  # ty: ignore[invalid-argument-type]
    )


def _continuous_x() -> list[ParameterSpec]:
    return [ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))]


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


class TestLogTransformCapability:
    """Per-objective ``log_transform`` reports DEGRADED, not silence.

    BayBE honors the flag via a logarithmic target transformation, but
    only at the acquisition-objective level — the GP surrogate still fits
    the raw target scale (verified against baybe 0.14 ``Surrogate.fit``,
    which pipes raw target values through ``Objective._pre_transform``).
    DEGRADED keeps the spec compatible while surfacing the semantic gap.
    """

    def test_log_transform_reports_degraded(self) -> None:
        spec = _make_spec(
            parameters=_continuous_x(),
            objectives=[ObjectiveSpec(name="rate", minimize=True, log_transform=True)],
        )
        result = BayBEBackend().validate_capabilities(spec)
        reports = [r for r in result.option_reports if r.key == "objectives[0].log_transform"]
        assert reports
        assert reports[0].status == CapabilityStatus.DEGRADED
        assert "raw target scale" in reports[0].reason
        # DEGRADED must not block the spec — no acknowledgement gate.
        assert result.is_compatible

    def test_log_transform_appears_in_validate_spec_warnings(self) -> None:
        spec = _make_spec(
            parameters=_continuous_x(),
            objectives=[ObjectiveSpec(name="rate", minimize=True, log_transform=True)],
        )
        warnings = BayBEBackend().validate_spec(spec)
        assert any("log_transform" in w and "rate" in w for w in warnings)

    def test_log_transform_match_mode_warning_reuses_capability_reason(self) -> None:
        """The legacy warning surface must not imply log x match works.

        The combination is UNSUPPORTED at the capability layer (a match
        target has no slot for a chained log transformation), so the
        string surface must repeat that rejection instead of the
        applied-at-acquisition-level phrasing that describes the
        supported minimize case.
        """
        spec = _make_spec(
            parameters=_continuous_x(),
            objectives=[
                ObjectiveSpec(
                    name="rate",
                    log_transform=True,
                    target_mode=TargetMode.MATCH,
                    target_value=5.0,
                )
            ],
        )
        warnings = BayBEBackend().validate_spec(spec)
        assert any("cannot be combined with target_mode='match'" in w for w in warnings)
        assert not any("acquisition-objective level" in w for w in warnings)

    def test_per_objective_keys_for_multi_objective_spec(self) -> None:
        """Only the flagged objective gets a report, keyed by its index."""
        spec = _make_spec(
            parameters=_continuous_x(),
            objectives=[
                ObjectiveSpec(name="yield", minimize=True),
                ObjectiveSpec(name="impurity", minimize=True, log_transform=True),
            ],
        )
        result = BayBEBackend().validate_capabilities(spec)
        keys = [r.key for r in result.option_reports if "log_transform" in r.key]
        assert keys == ["objectives[1].log_transform"]

    def test_plain_objective_emits_no_report(self) -> None:
        result = BayBEBackend().validate_capabilities(_make_spec(parameters=_continuous_x()))
        assert not any("log_transform" in r.key for r in result.option_reports)

    def test_log_transform_with_maximize_reports_unsupported(self) -> None:
        """Capability validation must reject what construction would reject.

        ``_build_baybe_target`` raises for ``log_transform=True`` +
        ``minimize=False`` (the neutral contract restricts the flag to
        minimize objectives), so ``validate_capabilities`` must report
        UNSUPPORTED — a compatible verdict here would let a pinned
        ``backend="baybe"`` campaign pass intake and then fail at
        suggestion generation.
        """
        spec = _make_spec(
            parameters=_continuous_x(),
            objectives=[ObjectiveSpec(name="rate", minimize=False, log_transform=True)],
        )
        result = BayBEBackend().validate_capabilities(spec)
        reports = [r for r in result.option_reports if r.key == "objectives[0].log_transform"]
        assert reports
        assert reports[0].status == CapabilityStatus.UNSUPPORTED
        assert "minimize=True" in reports[0].reason
        assert not result.is_compatible


class TestAcquisitionMethodCapability:
    """Unmappable acquisition methods follow the degradable-knob policy."""

    def test_cost_weighted_ei_unsupported_by_default(self) -> None:
        spec = _make_spec(
            parameters=_continuous_x(),
            acquisition_method=AcquisitionMethod.COST_WEIGHTED_EI,
        )
        result = BayBEBackend().validate_capabilities(spec)
        reports = [r for r in result.option_reports if r.key == "acquisition_method"]
        assert reports
        assert reports[0].status == CapabilityStatus.UNSUPPORTED
        assert not result.is_compatible

    def test_acknowledged_cost_weighted_ei_downgrades_to_ignored(self) -> None:
        spec = _make_spec(
            parameters=_continuous_x(),
            acquisition_method=AcquisitionMethod.COST_WEIGHTED_EI,
            acknowledge_degradations=("acquisition_method",),
        )
        result = BayBEBackend().validate_capabilities(spec)
        reports = [r for r in result.option_reports if r.key == "acquisition_method"]
        assert reports
        assert reports[0].status == CapabilityStatus.IGNORED
        assert result.is_compatible

    def test_multi_fidelity_kg_unsupported_by_default(self) -> None:
        spec = _make_spec(
            parameters=_continuous_x(),
            acquisition_method=AcquisitionMethod.MULTI_FIDELITY_KG,
        )
        result = BayBEBackend().validate_capabilities(spec)
        reports = [r for r in result.option_reports if r.key == "acquisition_method"]
        assert reports
        assert reports[0].status == CapabilityStatus.UNSUPPORTED

    def test_mappable_methods_emit_no_report(self) -> None:
        """Honored methods (wired into BotorchRecommender) need no warning."""
        for method in (
            AcquisitionMethod.AUTO,
            AcquisitionMethod.EXPECTED_IMPROVEMENT,
            AcquisitionMethod.NOISY_EI,
        ):
            spec = _make_spec(parameters=_continuous_x(), acquisition_method=method)
            result = BayBEBackend().validate_capabilities(spec)
            assert not any(r.key == "acquisition_method" for r in result.option_reports), method

    def test_unmappable_method_appears_in_validate_spec_warnings(self) -> None:
        spec = _make_spec(
            parameters=_continuous_x(),
            acquisition_method=AcquisitionMethod.COST_WEIGHTED_EI,
        )
        warnings = BayBEBackend().validate_spec(spec)
        assert any("cost_weighted_ei" in w for w in warnings)


class TestAcquisitionOptimizationCapability:
    """Explicit L-BFGS-B budget overrides are IGNORED (not UNSUPPORTED) on BayBE.

    ``acquisition_optimization.num_restarts`` / ``raw_samples`` tune the
    BoTorch acquisition optimizer; BayBE optimizes acquisition internally
    and cannot honor them. The report is ``IGNORED`` so an explicit
    ``backend="baybe"`` still runs, while ``backend="auto"`` demotes BayBE
    below a backend that honors the override (the tier logic counts
    IGNORED like DEGRADED).
    """

    def test_explicit_override_reports_ignored(self) -> None:
        spec = _make_spec(
            parameters=_continuous_x(),
            acquisition_optimization=AcquisitionOptimizationConfig(num_restarts=7, raw_samples=33),
        )
        result = BayBEBackend().validate_capabilities(spec)
        reports = [r for r in result.option_reports if r.key == "acquisition_optimization"]
        assert reports
        assert reports[0].status == CapabilityStatus.IGNORED
        assert result.is_compatible

    def test_partial_override_reports_ignored(self) -> None:
        """A single overridden knob is enough to trigger the report."""
        spec = _make_spec(
            parameters=_continuous_x(),
            acquisition_optimization=AcquisitionOptimizationConfig(num_restarts=7),
        )
        result = BayBEBackend().validate_capabilities(spec)
        assert any(r.key == "acquisition_optimization" for r in result.option_reports)

    def test_default_config_emits_no_report(self) -> None:
        """The field is always present (default_factory); only overrides count."""
        spec = _make_spec(parameters=_continuous_x())
        result = BayBEBackend().validate_capabilities(spec)
        assert not any(r.key == "acquisition_optimization" for r in result.option_reports)


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
