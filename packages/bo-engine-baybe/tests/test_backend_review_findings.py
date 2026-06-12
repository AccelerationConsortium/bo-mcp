"""Regression tests for the BayBE-backend review findings.

Each test maps directly to a finding from the post-implementation
code review:

1. Empty-observation reconciliation must still trigger rebuild when the
   restored campaign has measurements.
2. Duplicate identical observations are real replicates and must both
   be added to BayBE (multiset-style reconciliation).
3. Discrete ``LINEAR`` constraints have no native BayBE equivalent and
   must be reported as ``UNSUPPORTED`` instead of silently mis-mapping
   to ``ContinuousLinearConstraint``.
4. ``BayBEParameterOptions.active_values`` outside the declared
   categories and ``substance_data`` missing categories must surface as
   :class:`CapabilityStatus.UNSUPPORTED` reports at intake.
5. Random-warmup provenance must not claim a GP / qLogNEI acquisition.
"""

from __future__ import annotations

from baybe import Campaign

from bo_engine.backend_base import CapabilityStatus
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.backend import BayBEBackend


def _basic_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


def _restored_measurement_count(state: dict | None) -> int:
    """Round-trip the BayBE campaign JSON and count its measurements."""
    assert state is not None
    return len(Campaign.from_json(state["payload"]["campaign_json"]).measurements)


class TestFinding1EmptyObservationsRebuild:
    """If storage returns no observations, the restored campaign must be rebuilt.

    The pre-fix code only ran ``_reconcile_measurements`` inside the
    ``if observations:`` branch, so a state carrying measurements would
    be serialized again unchanged when storage was emptied. The new
    code always reconciles and rebuilds from the empty observation set.
    """

    def test_empty_observations_drops_stale_measurements(self) -> None:
        backend = BayBEBackend()
        spec = _basic_spec()
        obs = [
            ObservationData(parameter_values={"x1": 0.3, "x2": 0.7}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.6, "x2": 0.2}, objective_values={"y": 0.5}),
        ]
        batch1 = backend.generate_suggestions(spec, obs, batch_size=1, iteration=1)
        assert _restored_measurement_count(batch1.backend_state) == 2

        # Storage now returns []. The new state must not still report 2.
        batch2 = backend.generate_suggestions(
            spec,
            [],
            batch_size=1,
            iteration=2,
            backend_state=batch1.backend_state,
        )
        assert _restored_measurement_count(batch2.backend_state) == 0
        assert batch2.backend_state is not None
        assert batch2.backend_state["payload"].get("observation_identity") == []


class TestFinding2DuplicateObservationsArePreserved:
    """Two observations with identical values are real replicates.

    The pre-fix reconciliation built a ``set[str]`` of fingerprints,
    so the second replicate was silently dropped from the BayBE
    campaign. The new multiset reconciliation keeps both rows.
    """

    def test_replicates_both_added(self) -> None:
        backend = BayBEBackend()
        spec = _basic_spec()
        replicate = ObservationData(
            parameter_values={"x1": 0.4, "x2": 0.6}, objective_values={"y": 0.7}
        )
        obs = [replicate, replicate]
        batch = backend.generate_suggestions(spec, obs, batch_size=1, iteration=1)
        assert _restored_measurement_count(batch.backend_state) == 2

    def test_replicate_added_in_second_call(self) -> None:
        backend = BayBEBackend()
        spec = _basic_spec()
        original = ObservationData(
            parameter_values={"x1": 0.4, "x2": 0.6}, objective_values={"y": 0.7}
        )
        first = backend.generate_suggestions(spec, [original], batch_size=1, iteration=1)

        # Second call appends an identical replicate; both must remain.
        batch = backend.generate_suggestions(
            spec,
            [original, original],
            batch_size=1,
            iteration=2,
            backend_state=first.backend_state,
        )
        assert _restored_measurement_count(batch.backend_state) == 2


class TestFinding3DiscreteLinearUnsupported:
    """Discrete ``LINEAR`` constraints have no native BayBE equivalent."""

    def _discrete_linear_spec(self) -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="a", type=ParameterType.DISCRETE, values=[0.0, 0.5, 1.0]),
                ParameterSpec(name="b", type=ParameterType.DISCRETE, values=[0.0, 0.5, 1.0]),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            constraints=[
                ConstraintSpec(
                    type=ConstraintType.LINEAR,
                    parameters=["a", "b"],
                    value=1.0,
                    coefficients=[2.0, 1.0],
                ),
            ],
        )

    def test_validate_capabilities_marks_discrete_linear_unsupported(self) -> None:
        result = BayBEBackend().validate_capabilities(self._discrete_linear_spec())
        assert not result.is_compatible
        bad = [
            r
            for r in result.feature_reports
            if r.key.startswith("constraint[") and r.status == CapabilityStatus.UNSUPPORTED
        ]
        assert bad
        assert "discrete equivalent" in bad[0].reason

    def test_converter_rejects_discrete_linear(self) -> None:
        from bo_engine_baybe.converters import spec_to_constraints

        spec = self._discrete_linear_spec()
        import pytest

        with pytest.raises(ValueError, match="discrete equivalent"):
            spec_to_constraints(spec.constraints, spec.parameters)


class TestFinding4TaskAndSubstanceSemanticValidation:
    """Typed-option validation must catch BayBE semantic invariants."""

    def test_task_active_values_must_be_subset_of_categories(self) -> None:
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="lab",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B", "C"],
                    parameter_options={
                        "baybe": {"role": "task", "active_values": ["A", "Z"]},
                    },
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        result = BayBEBackend().validate_capabilities(spec)
        bad = [
            r
            for r in result.option_reports
            if "active_values" in r.key and r.status == CapabilityStatus.UNSUPPORTED
        ]
        assert bad
        assert "['Z']" in bad[0].reason

    def test_substance_data_must_cover_categories(self) -> None:
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="solvent",
                    type=ParameterType.CATEGORICAL,
                    categories=["water", "ethanol", "dmf"],
                    parameter_options={
                        "baybe": {
                            "role": "substance",
                            "substance_data": {"water": "O"},  # missing ethanol & dmf
                        },
                    },
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        result = BayBEBackend().validate_capabilities(spec)
        bad = [
            r
            for r in result.option_reports
            if "substance_data" in r.key and r.status == CapabilityStatus.UNSUPPORTED
        ]
        assert bad
        assert "missing SMILES" in bad[0].reason


class TestFinding5RandomPhaseProvenance:
    """Random / nonpredictive phases must not advertise a GP."""

    def test_random_warmup_reports_no_surrogate(self) -> None:
        backend = BayBEBackend()
        batch = backend.generate_suggestions(
            spec=_basic_spec(),
            observations=[],
            batch_size=1,
            iteration=1,
        )
        assert batch.method_info["is_nonpredictive"] is True
        assert batch.method_info["model_type"] == "none (space-filling)"
        assert batch.method_info["acquisition_function"] == "none (space-filling)"

    def test_post_switch_phase_reports_gp(self) -> None:
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x1": 0.1, "x2": 0.9}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.5}),
            ObservationData(parameter_values={"x1": 0.8, "x2": 0.2}, objective_values={"y": 0.3}),
        ]
        batch = backend.generate_suggestions(
            spec=_basic_spec(),
            observations=observations,
            batch_size=1,
            iteration=1,
        )
        assert batch.method_info["is_nonpredictive"] is False
        assert batch.method_info["model_type"] == "BayBE GP"
