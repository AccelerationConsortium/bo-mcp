"""Tests for outcome constraint modeling."""

import numpy as np
import pytest

from bo_engine import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    OutcomeConstraintConfigurationError,
    OutcomeConstraintSpec,
    ParameterSpec,
    ParameterType,
    generate_next_batch,
)


def make_spec_with_constraint(
    threshold: float = 0.8,
    greater_than: bool = True,
) -> OptimizationSpec:
    """Create optimization spec with an outcome constraint."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="x1",
                type=ParameterType.CONTINUOUS,
                bounds=(0.0, 1.0),
            ),
            ParameterSpec(
                name="x2",
                type=ParameterType.CONTINUOUS,
                bounds=(0.0, 1.0),
            ),
        ],
        objectives=[ObjectiveSpec(name="yield", minimize=False)],
        batch_size=2,
        outcome_constraints=[
            OutcomeConstraintSpec(
                objective_name="yield",
                threshold=threshold,
                greater_than=greater_than,
            )
        ],
    )


def generate_observations_with_feasibility() -> list[ObservationData]:
    """Generate observations with mixed feasibility."""
    return [
        # Feasible (yield >= 0.8)
        ObservationData(
            parameter_values={"x1": 0.2, "x2": 0.3},
            objective_values={"yield": 0.85},
        ),
        ObservationData(
            parameter_values={"x1": 0.4, "x2": 0.5},
            objective_values={"yield": 0.92},
        ),
        ObservationData(
            parameter_values={"x1": 0.6, "x2": 0.7},
            objective_values={"yield": 0.88},
        ),
        # Infeasible (yield < 0.8)
        ObservationData(
            parameter_values={"x1": 0.1, "x2": 0.1},
            objective_values={"yield": 0.45},
        ),
        ObservationData(
            parameter_values={"x1": 0.9, "x2": 0.9},
            objective_values={"yield": 0.55},
        ),
        ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.2},
            objective_values={"yield": 0.72},
        ),
    ]


class TestOutcomeConstraintSpec:
    """Test OutcomeConstraintSpec dataclass."""

    def test_greater_than_constraint(self) -> None:
        """OutcomeConstraintSpec with greater_than=True."""
        oc = OutcomeConstraintSpec(
            objective_name="yield",
            threshold=0.8,
            greater_than=True,
        )
        assert oc.objective_name == "yield"
        assert oc.threshold == 0.8
        assert oc.greater_than is True

    def test_less_than_constraint(self) -> None:
        """OutcomeConstraintSpec with greater_than=False."""
        oc = OutcomeConstraintSpec(
            objective_name="cost",
            threshold=100.0,
            greater_than=False,
        )
        assert oc.objective_name == "cost"
        assert oc.threshold == 100.0
        assert oc.greater_than is False


class TestOutcomeConstraintIntegration:
    """Test outcome constraint integration with suggestions."""

    def test_generates_suggestions_with_constraint(self, rng: np.random.Generator) -> None:
        """generate_next_batch works with outcome constraints."""
        spec = make_spec_with_constraint(threshold=0.8)
        observations = generate_observations_with_feasibility()

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2
        for sugg in suggestions:
            assert "x1" in sugg.parameter_values
            assert "x2" in sugg.parameter_values
            # Values should be within bounds
            assert 0.0 <= sugg.parameter_values["x1"] <= 1.0
            assert 0.0 <= sugg.parameter_values["x2"] <= 1.0

    def test_multiple_outcome_constraints(self, rng: np.random.Generator) -> None:
        """Works with multiple outcome constraints."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="yield", minimize=False),
            ],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="yield",
                    threshold=0.7,
                    greater_than=True,
                ),
            ],
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.4},
                objective_values={"yield": 0.85},
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.6},
                objective_values={"yield": 0.75},
            ),
            ObservationData(
                parameter_values={"x1": 0.2, "x2": 0.2},
                objective_values={"yield": 0.55},
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_less_than_constraint(self, rng: np.random.Generator) -> None:
        """Works with less-than constraints (e.g., cost <= 100)."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="value", minimize=False)],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="value",
                    threshold=50.0,
                    greater_than=False,  # value <= 50
                )
            ],
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.1},
                objective_values={"value": 30.0},  # Feasible
            ),
            ObservationData(
                parameter_values={"x1": 0.5},
                objective_values={"value": 45.0},  # Feasible
            ),
            ObservationData(
                parameter_values={"x1": 0.9},
                objective_values={"value": 80.0},  # Infeasible
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_no_outcome_constraints_works(self, rng: np.random.Generator) -> None:
        """generate_next_batch works when outcome_constraints is empty."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
            batch_size=2,
            outcome_constraints=[],  # Empty
        )

        observations = [
            ObservationData(parameter_values={"x1": 0.3}, objective_values={"f": 0.5}),
            ObservationData(parameter_values={"x1": 0.7}, objective_values={"f": 0.3}),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2


class TestOutcomeConstraintEngineGuard:
    """Engine-path guard against silent outcome-constraint disabling.

    Pre-fix, the engine returned ``None`` from
    ``_build_outcome_constraint_models`` whenever observations lacked
    the constrained objective. That converted a typo / data-loss bug
    into a silent semantic change. The guard is now an
    :class:`OutcomeConstraintConfigurationError` so regressions surface
    loudly instead of producing wrong-but-plausible suggestions.

    Intake validation rejects an undeclared ``objective_name`` before
    the spec ever reaches the engine, so this test exercises a direct
    engine caller (third-party agent, legacy test, mutated spec) that
    bypasses the intake validator.
    """

    def test_engine_raises_when_objective_not_declared(self, rng: np.random.Generator) -> None:
        """Constraint that references an undeclared objective raises in the engine."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="yield", minimize=False)],
            batch_size=1,
            outcome_constraints=[
                OutcomeConstraintSpec(objective_name="ield", threshold=0.5),  # typo
            ],
        )
        observations = [
            ObservationData(parameter_values={"x1": 0.3}, objective_values={"yield": 0.85}),
            ObservationData(parameter_values={"x1": 0.6}, objective_values={"yield": 0.7}),
            ObservationData(parameter_values={"x1": 0.9}, objective_values={"yield": 0.6}),
        ]
        with pytest.raises(OutcomeConstraintConfigurationError):
            generate_next_batch(
                spec=spec,
                observations=observations,
                batch_size=1,
                iteration=1,
                rng=rng,
            )


class TestOutcomeConstraintEdgeCases:
    """Test edge cases for outcome constraints."""

    def test_all_infeasible_observations(self, rng: np.random.Generator) -> None:
        """Handles case when all observations are infeasible."""
        spec = make_spec_with_constraint(threshold=0.99)  # Very high threshold

        observations = [
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.4}, objective_values={"yield": 0.5}
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.6}, objective_values={"yield": 0.6}
            ),
            ObservationData(
                parameter_values={"x1": 0.7, "x2": 0.8}, objective_values={"yield": 0.7}
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        # Should still generate suggestions (exploring to find feasible region)
        assert len(suggestions) == 2

    def test_all_feasible_observations(self, rng: np.random.Generator) -> None:
        """Handles case when all observations are feasible."""
        spec = make_spec_with_constraint(threshold=0.1)  # Very low threshold

        observations = [
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.4}, objective_values={"yield": 0.5}
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.6}, objective_values={"yield": 0.6}
            ),
            ObservationData(
                parameter_values={"x1": 0.7, "x2": 0.8}, objective_values={"yield": 0.7}
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2


class TestMultipleOutcomeConstraints:
    """Test multiple outcome constraints working together.

    Reference: Constrained Bayesian Optimization literature (e.g., Gardner et al., 2014)
    demonstrates that multiple constraints can be handled via probability of feasibility.
    """

    def test_two_constraints_same_objective(self, rng: np.random.Generator) -> None:
        """Test two constraints on the same objective (e.g., 0.3 <= yield <= 0.8).

        This creates a feasible band where both constraints must be satisfied.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="yield", minimize=False)],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="yield",
                    threshold=0.3,
                    greater_than=True,  # yield >= 0.3
                ),
                OutcomeConstraintSpec(
                    objective_name="yield",
                    threshold=0.8,
                    greater_than=False,  # yield <= 0.8
                ),
            ],
        )

        observations = [
            # Feasible (0.3 <= 0.5 <= 0.8)
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.4},
                objective_values={"yield": 0.5},
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.6},
                objective_values={"yield": 0.6},
            ),
            # Infeasible (0.2 < 0.3)
            ObservationData(
                parameter_values={"x1": 0.1, "x2": 0.1},
                objective_values={"yield": 0.2},
            ),
            # Infeasible (0.9 > 0.8)
            ObservationData(
                parameter_values={"x1": 0.9, "x2": 0.9},
                objective_values={"yield": 0.9},
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_constraints_on_different_objectives(self, rng: np.random.Generator) -> None:
        """Test constraints on different objectives in multi-objective optimization.

        This is the common case: optimize objectives while respecting constraints on each.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="yield", minimize=False),
                ObjectiveSpec(name="purity", minimize=False),
            ],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="yield",
                    threshold=0.5,
                    greater_than=True,  # yield >= 0.5
                ),
                OutcomeConstraintSpec(
                    objective_name="purity",
                    threshold=0.6,
                    greater_than=True,  # purity >= 0.6
                ),
            ],
        )

        observations = [
            # Feasible for both
            ObservationData(
                parameter_values={"x1": 0.5},
                objective_values={"yield": 0.7, "purity": 0.8},
            ),
            # Infeasible: yield < 0.5
            ObservationData(
                parameter_values={"x1": 0.2},
                objective_values={"yield": 0.3, "purity": 0.7},
            ),
            # Infeasible: purity < 0.6
            ObservationData(
                parameter_values={"x1": 0.8},
                objective_values={"yield": 0.8, "purity": 0.4},
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_three_constraints(self, rng: np.random.Generator) -> None:
        """Test with three outcome constraints."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="value", minimize=True)],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="value",
                    threshold=10.0,
                    greater_than=True,
                ),
                OutcomeConstraintSpec(
                    objective_name="value",
                    threshold=100.0,
                    greater_than=False,
                ),
                OutcomeConstraintSpec(
                    objective_name="value",
                    threshold=50.0,
                    greater_than=False,
                ),
            ],
        )

        observations = [
            ObservationData(parameter_values={"x1": 0.3}, objective_values={"value": 30.0}),
            ObservationData(parameter_values={"x1": 0.5}, objective_values={"value": 40.0}),
            ObservationData(parameter_values={"x1": 0.7}, objective_values={"value": 20.0}),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2


class TestFeasibilityConversion:
    """Test that feasibility conversion logic works correctly.

    The internal `_build_outcome_constraint_models` function converts
    objective values to binary feasibility (1 if feasible, 0 if not).
    """

    def test_greater_than_feasibility(self, rng: np.random.Generator) -> None:
        """Verify greater_than=True correctly marks feasible points.

        For yield >= 0.5: values 0.6, 0.7, 0.5 are feasible; 0.3, 0.4 are not.
        """
        spec = make_spec_with_constraint(threshold=0.5, greater_than=True)

        observations = [
            ObservationData(
                parameter_values={"x1": 0.1, "x2": 0.1}, objective_values={"yield": 0.6}
            ),  # Feasible
            ObservationData(
                parameter_values={"x1": 0.2, "x2": 0.2}, objective_values={"yield": 0.3}
            ),  # Infeasible
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.3}, objective_values={"yield": 0.7}
            ),  # Feasible
            ObservationData(
                parameter_values={"x1": 0.4, "x2": 0.4}, objective_values={"yield": 0.4}
            ),  # Infeasible
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"yield": 0.5}
            ),  # Feasible (exactly at threshold)
        ]

        # This implicitly tests the feasibility conversion via successful generation
        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_less_than_feasibility(self, rng: np.random.Generator) -> None:
        """Verify greater_than=False correctly marks feasible points.

        For cost <= 50: values 30, 40, 50 are feasible; 60, 70 are not.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="cost", minimize=True)],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="cost",
                    threshold=50.0,
                    greater_than=False,  # cost <= 50
                )
            ],
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.1}, objective_values={"cost": 30.0}
            ),  # Feasible
            ObservationData(
                parameter_values={"x1": 0.2}, objective_values={"cost": 60.0}
            ),  # Infeasible
            ObservationData(
                parameter_values={"x1": 0.3}, objective_values={"cost": 40.0}
            ),  # Feasible
            ObservationData(
                parameter_values={"x1": 0.4}, objective_values={"cost": 70.0}
            ),  # Infeasible
            ObservationData(
                parameter_values={"x1": 0.5}, objective_values={"cost": 50.0}
            ),  # Feasible (exactly at threshold)
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_threshold_boundary(self, rng: np.random.Generator) -> None:
        """Test behavior exactly at the threshold boundary."""
        spec = make_spec_with_constraint(threshold=0.5, greater_than=True)

        # All observations exactly at threshold
        observations = [
            ObservationData(
                parameter_values={"x1": 0.1, "x2": 0.1}, objective_values={"yield": 0.5}
            ),
            ObservationData(
                parameter_values={"x1": 0.2, "x2": 0.2}, objective_values={"yield": 0.5}
            ),
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.3}, objective_values={"yield": 0.5}
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2


class TestOutcomeConstraintWithMinimization:
    """Test outcome constraints with minimization objectives.

    Reference: Constraints should work regardless of whether the
    objective is being minimized or maximized.
    """

    def test_minimize_with_greater_than_constraint(self, rng: np.random.Generator) -> None:
        """Minimize cost while ensuring yield >= threshold."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="cost", minimize=True)],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="cost",
                    threshold=20.0,
                    greater_than=True,  # cost >= 20 (minimum quality)
                )
            ],
        )

        observations = [
            ObservationData(parameter_values={"x1": 0.2}, objective_values={"cost": 30.0}),
            ObservationData(parameter_values={"x1": 0.4}, objective_values={"cost": 25.0}),
            ObservationData(parameter_values={"x1": 0.6}, objective_values={"cost": 15.0}),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_minimize_with_less_than_constraint(self, rng: np.random.Generator) -> None:
        """Minimize cost while ensuring cost <= threshold (budget constraint)."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="cost", minimize=True)],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="cost",
                    threshold=50.0,
                    greater_than=False,  # cost <= 50 (budget)
                )
            ],
        )

        observations = [
            ObservationData(parameter_values={"x1": 0.2}, objective_values={"cost": 30.0}),
            ObservationData(parameter_values={"x1": 0.5}, objective_values={"cost": 60.0}),
            ObservationData(parameter_values={"x1": 0.8}, objective_values={"cost": 40.0}),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2


class TestOutcomeConstraintDataDistribution:
    """Test outcome constraints with various data distributions."""

    def test_imbalanced_feasibility(self, rng: np.random.Generator) -> None:
        """Test with highly imbalanced feasibility (95% feasible).

        This tests the GP's ability to model rare infeasibility.
        """
        spec = make_spec_with_constraint(threshold=0.1, greater_than=True)

        observations = []
        # 19 feasible points
        for i in range(19):
            observations.append(
                ObservationData(
                    parameter_values={"x1": i / 20, "x2": i / 20},
                    objective_values={"yield": 0.5 + i * 0.01},
                )
            )
        # 1 infeasible point
        observations.append(
            ObservationData(
                parameter_values={"x1": 0.95, "x2": 0.95},
                objective_values={"yield": 0.05},
            )
        )

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_extreme_threshold_values(self, rng: np.random.Generator) -> None:
        """Test with extreme threshold values."""
        # Very high threshold (almost impossible to satisfy)
        spec_high = make_spec_with_constraint(threshold=0.999, greater_than=True)

        observations = [
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.4}, objective_values={"yield": 0.5}
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.6}, objective_values={"yield": 0.6}
            ),
            ObservationData(
                parameter_values={"x1": 0.7, "x2": 0.8}, objective_values={"yield": 0.7}
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec_high,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_negative_threshold(self, rng: np.random.Generator) -> None:
        """Test with negative threshold value."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="delta", minimize=True)],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="delta",
                    threshold=-0.5,
                    greater_than=True,  # delta >= -0.5
                )
            ],
        )

        observations = [
            ObservationData(parameter_values={"x1": 0.2}, objective_values={"delta": -0.3}),
            ObservationData(parameter_values={"x1": 0.5}, objective_values={"delta": -0.8}),
            ObservationData(parameter_values={"x1": 0.8}, objective_values={"delta": 0.1}),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2


class TestOutcomeConstraintMultiObjective:
    """Test outcome constraints in multi-objective optimization scenarios."""

    def test_multi_objective_with_constraint(self, rng: np.random.Generator) -> None:
        """Test outcome constraint with two objectives."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="yield", minimize=False),
                ObjectiveSpec(name="purity", minimize=False),
            ],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="yield",
                    threshold=0.5,
                    greater_than=True,
                )
            ],
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.4},
                objective_values={"yield": 0.6, "purity": 0.7},
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.6},
                objective_values={"yield": 0.4, "purity": 0.8},
            ),
            ObservationData(
                parameter_values={"x1": 0.7, "x2": 0.8},
                objective_values={"yield": 0.7, "purity": 0.5},
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_pareto_optimization_with_constraints(self, rng: np.random.Generator) -> None:
        """Test that constraints work correctly in Pareto optimization.

        The constraint should filter feasible points while Pareto
        dominance determines the optimal front.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f1", minimize=True),
                ObjectiveSpec(name="f2", minimize=True),
            ],
            batch_size=2,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="f1",
                    threshold=5.0,
                    greater_than=False,  # f1 <= 5.0
                )
            ],
        )

        observations = [
            # Feasible Pareto points
            ObservationData(parameter_values={"x1": 0.2}, objective_values={"f1": 2.0, "f2": 8.0}),
            ObservationData(parameter_values={"x1": 0.4}, objective_values={"f1": 4.0, "f2": 4.0}),
            # Infeasible (f1 > 5.0)
            ObservationData(parameter_values={"x1": 0.8}, objective_values={"f1": 6.0, "f2": 2.0}),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2
