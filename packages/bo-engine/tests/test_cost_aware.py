"""Tests for cost-aware Bayesian Optimization (EIpu).

References:
- EIpu (Expected Improvement per Unit cost) concept:
  https://arxiv.org/abs/1406.2541 (Snoek et al., "Input Warping for Bayesian Optimization")
- BoTorch acquisition function patterns:
  https://botorch.org/docs/acquisition
"""

from bo_engine import (
    AcquisitionMethod,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    generate_next_batch,
)


def make_cost_aware_spec() -> OptimizationSpec:
    """Create cost-aware optimization spec."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="f", minimize=True)],
        batch_size=2,
        use_cost_aware=True,
        acquisition_method=AcquisitionMethod.COST_WEIGHTED_EI,
    )


def generate_observations_with_cost() -> list[ObservationData]:
    """Generate observations with cost data."""
    return [
        ObservationData(
            parameter_values={"x1": 0.1, "x2": 0.2},
            objective_values={"f": 0.5},
            cost=10.0,
        ),
        ObservationData(
            parameter_values={"x1": 0.3, "x2": 0.4},
            objective_values={"f": 0.3},
            cost=50.0,  # Higher cost
        ),
        ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.6},
            objective_values={"f": 0.2},
            cost=100.0,  # Very high cost
        ),
        ObservationData(
            parameter_values={"x1": 0.7, "x2": 0.8},
            objective_values={"f": 0.4},
            cost=5.0,  # Low cost
        ),
        ObservationData(
            parameter_values={"x1": 0.9, "x2": 0.1},
            objective_values={"f": 0.35},
            cost=20.0,
        ),
    ]


class TestCostAwareSpec:
    """Test cost-aware optimization specification."""

    def test_eipu_acquisition_method(self) -> None:
        """EIpu acquisition method is available."""
        assert AcquisitionMethod.COST_WEIGHTED_EI.value == "cost_weighted_ei"

    def test_use_cost_aware_flag(self) -> None:
        """use_cost_aware flag is set correctly."""
        spec = make_cost_aware_spec()
        assert spec.use_cost_aware is True
        assert spec.acquisition_method == AcquisitionMethod.COST_WEIGHTED_EI


class TestCostAwareOptimization:
    """Test cost-aware BO integration."""

    def test_generates_suggestions_with_cost(self) -> None:
        """generate_next_batch works with cost-aware optimization."""
        spec = make_cost_aware_spec()
        observations = generate_observations_with_cost()

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )

        assert len(suggestions) == 2
        for sugg in suggestions:
            assert "x1" in sugg.parameter_values
            assert "x2" in sugg.parameter_values
            assert 0.0 <= sugg.parameter_values["x1"] <= 1.0
            assert 0.0 <= sugg.parameter_values["x2"] <= 1.0

    def test_acquisition_function_is_eipu(self) -> None:
        """Suggestions show EIpu as acquisition function."""
        spec = make_cost_aware_spec()
        observations = generate_observations_with_cost()

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )

        for sugg in suggestions:
            assert sugg.acquisition_function == "cost_weighted_ei"

    def test_cost_aware_with_missing_cost(self) -> None:
        """Falls back gracefully when some observations lack cost."""
        spec = make_cost_aware_spec()
        observations = [
            ObservationData(
                parameter_values={"x1": 0.1, "x2": 0.2},
                objective_values={"f": 0.5},
                cost=10.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.4},
                objective_values={"f": 0.3},
                cost=None,  # Missing cost
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.6},
                objective_values={"f": 0.2},
                cost=50.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.7, "x2": 0.8},
                objective_values={"f": 0.4},
                cost=15.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.9, "x2": 0.1},
                objective_values={"f": 0.35},
                cost=25.0,
            ),
        ]

        # Should still work (falls back to standard acquisition)
        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )

        assert len(suggestions) == 2

    def test_cost_aware_disabled(self) -> None:
        """Works normally when cost-aware is disabled."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
            batch_size=2,
            use_cost_aware=False,
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.3},
                objective_values={"f": 0.5},
                cost=10.0,  # Cost is ignored
            ),
            ObservationData(
                parameter_values={"x1": 0.7},
                objective_values={"f": 0.3},
                cost=50.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.5},
                objective_values={"f": 0.4},
                cost=30.0,
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )

        assert len(suggestions) == 2
        # Should use standard acquisition, not EIpu
        for sugg in suggestions:
            assert sugg.acquisition_function in [
                "noisy_expected_improvement",
                "expected_improvement",
            ]


class TestObservationDataWithCost:
    """Test ObservationData with cost field."""

    def test_observation_with_cost(self) -> None:
        """ObservationData can store cost."""
        obs = ObservationData(
            parameter_values={"x": 0.5},
            objective_values={"f": 0.3},
            cost=25.0,
        )
        assert obs.cost == 25.0

    def test_observation_without_cost(self) -> None:
        """ObservationData cost defaults to None."""
        obs = ObservationData(
            parameter_values={"x": 0.5},
            objective_values={"f": 0.3},
        )
        assert obs.cost is None


class TestCostAwareWithConstraints:
    """Test cost-aware BO combined with other features."""

    def test_cost_aware_with_outcome_constraint(self) -> None:
        """Cost-aware works with outcome constraints."""
        from bo_engine import OutcomeConstraintSpec

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
            batch_size=2,
            use_cost_aware=True,
            acquisition_method=AcquisitionMethod.COST_WEIGHTED_EI,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    name="f",
                    bound=0.5,
                    constraint_type="<=",  # f <= 0.5
                )
            ],
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.2},
                objective_values={"f": 0.3},  # Feasible
                cost=10.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.5},
                objective_values={"f": 0.4},  # Feasible
                cost=30.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.8},
                objective_values={"f": 0.7},  # Infeasible
                cost=5.0,
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )

        assert len(suggestions) == 2

    def test_cost_aware_not_for_multi_objective(self) -> None:
        """Cost-aware (EIpu) is single-objective only."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f1", minimize=True),
                ObjectiveSpec(name="f2", minimize=True),
            ],
            batch_size=2,
            use_cost_aware=True,  # Ignored for multi-objective
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.3},
                objective_values={"f1": 0.5, "f2": 0.4},
                cost=10.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.7},
                objective_values={"f1": 0.3, "f2": 0.6},
                cost=20.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.5},
                objective_values={"f1": 0.4, "f2": 0.5},
                cost=15.0,
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )

        assert len(suggestions) == 2
        # Should use multi-objective acquisition, not EIpu
        for sugg in suggestions:
            assert sugg.acquisition_function in [
                "hypervolume_improvement",
                "scalarized_multi_objective",
            ]
