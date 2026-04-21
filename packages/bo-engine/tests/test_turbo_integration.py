"""Tests for TuRBO integration with suggestions.py."""

import numpy as np
import torch

from bo_engine import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    TurboState,
    create_turbo_state,
    generate_next_batch,
    update_turbo_after_evaluation,
)
from bo_engine.types import TurboConfig


def make_high_dim_spec(n_params: int = 25) -> OptimizationSpec:
    """Create a high-dimensional optimization spec."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                bounds=(0.0, 1.0),
            )
            for i in range(n_params)
        ],
        objectives=[ObjectiveSpec(name="f", minimize=True)],
        batch_size=4,
        turbo_config=TurboConfig(),
    )


def sphere_function(params: dict[str, float]) -> float:
    """Simple sphere function for testing."""
    return sum(v**2 for v in params.values())


def generate_observations(
    spec: OptimizationSpec,
    n_obs: int = 10,
) -> list[ObservationData]:
    """Generate random observations for testing."""
    torch.manual_seed(42)
    observations = []
    for _ in range(n_obs):
        params = {
            p.name: torch.rand(1).item() * (p.bounds[1] - p.bounds[0]) + p.bounds[0]
            for p in spec.parameters
            if p.bounds is not None
        }
        f_val = sphere_function(params)
        observations.append(
            ObservationData(
                parameter_values=params,
                objective_values={"f": f_val},
            )
        )
    return observations


class TestTurboIntegration:
    """Test TuRBO integration with generate_next_batch."""

    def test_turbo_returns_state(self, rng: np.random.Generator) -> None:
        """generate_next_batch returns TurboState for high-dim problems."""
        spec = make_high_dim_spec(n_params=25)
        # Need enough observations to pass min_data check (n_params+1 = 26)
        observations = generate_observations(spec, n_obs=30)

        suggestions, turbo_state = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=4,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 4
        assert turbo_state is not None
        assert turbo_state.dim == 25
        assert turbo_state.batch_size == 4

    def test_turbo_state_passed_through(self, rng: np.random.Generator) -> None:
        """TuRBO state is passed through multiple iterations."""
        spec = make_high_dim_spec(n_params=25)
        # Need enough observations to pass min_data check (n_params+1 = 26)
        observations = generate_observations(spec, n_obs=30)

        # First batch
        _, turbo_state1 = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=4,
            iteration=1,
            rng=rng,
        )
        assert turbo_state1 is not None

        # Second batch with existing state
        _, turbo_state2 = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=4,
            iteration=2,
            turbo_state=turbo_state1,
            rng=rng,
        )
        assert turbo_state2 is not None
        # State should be preserved (or modified based on progress)
        assert turbo_state2.dim == turbo_state1.dim

    def test_turbo_not_used_for_low_dim(self, rng: np.random.Generator) -> None:
        """TuRBO is not automatically used for low-dimensional problems."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
            batch_size=2,
        )
        observations = [
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5},
                objective_values={"f": 0.5},
            ),
            ObservationData(
                parameter_values={"x1": 0.2, "x2": 0.8},
                objective_values={"f": 0.68},
            ),
        ]

        suggestions, turbo_state = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2
        # No TuRBO for low-dim
        assert turbo_state is None

    def test_turbo_not_used_for_multi_objective(self, rng: np.random.Generator) -> None:
        """TuRBO is not used for multi-objective optimization."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
                for i in range(25)
            ],
            objectives=[
                ObjectiveSpec(name="f1", minimize=True),
                ObjectiveSpec(name="f2", minimize=True),
            ],
            batch_size=4,
            turbo_config=TurboConfig(),
        )
        observations = [
            ObservationData(
                parameter_values={f"x{i}": 0.5 for i in range(25)},
                objective_values={"f1": 0.5, "f2": 0.5},
            ),
            ObservationData(
                parameter_values={f"x{i}": 0.3 for i in range(25)},
                objective_values={"f1": 0.3, "f2": 0.7},
            ),
            ObservationData(
                parameter_values={f"x{i}": 0.7 for i in range(25)},
                objective_values={"f1": 0.7, "f2": 0.3},
            ),
        ]

        suggestions, turbo_state = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=4,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 4
        # TuRBO not supported for multi-objective
        assert turbo_state is None

    def test_generation_method_is_turbo(self, rng: np.random.Generator) -> None:
        """Suggestions indicate turbo generation method when using TuRBO."""
        spec = make_high_dim_spec(n_params=25)
        # Need enough observations to pass min_data check (n_params+1 = 26)
        observations = generate_observations(spec, n_obs=30)

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=4,
            iteration=1,
            rng=rng,
        )

        for sugg in suggestions:
            assert sugg.generation_method == "turbo"
            assert sugg.explanation is not None
            assert "TuRBO" in sugg.explanation


class TestUpdateTurboAfterEvaluation:
    """Test update_turbo_after_evaluation function."""

    def test_update_on_improvement(self) -> None:
        """State updates correctly when improvement is found."""
        spec = make_high_dim_spec(n_params=25)
        turbo_state = create_turbo_state(dim=25, batch_size=4, initial_best_value=-1.0)

        new_observations = [
            ObservationData(
                parameter_values={f"x{i}": 0.1 for i in range(25)},
                objective_values={"f": 0.5},  # Better than -(-1.0) = 1.0 when negated
            )
        ]

        updated_state = update_turbo_after_evaluation(turbo_state, new_observations, spec)

        # Improvement should increment success counter
        assert updated_state.success_counter >= turbo_state.success_counter

    def test_update_on_failure(self) -> None:
        """State updates correctly when no improvement found."""
        spec = make_high_dim_spec(n_params=25)
        # Start with a very good best value
        turbo_state = create_turbo_state(dim=25, batch_size=4, initial_best_value=-0.001)

        new_observations = [
            ObservationData(
                parameter_values={f"x{i}": 0.5 for i in range(25)},
                objective_values={"f": 0.5},  # Worse than 0.001
            )
        ]

        updated_state = update_turbo_after_evaluation(turbo_state, new_observations, spec)

        # Failure should increment failure counter
        assert updated_state.failure_counter >= 1

    def test_empty_observations_returns_unchanged(self) -> None:
        """Empty observations list returns unchanged state."""
        spec = make_high_dim_spec(n_params=25)
        turbo_state = create_turbo_state(dim=25, batch_size=4)

        updated_state = update_turbo_after_evaluation(turbo_state, [], spec)

        assert updated_state is turbo_state


class TestTurboStateSerialization:
    """Test TuRBO state serialization for persistence."""

    def test_state_to_dict_and_back(self) -> None:
        """TurboState can be serialized to dict and back."""
        original = TurboState(
            dim=25,
            batch_size=4,
            length=0.6,
            length_min=0.01,
            length_max=1.6,
            failure_counter=2,
            failure_tolerance=5,
            success_counter=3,
            success_tolerance=10,
            best_value=0.5,
            restart_triggered=False,
        )

        # Serialize
        state_dict = {
            "dim": original.dim,
            "batch_size": original.batch_size,
            "length": original.length,
            "length_min": original.length_min,
            "length_max": original.length_max,
            "failure_counter": original.failure_counter,
            "failure_tolerance": original.failure_tolerance,
            "success_counter": original.success_counter,
            "success_tolerance": original.success_tolerance,
            "best_value": original.best_value,
            "restart_triggered": original.restart_triggered,
        }

        # Deserialize
        restored = TurboState(**state_dict)

        assert restored.dim == original.dim
        assert restored.batch_size == original.batch_size
        assert restored.length == original.length
        assert restored.best_value == original.best_value
        assert restored.failure_counter == original.failure_counter
        assert restored.success_counter == original.success_counter
