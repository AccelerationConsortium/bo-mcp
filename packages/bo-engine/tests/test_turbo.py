"""Tests for TuRBO (Trust Region BO) implementation."""

import math

import pytest
import torch

from bo_engine import (
    TurboState,
    create_turbo_state,
    should_use_turbo,
    update_turbo_state,
)


class TestTurboState:
    """Test TurboState dataclass."""

    def test_create_turbo_state(self) -> None:
        """create_turbo_state creates a valid state."""
        state = create_turbo_state(dim=10, batch_size=4)
        assert state.dim == 10
        assert state.batch_size == 4
        assert state.length == 0.8
        assert state.restart_triggered is False

    def test_failure_tolerance_computed(self) -> None:
        """Failure tolerance is computed from dim and batch_size."""
        state = create_turbo_state(dim=20, batch_size=4)
        # Default: ceil(max(4/batch, dim/batch))
        expected = math.ceil(max(4.0 / 4, 20 / 4))  # ceil(max(1, 5)) = 5
        assert state.failure_tolerance == expected

    def test_initial_best_value(self) -> None:
        """Initial best value can be set."""
        state = create_turbo_state(dim=10, batch_size=2, initial_best_value=5.0)
        assert state.best_value == 5.0

    def test_length_bounds(self) -> None:
        """Length min and max are set correctly."""
        state = create_turbo_state(dim=10, batch_size=2)
        assert state.length_min == pytest.approx(0.5**7)
        assert state.length_max == 1.6


class TestUpdateTurboState:
    """Test update_turbo_state function."""

    def test_success_increments_counter(self) -> None:
        """Success increments success counter and resets failure counter."""
        state = create_turbo_state(dim=10, batch_size=2, initial_best_value=0.0)
        # Provide improvement
        state = update_turbo_state(state, torch.tensor([1.0]))

        assert state.success_counter == 1
        assert state.failure_counter == 0

    def test_failure_increments_counter(self) -> None:
        """Failure increments failure counter and resets success counter."""
        state = create_turbo_state(dim=10, batch_size=2, initial_best_value=10.0)
        state = update_turbo_state(state, torch.tensor([5.0]))  # Not improving

        assert state.failure_counter == 1
        assert state.success_counter == 0

    def test_expansion_on_sustained_success(self) -> None:
        """Trust region expands after success_tolerance consecutive improvements."""
        state = TurboState(
            dim=10,
            batch_size=2,
            success_counter=9,  # One more success triggers expansion
            success_tolerance=10,
            failure_tolerance=5,
            length=0.4,
            length_max=1.6,
            best_value=0.0,
        )
        initial_length = state.length

        # Trigger expansion with improvement
        state = update_turbo_state(state, torch.tensor([1.0]))  # Better than best_value=0

        assert state.length == min(2.0 * initial_length, state.length_max)
        assert state.success_counter == 0  # Reset after expansion

    def test_contraction_on_sustained_failure(self) -> None:
        """Trust region contracts after failure_tolerance consecutive failures."""
        state = TurboState(
            dim=10,
            batch_size=2,
            failure_counter=4,  # One more failure triggers contraction
            failure_tolerance=5,
            length=0.8,
            best_value=10.0,
        )
        initial_length = state.length

        # Trigger contraction
        state = update_turbo_state(state, torch.tensor([5.0]))  # Not improving

        assert state.length == initial_length / 2.0
        assert state.failure_counter == 0  # Reset after contraction

    def test_best_value_updated(self) -> None:
        """Best value is updated when improvement found."""
        state = create_turbo_state(dim=10, batch_size=2, initial_best_value=5.0)
        state = update_turbo_state(state, torch.tensor([10.0]))

        assert state.best_value == 10.0

    def test_best_value_not_decreased(self) -> None:
        """Best value is not decreased on failure."""
        state = create_turbo_state(dim=10, batch_size=2, initial_best_value=10.0)
        state = update_turbo_state(state, torch.tensor([5.0]))

        assert state.best_value == 10.0

    def test_restart_triggered_when_length_too_small(self) -> None:
        """Restart is triggered when length falls below minimum."""
        # Start with a length that after one contraction will be below minimum
        state = TurboState(
            dim=10,
            batch_size=2,
            length=0.01,  # After contraction: 0.005 < 0.0078
            length_min=0.5**7,  # ~0.0078
            failure_counter=4,  # One more failure triggers contraction
            failure_tolerance=5,
            best_value=10.0,
        )

        # Trigger contraction with non-improving value
        state = update_turbo_state(state, torch.tensor([5.0]))

        # Length should now be 0.005, which is less than 0.0078
        assert state.restart_triggered

    def test_handles_batch_tensor(self) -> None:
        """Handles batch of objective values."""
        state = create_turbo_state(dim=10, batch_size=4, initial_best_value=0.0)
        batch_values = torch.tensor([1.0, 2.0, 3.0, 5.0])  # Best is 5.0

        state = update_turbo_state(state, batch_values)

        assert state.best_value == 5.0

    def test_handles_2d_tensor(self) -> None:
        """Handles 2D tensor [batch_size, 1]."""
        state = create_turbo_state(dim=10, batch_size=4, initial_best_value=0.0)
        batch_values = torch.tensor([[1.0], [2.0], [3.0], [5.0]])

        state = update_turbo_state(state, batch_values)

        assert state.best_value == 5.0


class TestShouldUseTurbo:
    """Test should_use_turbo helper."""

    def test_low_dim_returns_false(self) -> None:
        """Low dimensional problems don't need TuRBO."""
        assert should_use_turbo(5) is False
        assert should_use_turbo(10) is False
        assert should_use_turbo(19) is False

    def test_high_dim_returns_true(self) -> None:
        """High dimensional problems benefit from TuRBO."""
        assert should_use_turbo(20) is True
        assert should_use_turbo(50) is True
        assert should_use_turbo(100) is True

    def test_custom_threshold(self) -> None:
        """Custom threshold can be specified."""
        assert should_use_turbo(10, threshold=5) is True
        assert should_use_turbo(10, threshold=15) is False


class TestTurboStateImmutability:
    """Test that TurboState updates return new instances."""

    def test_update_returns_new_instance(self) -> None:
        """update_turbo_state returns a new state instance."""
        state1 = create_turbo_state(dim=10, batch_size=2, initial_best_value=0.0)
        state2 = update_turbo_state(state1, torch.tensor([5.0]))

        # Should be different objects
        assert state1 is not state2
        # Original should be unchanged
        assert state1.best_value == 0.0
        assert state2.best_value == 5.0
