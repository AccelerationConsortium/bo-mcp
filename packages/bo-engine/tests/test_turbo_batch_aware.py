"""Tests for the batch-aware TuRBO success-counter update (8.34).

The original TuRBO (Eriksson et al., NeurIPS 2019, Algorithm 1) counts a
"success" per batch when *any* point in the batch improves the incumbent.
The previous implementation incremented ``success_counter`` by exactly 1
regardless of how many points in the batch improved, which paced the
trust-region expansion conservatively for batch sizes > 1. The batch-aware
update increments by the number of improving points in the batch (capped
at ``batch_size``).

References:
    - Eriksson et al., "Scalable Global Optimization via Local Bayesian
      Optimization", NeurIPS 2019 — Algorithm 1, success/failure update.
    - BoTorch TuRBO tutorial:
      https://botorch.org/tutorials/turbo_1
"""

from __future__ import annotations

import torch

from bo_engine.turbo import TurboState, update_turbo_state


def _make_state(batch_size: int, success_tolerance: int) -> TurboState:
    return TurboState(
        dim=10,
        batch_size=batch_size,
        success_tolerance=success_tolerance,
        # Best so far is 0; improvements are positive numbers in TuRBO's
        # internal maximization convention (which is what update_turbo_state
        # converts to when ``minimize=False``).
        best_value=0.0,
    )


class TestSuccessCounterBatchAware:
    """Counter increment must match the number of improving points in the batch."""

    def test_single_improving_point_increments_by_one(self) -> None:
        state = _make_state(batch_size=4, success_tolerance=10)
        # Only one of four points beats the incumbent.
        y_next = torch.tensor([0.5, -0.1, -0.2, -0.3], dtype=torch.float64)

        updated = update_turbo_state(state, y_next, minimize=False)

        assert updated.success_counter == 1
        assert updated.failure_counter == 0

    def test_all_improving_points_increment_by_batch_size(self) -> None:
        state = _make_state(batch_size=4, success_tolerance=10)
        y_next = torch.tensor([0.5, 0.6, 0.7, 0.8], dtype=torch.float64)

        updated = update_turbo_state(state, y_next, minimize=False)

        assert updated.success_counter == 4

    def test_no_improving_points_falls_back_to_failure_counter(self) -> None:
        state = _make_state(batch_size=4, success_tolerance=10)
        # No point strictly beats best_value=0 by more than the tolerance.
        y_next = torch.tensor([-0.5, -0.4, -0.3, -0.2], dtype=torch.float64)

        updated = update_turbo_state(state, y_next, minimize=False)

        assert updated.success_counter == 0
        assert updated.failure_counter == 1


class TestExpansionCadenceMatchesPaperOnLargerBatches:
    """A b=4 campaign should expand the trust region at least as fast as b=1.

    Previously the b=4 path increased the success counter by 1 per batch
    even when all 4 points improved, so reaching ``success_tolerance=10``
    took ~10 batches — exactly the same as b=1. With the batch-aware fix
    b=4 reaches the same threshold in ~3 batches when every batch improves
    its full quota.
    """

    def test_batch_four_reaches_expansion_in_three_batches(self) -> None:
        state = _make_state(batch_size=4, success_tolerance=10)
        # 3 successive batches where every point improves.
        for _ in range(3):
            state = update_turbo_state(
                state,
                torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64) + state.best_value,
                minimize=False,
            )
        # 4 + 4 + 4 = 12 ≥ 10 → expansion fires on the third batch and the
        # counter resets to 0.
        assert state.success_counter == 0
        # And the trust region length must have expanded.
        initial_length = TurboState(dim=10, batch_size=4).length
        assert state.length > initial_length


class TestBatchSizeOneRegression:
    """The b=1 path must keep its original semantics (one improvement per batch)."""

    def test_b1_increments_by_one_per_improving_batch(self) -> None:
        state = _make_state(batch_size=1, success_tolerance=3)
        for _ in range(3):
            state = update_turbo_state(
                state,
                torch.tensor([state.best_value + 0.1], dtype=torch.float64),
                minimize=False,
            )
        # Three successive improvements → counter resets at success_tolerance=3
        # and length expands.
        assert state.success_counter == 0
        initial_length = TurboState(dim=10, batch_size=1).length
        assert state.length > initial_length
