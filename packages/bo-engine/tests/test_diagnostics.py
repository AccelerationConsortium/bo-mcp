"""Tests for BO engine diagnostics."""

import torch

from bo_engine.diagnostics import (
    compute_hypervolume,
    compute_pareto_front,
    summarize_pareto_front,
)


class TestDiagnostics:
    """Tests for BO diagnostics functions."""

    def test_compute_pareto_front_simple(self):
        """compute_pareto_front finds non-dominated points."""
        # 3 points, 2 objectives (minimization)
        y = torch.tensor(
            [
                [1.0, 3.0],  # Pareto optimal
                [2.0, 2.0],  # Pareto optimal
                [3.0, 1.0],  # Pareto optimal
                [2.5, 2.5],  # Dominated by [2, 2]
            ]
        )

        pareto_y, pareto_mask = compute_pareto_front(y, minimize=True)

        # 3 points should be Pareto optimal
        assert pareto_y.shape[0] == 3
        assert pareto_mask.sum().item() == 3

    def test_compute_hypervolume(self):
        """compute_hypervolume computes correct value."""
        # Simple case: single point
        pareto_y = torch.tensor([[1.0, 1.0]])
        ref_point = torch.tensor([2.0, 2.0])

        hv = compute_hypervolume(pareto_y, ref_point)
        # Area = (2-1) * (2-1) = 1
        assert hv == 1.0

    def test_compute_hypervolume_empty(self):
        """compute_hypervolume returns 0 for empty Pareto front."""
        pareto_y = torch.zeros((0, 2))
        ref_point = torch.tensor([1.0, 1.0])

        hv = compute_hypervolume(pareto_y, ref_point)
        assert hv == 0.0

    def test_summarize_pareto_front(self):
        """summarize_pareto_front formats points correctly."""
        pareto_y = torch.tensor(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        )
        objective_names = ["cost", "time"]

        summary = summarize_pareto_front(pareto_y, objective_names)

        assert len(summary) == 2
        assert summary[0]["cost"] == 1.0
        assert summary[0]["time"] == 2.0
        assert summary[1]["cost"] == 2.0
        assert summary[1]["time"] == 1.0
