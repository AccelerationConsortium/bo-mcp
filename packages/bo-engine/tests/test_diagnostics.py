"""Tests for BO engine diagnostics."""

import pytest
import torch

from bo_engine.diagnostics import (
    compute_convergence_metric,
    compute_hypervolume,
    compute_hypervolume_improvement,
    compute_pareto_front,
    determine_progress_status,
    summarize_pareto_front,
)
from bo_engine.diagnostics_single import compute_single_objective_improvement_rate


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


class TestNumericalSafety:
    """Pin numerical-safety branches against degenerate inputs.

    The diagnostics helpers run on user data that may include zero or
    near-zero baselines (a flat initial trajectory, a degenerate
    hypervolume history, an exactly-zero initial sample). Without the
    NUMERICAL_EPSILON clamps swept into these helpers (TODO 1.18 + the
    god-module split), any of these inputs would produce ``inf`` /
    ``nan`` instead of a finite result. These tests pin the contract.

    Reference: numerical-safety guidance in project CLAUDE.md (no bare
    zero comparisons on computed floats; clamp before division).
    """

    def test_hypervolume_improvement_zero_baseline_returns_zero(self) -> None:
        """A previous hypervolume of zero must not divide by zero."""
        assert compute_hypervolume_improvement(current_hv=1.0, previous_hv=0.0) == pytest.approx(
            0.0
        )

    def test_convergence_metric_zero_baseline_returns_not_converged(self) -> None:
        """A zero previous average must surface as ``not converged`` (1.0)."""
        history = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert compute_convergence_metric(history, window=5) == pytest.approx(1.0)

    def test_determine_progress_status_zero_baseline_branch(self) -> None:
        """An exactly-zero previous value must not raise ``ZeroDivisionError``."""
        # Two-element history takes the short-circuit branch; the second
        # value is non-zero so the rate calculation must use the
        # ``is_zero`` helper instead of bare ``!= 0``.
        assert determine_progress_status([0.0, 0.5]) in {"improving", "stagnant", "regressing"}

    def test_single_objective_improvement_rate_zero_initial_returns_zero(self) -> None:
        """A zero initial best must short-circuit to a 0.0 rate."""
        history = [0.0, 0.1, 0.2, 0.3, 0.4]
        # The branch ``is_zero(initial)`` must fire here.
        assert compute_single_objective_improvement_rate(history, window=10) == pytest.approx(0.0)
