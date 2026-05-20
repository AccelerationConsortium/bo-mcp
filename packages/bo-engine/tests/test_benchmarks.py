"""Tests for benchmark functions.

These tests verify that benchmark functions behave correctly
and can be used for testing BO algorithms.
"""

import pytest
import torch

from bo_engine.benchmarks import (
    augmented_hartmann,
    branin,
    branin_50d,
    branin_100d,
    branin_bounds,
    branin_currin,
    c2_dtlz2,
    dtlz2,
    get_benchmark_spec,
    hartmann6,
    list_benchmarks,
)


class TestSingleObjectiveBenchmarks:
    """Tests for single-objective benchmark functions."""

    def test_branin_shape(self):
        """branin returns correct output shape."""
        x = torch.rand(10, 2)
        y = branin(x)
        assert y.shape == (10,)

    def test_branin_minimum(self):
        """branin has correct minimum value."""
        # One of the global minima
        x_opt = torch.tensor([[-3.14159, 12.275]])
        y = branin(x_opt)
        assert y.item() == pytest.approx(0.397887, rel=0.01)

    def test_branin_bounds(self):
        """branin_bounds returns correct shape."""
        bounds = branin_bounds()
        assert bounds.shape == (2, 2)
        assert bounds[0, 0].item() == -5.0
        assert bounds[1, 0].item() == 10.0

    def test_hartmann6_shape(self):
        """hartmann6 returns correct output shape."""
        x = torch.rand(10, 6)
        y = hartmann6(x)
        assert y.shape == (10,)

    def test_hartmann6_minimum(self):
        """hartmann6 has correct minimum value."""
        x_opt = torch.tensor([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
        y = hartmann6(x_opt)
        assert y.item() == pytest.approx(-3.32237, rel=0.01)

    def test_branin_50d_shape(self):
        """branin_50d returns correct output shape."""
        x = torch.rand(5, 50)
        y = branin_50d(x)
        assert y.shape == (5,)

    def test_branin_50d_irrelevant_dims(self):
        """branin_50d ignores irrelevant dimensions."""
        # Same first 2 dims, different rest
        x1 = torch.zeros(1, 50)
        x2 = torch.ones(1, 50)
        x1[0, :2] = torch.tensor([0.5, 0.5])
        x2[0, :2] = torch.tensor([0.5, 0.5])

        y1 = branin_50d(x1)
        y2 = branin_50d(x2)
        assert y1.item() == pytest.approx(y2.item())

    def test_branin_100d_shape(self):
        """branin_100d returns correct output shape."""
        x = torch.rand(5, 100)
        y = branin_100d(x)
        assert y.shape == (5,)


class TestMultiObjectiveBenchmarks:
    """Tests for multi-objective benchmark functions."""

    def test_branin_currin_shape(self):
        """branin_currin returns correct output shape."""
        x = torch.rand(10, 2)
        y = branin_currin(x)
        assert y.shape == (10, 2)

    def test_branin_currin_range(self):
        """branin_currin outputs are in reasonable range."""
        x = torch.rand(100, 2)
        y = branin_currin(x)
        # Both objectives should be positive
        assert (y >= 0).all()
        # Scaled to reasonable range (branin objective can be up to ~6)
        assert (y <= 10).all()

    def test_dtlz2_shape(self):
        """dtlz2 returns correct output shape."""
        x = torch.rand(10, 4)
        y = dtlz2(x, n_objectives=2)
        assert y.shape == (10, 2)

    def test_dtlz2_pareto_front(self):
        """dtlz2 Pareto front lies on unit sphere."""
        # On Pareto front, all objectives equal and sum of squares = 1
        x = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # Pareto optimal point
        y = dtlz2(x, n_objectives=2)
        # Check on unit sphere
        assert torch.norm(y).item() == pytest.approx(1.0, rel=0.1)

    def test_dtlz2_scalable(self):
        """dtlz2 can scale to more objectives."""
        x = torch.rand(5, 6)
        y = dtlz2(x, n_objectives=3)
        assert y.shape == (5, 3)

    def test_c2_dtlz2_shape(self):
        """c2_dtlz2 returns correct shapes for objectives and constraints."""
        x = torch.rand(10, 4)
        f, c = c2_dtlz2(x, n_objectives=2)
        assert f.shape == (10, 2)
        assert c.shape == (10,)

    def test_c2_dtlz2_feasibility(self):
        """c2_dtlz2 constraint identifies feasible/infeasible points."""
        x = torch.rand(100, 4)
        _f, c = c2_dtlz2(x, n_objectives=2, r=0.2)
        # Some points should be feasible (c >= 0), some infeasible
        n_feasible = (c >= 0).sum().item()
        n_infeasible = (c < 0).sum().item()
        # With r=0.2, expect mix of feasible and infeasible
        assert n_feasible > 0
        assert n_infeasible > 0


class TestMultiFidelityBenchmarks:
    """Tests for multi-fidelity benchmark functions."""

    def test_augmented_hartmann_shape(self):
        """augmented_hartmann returns correct output shape."""
        x = torch.rand(10, 7)
        y = augmented_hartmann(x)
        assert y.shape == (10,)

    def test_augmented_hartmann_high_fidelity(self):
        """augmented_hartmann at fidelity=1 equals hartmann6."""
        x = torch.rand(5, 7)
        x[..., 6] = 1.0  # Set fidelity to 1

        y_aug = augmented_hartmann(x)
        y_h6 = hartmann6(x[..., :6])

        # Should be very close (fidelity=1 means no bias)
        assert torch.allclose(y_aug, y_h6, atol=0.1)

    def test_augmented_hartmann_fidelity_effect(self):
        """augmented_hartmann varies with fidelity."""
        x = torch.rand(1, 7)
        x_low = x.clone()
        x_low[..., 6] = 0.0
        x_high = x.clone()
        x_high[..., 6] = 1.0

        y_low = augmented_hartmann(x_low)
        y_high = augmented_hartmann(x_high)

        # Low fidelity should differ from high fidelity
        assert not torch.allclose(y_low, y_high)


class TestBenchmarkRegistry:
    """Tests for benchmark registry functions."""

    def test_list_benchmarks(self):
        """list_benchmarks returns expected benchmarks."""
        benchmarks = list_benchmarks()
        assert "branin" in benchmarks
        assert "hartmann6" in benchmarks
        assert "branin_currin" in benchmarks
        assert "dtlz2" in benchmarks
        assert len(benchmarks) >= 8

    def test_get_benchmark_spec(self):
        """get_benchmark_spec returns valid specs."""
        spec = get_benchmark_spec("branin")
        assert "function" in spec
        assert "bounds" in spec
        assert "n_dims" in spec
        assert spec["n_dims"] == 2
        assert spec["n_objectives"] == 1

    def test_get_benchmark_spec_multiobjective(self):
        """get_benchmark_spec works for multi-objective benchmarks."""
        spec = get_benchmark_spec("branin_currin")
        assert spec["n_objectives"] == 2

    def test_get_benchmark_spec_constrained(self):
        """get_benchmark_spec works for constrained benchmarks."""
        spec = get_benchmark_spec("c2_dtlz2")
        assert spec.get("has_constraints", False) is True

    def test_get_benchmark_spec_unknown(self):
        """get_benchmark_spec raises for unknown benchmark."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark_spec("nonexistent_benchmark")
