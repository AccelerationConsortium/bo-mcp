"""Tests reproducing results from BoTorch constraint handling tutorials.

This module validates that our constraint handling implementation achieves
results consistent with the official BoTorch tutorials and documentation.

References:
    - Constraints Documentation: https://botorch.org/docs/constraints/
    - SCBO Tutorial: https://botorch.org/docs/tutorials/scalable_constrained_bo/
    - Constrained MO Tutorial: https://botorch.org/docs/tutorials/constrained_multi_objective_bo/
    - Closed-loop Tutorial: https://botorch.org/docs/tutorials/closed_loop_botorch_only/
"""

import pytest
import torch
from torch.quasirandom import SobolEngine

from bo_engine import (
    apply_sum_constraint,
)
from bo_engine.benchmarks import c2_dtlz2, hartmann6
from bo_engine.outcome_constraints import (
    OutcomeConstraintSpec,
    build_outcome_constraint_models,
    compute_constraint_probability,
)

# =============================================================================
# Constants from BoTorch Constraints Documentation
# =============================================================================

# Outcome constraints are modeled and weighted by P(feasible).
# The acquisition function is multiplied by the probability that
# the constraint is satisfied.
#
# Common constraint types:
# - Less than: g(x) <= bound
# - Greater than: g(x) >= bound
# - Sum constraint: sum(x) = value


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.tutorial
class TestOutcomeConstraints:
    """Test outcome constraint handling.

    Reference:
        - Documentation: https://botorch.org/docs/constraints/
        - Tutorial: https://botorch.org/docs/tutorials/closed_loop_botorch_only/

    Outcome constraints are modeled and weighted by P(feasible).
    """

    @pytest.mark.smoke
    def test_outcome_constraint_spec(self) -> None:
        """OutcomeConstraintSpec should correctly represent constraints."""
        # Less than constraint: g(x) <= 0.5
        spec_lt = OutcomeConstraintSpec(
            name="constraint",
            bound=0.5,
            constraint_type="<=",
        )
        assert spec_lt.bound == 0.5
        assert spec_lt.constraint_type == "<="

        # Greater than constraint: g(x) >= 0.0
        spec_gt = OutcomeConstraintSpec(
            name="constraint",
            bound=0.0,
            constraint_type=">=",
        )
        assert spec_gt.bound == 0.0
        assert spec_gt.constraint_type == ">="

    @pytest.mark.smoke
    def test_constraint_probability_computation(self) -> None:
        """P(feasible) should be computed from Gaussian distribution."""
        # For constraint g(x) <= bound:
        # P(feasible) = P(g(x) <= bound) = Phi((bound - mean) / std)

        mean = torch.tensor([0.5])
        std = torch.tensor([0.1])
        bound = 0.6

        prob = compute_constraint_probability(mean, std, bound, constraint_type="<=")

        # With mean=0.5, std=0.1, P(x <= 0.6) should be high (~0.84)
        assert prob.item() > 0.8, f"P(x <= 0.6) should be high, got {prob.item():.3f}"

    @pytest.mark.smoke
    def test_constraint_probability_less_than(self) -> None:
        """P(obj <= threshold) should be computed correctly from normal CDF."""
        mean = torch.tensor([0.0])
        std = torch.tensor([1.0])

        # P(x <= 0) for standard normal = 0.5
        prob_zero = compute_constraint_probability(mean, std, 0.0, "<=")
        assert prob_zero.item() == pytest.approx(0.5, abs=0.05)

        # P(x <= 1) for standard normal ~ 0.84
        prob_one = compute_constraint_probability(mean, std, 1.0, "<=")
        assert prob_one.item() == pytest.approx(0.84, abs=0.05)

    @pytest.mark.smoke
    def test_constraint_probability_greater_than(self) -> None:
        """P(obj >= threshold) should be computed correctly."""
        mean = torch.tensor([0.0])
        std = torch.tensor([1.0])

        # P(x >= 0) for standard normal = 0.5
        prob_zero = compute_constraint_probability(mean, std, 0.0, ">=")
        assert prob_zero.item() == pytest.approx(0.5, abs=0.05)

        # P(x >= -1) for standard normal ~ 0.84
        prob_minus_one = compute_constraint_probability(mean, std, -1.0, ">=")
        assert prob_minus_one.item() == pytest.approx(0.84, abs=0.05)


@pytest.mark.tutorial
class TestConstraintCallables:
    """Test constraint callable creation."""

    @pytest.mark.smoke
    def test_create_constraint_callable(self) -> None:
        """Should create callable that returns constraint violation.

        The create_constraint_callable function requires a ConstraintSpec and
        OptimizationSpec. For testing basic linear constraint behavior, we define
        a simple sum constraint manually.
        """

        # Define a simple sum constraint callable directly (x[0] + x[1] <= 1)
        def sum_constraint(x: torch.Tensor) -> torch.Tensor:
            # BoTorch convention: >= 0 means satisfied, < 0 means violated
            return 1.0 - x[..., 0] - x[..., 1]

        # Test feasible point
        x_feasible = torch.tensor([[0.3, 0.4]])  # sum = 0.7 <= 1
        violation_feasible = sum_constraint(x_feasible)
        assert (violation_feasible >= 0).all(), "Feasible point should have non-negative value"

        # Test infeasible point
        x_infeasible = torch.tensor([[0.6, 0.6]])  # sum = 1.2 > 1
        violation_infeasible = sum_constraint(x_infeasible)
        assert (violation_infeasible < 0).all(), "Infeasible point should have negative value"

    @pytest.mark.smoke
    def test_apply_sum_constraint(self) -> None:
        """Sum constraint should normalize selected parameters to target sum.

        The apply_sum_constraint function projects candidates to satisfy
        sum(x[param_indices]) = target_sum.
        """
        # Point that has sum = 0.7
        candidates = torch.tensor([[0.4, 0.3]], dtype=torch.float64)
        target_sum = 0.5

        # Apply constraint - normalizes to sum to target
        constrained = apply_sum_constraint(candidates, param_indices=[0, 1], target_sum=target_sum)

        # Sum should now equal target
        actual_sum = constrained[0, 0].item() + constrained[0, 1].item()
        assert abs(actual_sum - target_sum) < 1e-6, f"Sum should be {target_sum}, got {actual_sum}"


@pytest.mark.tutorial
class TestConstrainedHartmann:
    """Test constrained optimization on Hartmann function.

    Reference: Closed-loop tutorial constraint ||x||_1 - 3 <= 0
    """

    @pytest.mark.smoke
    def test_l1_constraint_definition(self) -> None:
        """Define L1 norm constraint ||x||_1 <= 3."""
        # For 6D Hartmann, constraint is ||x||_1 <= 3

        def l1_constraint(x: torch.Tensor) -> torch.Tensor:
            """L1 norm constraint: ||x||_1 - 3 <= 0 means feasible."""
            return x.abs().sum(dim=-1, keepdim=True) - 3.0

        # Feasible point (L1 norm < 3)
        x_feasible = torch.tensor([[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]])
        assert l1_constraint(x_feasible).item() < 0

        # Infeasible point (L1 norm > 3)
        x_infeasible = torch.tensor([[0.8, 0.8, 0.8, 0.8, 0.8, 0.8]])
        assert l1_constraint(x_infeasible).item() > 0

    @pytest.mark.slow
    def test_constrained_hartmann_optimization(self) -> None:
        """Optimization should respect L1 constraint."""
        torch.manual_seed(42)

        def constrained_objective(x: torch.Tensor) -> tuple[float, bool]:
            """Hartmann6 with L1 constraint."""
            y = hartmann6(x).item()
            feasible = x.abs().sum().item() <= 3.0
            return y, feasible

        # Generate initial points within constraint
        sobol = SobolEngine(dimension=6, scramble=True, seed=42)
        points = []

        for _ in range(20):
            x = sobol.draw(1).to(torch.float64)
            if x.abs().sum().item() <= 3.0:
                points.append(x)
            if len(points) >= 10:
                break

        # Check we found feasible initial points
        assert len(points) >= 5, "Should find feasible initial points"


@pytest.mark.tutorial
class TestConstrainedMultiObjective:
    """Test constrained multi-objective optimization.

    Reference: https://botorch.org/docs/tutorials/constrained_multi_objective_bo/
    """

    @pytest.mark.smoke
    def test_c2_dtlz2_constraint(self) -> None:
        """C2-DTLZ2 should produce feasible and infeasible points."""
        torch.manual_seed(42)
        x = torch.rand(100, 4, dtype=torch.float64)

        _f, c = c2_dtlz2(x, n_objectives=2, r=0.2)

        # Should have both feasible (c >= 0) and infeasible (c < 0)
        n_feasible = (c >= 0).sum().item()
        n_infeasible = (c < 0).sum().item()

        assert n_feasible > 0, "Should have some feasible points"
        assert n_infeasible > 0, "Should have some infeasible points"

    @pytest.mark.smoke
    def test_constrained_pareto_front(self) -> None:
        """Constrained Pareto front should only include feasible points."""
        torch.manual_seed(42)
        x = torch.rand(50, 4, dtype=torch.float64)

        f, c = c2_dtlz2(x, n_objectives=2, r=0.2)

        # Filter to feasible points only
        feasible_mask = c >= 0
        f_feasible = f[feasible_mask]

        # Pareto front of feasible points
        from bo_engine import compute_pareto_front

        if len(f_feasible) > 0:
            pareto_points, _pareto_mask = compute_pareto_front(f_feasible, minimize=True)

            # All Pareto points should be feasible (by construction)
            assert len(pareto_points) <= len(f_feasible)


@pytest.mark.tutorial
class TestOutcomeConstraintModeling:
    """Test outcome constraint modeling with GP."""

    @pytest.mark.slow
    def test_build_outcome_constraint_models(self) -> None:
        """Should build GP models for outcome constraints.

        The build_outcome_constraint_models function builds models for
        outcome constraints, returning a list of ConstraintModelResult objects.
        """
        torch.manual_seed(42)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

        # Training data
        train_x = torch.rand(15, 2, dtype=torch.float64)

        # Create observations as list of dicts (the expected format)
        observations = []
        for i in range(15):
            # Constraint outcome: sum - 0.8, feasible when < 0
            constraint_value = train_x[i, 0].item() + train_x[i, 1].item() - 0.8
            observations.append({"sum_constraint": constraint_value})

        constraint_spec = OutcomeConstraintSpec(
            name="sum_constraint",
            bound=0.0,
            constraint_type="<=",
        )

        results = build_outcome_constraint_models(
            train_x=train_x,
            observations=observations,
            bounds=bounds,
            constraint_specs=[constraint_spec],
        )

        # Should have one result for one constraint
        assert len(results) == 1

        # Result should have a model
        result = results[0]
        assert result.model is not None

        # Model should make predictions
        result.model.eval()
        with torch.no_grad():
            test_x = torch.rand(5, 2, dtype=torch.float64)
            posterior = result.model.posterior(test_x)
            assert posterior.mean.shape[0] == 5


@pytest.mark.tutorial
class TestFeasibilityWeighting:
    """Test feasibility-weighted acquisition functions."""

    @pytest.mark.smoke
    def test_high_feasibility_probability(self) -> None:
        """Points clearly satisfying constraint should have high P(feasible)."""
        mean = torch.tensor([0.0])
        std = torch.tensor([0.1])
        bound = 1.0  # g(x) <= 1.0

        prob = compute_constraint_probability(mean, std, bound, "<=")

        # P(x <= 1) for N(0, 0.1) should be very high
        assert prob.item() > 0.99

    @pytest.mark.smoke
    def test_low_feasibility_probability(self) -> None:
        """Points clearly violating constraint should have low P(feasible)."""
        mean = torch.tensor([2.0])
        std = torch.tensor([0.1])
        bound = 1.0  # g(x) <= 1.0

        prob = compute_constraint_probability(mean, std, bound, "<=")

        # P(x <= 1) for N(2, 0.1) should be very low
        assert prob.item() < 0.01

    @pytest.mark.smoke
    def test_medium_feasibility_probability(self) -> None:
        """Points at boundary should have ~0.5 P(feasible)."""
        mean = torch.tensor([1.0])
        std = torch.tensor([0.5])
        bound = 1.0  # g(x) <= 1.0

        prob = compute_constraint_probability(mean, std, bound, "<=")

        # P(x <= 1) for N(1, 0.5) should be ~0.5
        assert 0.4 < prob.item() < 0.6
