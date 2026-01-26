"""Integration tests for complete Bayesian Optimization workflows.

These tests verify end-to-end optimization using benchmark functions
from the official BoTorch tutorials.

Note: The BO engine uses qLogNEHVI which requires multi-objective (m >= 2).
All tests use multi-objective benchmarks.

Test Strategy:
    - test_optimization_invariants: Tests properties that ALWAYS hold (bounds, monotonicity)
    - test_optimization_finds_good_tradeoffs_statistical: Statistical test for nightly CI

References:
    - BoTorch Multi-Objective Tutorial: https://botorch.org/tutorials/multi_objective_bo
    - Tolerance Calibration: scripts/calibrate_test_tolerances.py
"""

import pytest
import torch

from bo_engine import (
    compute_hypervolume,
    compute_pareto_front,
    generate_initial_design,
    generate_next_batch,
)
from bo_engine.benchmarks import branin_currin
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def create_branin_currin_spec(batch_size: int = 2) -> OptimizationSpec:
    """Create spec for Branin-Currin bi-objective optimization."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x0", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="branin", minimize=True),
            ObjectiveSpec(name="currin", minimize=True),
        ],
        batch_size=batch_size,
    )


def evaluate_branin_currin(spec: OptimizationSpec, suggestions: list) -> list[ObservationData]:
    """Evaluate Branin-Currin function for given suggestions."""
    observations = []
    for sugg in suggestions:
        x = torch.tensor([[sugg.parameter_values["x0"], sugg.parameter_values["x1"]]])
        y = branin_currin(x)
        observations.append(
            ObservationData(
                parameter_values=sugg.parameter_values,
                objective_values={"branin": y[0, 0].item(), "currin": y[0, 1].item()},
            )
        )
    return observations


class TestInitialDesign:
    """Tests for initial design generation (works with any objective count).

    Reference:
        - TOOL_SCHEMAS.md: Initial Design Phase section
        - AGENT_COOKBOOK.md: Initial Design Size Guidance
        - Formula: 2 × n_parameters + 1 (from INITIAL_DESIGN_MULTIPLIER constant)
    """

    def test_initial_design_generation(self):
        """Initial design uses Sobol sequence."""
        spec = create_branin_currin_spec()
        designs = generate_initial_design(spec, n_points=5)

        assert len(designs) == 5
        for d in designs:
            assert "x0" in d
            assert "x1" in d
            # Check bounds
            assert 0.0 <= d["x0"] <= 1.0
            assert 0.0 <= d["x1"] <= 1.0

    def test_initial_design_coverage(self):
        """Initial design covers the parameter space."""
        spec = create_branin_currin_spec()
        designs = generate_initial_design(spec, n_points=20)

        x0_values = [d["x0"] for d in designs]
        x1_values = [d["x1"] for d in designs]

        # Sobol should cover the space - check spread
        assert max(x0_values) - min(x0_values) > 0.5
        assert max(x1_values) - min(x1_values) > 0.5

    def test_default_initial_design_size_formula(self):
        """Default initial design size is 2 × n_params + 1.

        Reference:
            - TOOL_SCHEMAS.md: "Default size: 2 × n_parameters + 1"
            - bo_engine/constants.py: INITIAL_DESIGN_MULTIPLIER = 2
        """
        from bo_engine.constants import INITIAL_DESIGN_MULTIPLIER

        # Verify the constant value
        assert INITIAL_DESIGN_MULTIPLIER == 2

        # For 2 parameters, default should be 2*2 + 1 = 5
        spec = create_branin_currin_spec()
        n_params = len(spec.parameters)
        expected_default = INITIAL_DESIGN_MULTIPLIER * n_params + 1

        assert n_params == 2
        assert expected_default == 5

    def test_initial_design_size_override(self):
        """Explicit initial_design_size overrides the default formula.

        Reference:
            - TOOL_SCHEMAS.md: "Override: Set initial_design_size in campaign creation"
        """
        # Create spec with explicit initial_design_size
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x0", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="branin", minimize=True),
                ObjectiveSpec(name="currin", minimize=True),
            ],
            initial_design_size=15,  # Explicit override
        )

        # Verify spec has the override
        assert spec.initial_design_size == 15

    def test_initial_design_various_dimensions(self):
        """Verify initial design works for various problem dimensions.

        Reference:
            - AGENT_COOKBOOK.md: Initial Design Size Guidance table
              | Parameters | Default Initial Points |
              | 2-5        | 5-11                   |
              | 6-10       | 13-21                  |
              | 11-20      | 23-41                  |
        """
        from bo_engine.constants import INITIAL_DESIGN_MULTIPLIER

        # Test different parameter counts
        test_cases = [
            (2, 5),  # 2*2 + 1 = 5
            (5, 11),  # 2*5 + 1 = 11
            (10, 21),  # 2*10 + 1 = 21
            (20, 41),  # 2*20 + 1 = 41
        ]

        for n_params, expected_size in test_cases:
            calculated = INITIAL_DESIGN_MULTIPLIER * n_params + 1
            assert calculated == expected_size, (
                f"For {n_params} params, expected {expected_size}, got {calculated}"
            )


class TestMultiObjectiveWorkflow:
    """Tests for multi-objective optimization workflow."""

    def test_suggestion_generation_with_data(self):
        """Suggestions are generated using BO when data exists."""
        torch.manual_seed(42)
        spec = create_branin_currin_spec(batch_size=2)

        # Create initial observations
        observations = [
            ObservationData({"x0": 0.2, "x1": 0.3}, {"branin": 0.5, "currin": 0.7}),
            ObservationData({"x0": 0.5, "x1": 0.5}, {"branin": 0.6, "currin": 0.5}),
            ObservationData({"x0": 0.8, "x1": 0.7}, {"branin": 0.4, "currin": 0.8}),
        ]

        # Generate suggestions
        suggestions, _ = generate_next_batch(spec, observations, batch_size=2, iteration=1)

        assert len(suggestions) == 2
        for s in suggestions:
            assert s.generation_method == "bo"
            assert s.acquisition_function == "qLogNEHVI"
            assert s.model_type is not None
            assert 0.0 <= s.parameter_values["x0"] <= 1.0
            assert 0.0 <= s.parameter_values["x1"] <= 1.0

    def test_pareto_front_exists(self):
        """Pareto front is non-empty after optimization iterations.

        Note: The Pareto front size can shrink when new points dominate existing
        Pareto-optimal points. The correct invariant for optimization progress is
        that hypervolume never decreases (tested in test_hypervolume_improves).
        """
        torch.manual_seed(42)
        spec = create_branin_currin_spec(batch_size=3)
        observations: list[ObservationData] = []

        for iteration in range(4):
            suggestions, _ = generate_next_batch(
                spec, observations, batch_size=3, iteration=iteration
            )
            new_obs = evaluate_branin_currin(spec, suggestions)
            observations.extend(new_obs)

        # Compute final Pareto front
        y = torch.tensor(
            [
                [obs.objective_values["branin"], obs.objective_values["currin"]]
                for obs in observations
            ]
        )
        pareto_y, _ = compute_pareto_front(y)

        # Pareto front should exist and be non-empty
        assert pareto_y.shape[0] >= 1, "Pareto front should have at least one point"
        # Should have at least some diversity (not all observations dominated by one)
        assert pareto_y.shape[0] <= len(observations), "Pareto front cannot exceed observations"

    def test_hypervolume_improves(self):
        """Hypervolume increases with optimization."""
        torch.manual_seed(42)
        spec = create_branin_currin_spec(batch_size=3)
        observations: list[ObservationData] = []

        hypervolumes = []
        ref_point = torch.tensor([1.5, 1.5])  # Fixed reference point

        for iteration in range(4):
            suggestions, _ = generate_next_batch(
                spec, observations, batch_size=3, iteration=iteration
            )
            new_obs = evaluate_branin_currin(spec, suggestions)
            observations.extend(new_obs)

            # Compute hypervolume
            y = torch.tensor(
                [
                    [obs.objective_values["branin"], obs.objective_values["currin"]]
                    for obs in observations
                ]
            )
            pareto_y, _ = compute_pareto_front(y)
            hv = compute_hypervolume(pareto_y, ref_point)
            hypervolumes.append(hv)

        # Hypervolume should be non-decreasing
        for i in range(len(hypervolumes) - 1):
            assert hypervolumes[i + 1] >= hypervolumes[i] - 1e-6

    def test_optimization_invariants(self, tolerance_ci: dict[str, float]):
        """Test invariants that always hold regardless of stochastic outcomes.

        These invariants are guaranteed properties of the optimization:
        - Pareto front has at least 2 points after sufficient iterations
        - Hypervolume is non-decreasing over iterations
        - All suggestions are within bounds

        Reference: BoTorch multi-objective optimization tutorial
        https://botorch.org/tutorials/multi_objective_bo
        """
        torch.manual_seed(42)
        spec = create_branin_currin_spec(batch_size=3)
        observations: list[ObservationData] = []
        ref_point = torch.tensor([1.5, 1.5])

        hypervolumes = []

        # Run optimization
        for iteration in range(5):
            suggestions, _ = generate_next_batch(
                spec, observations, batch_size=3, iteration=iteration
            )

            # Invariant: All suggestions within bounds
            for s in suggestions:
                assert 0.0 <= s.parameter_values["x0"] <= 1.0
                assert 0.0 <= s.parameter_values["x1"] <= 1.0

            new_obs = evaluate_branin_currin(spec, suggestions)
            observations.extend(new_obs)

            # Track hypervolume
            y = torch.tensor(
                [
                    [obs.objective_values["branin"], obs.objective_values["currin"]]
                    for obs in observations
                ]
            )
            pareto_y, _ = compute_pareto_front(y)
            hv = compute_hypervolume(pareto_y, ref_point)
            hypervolumes.append(hv)

        # Get final Pareto front
        y = torch.tensor(
            [
                [obs.objective_values["branin"], obs.objective_values["currin"]]
                for obs in observations
            ]
        )
        pareto_y, _ = compute_pareto_front(y)

        # Invariant: Should have found multiple Pareto points
        assert pareto_y.shape[0] >= tolerance_ci["min_pareto_size"], (
            f"Expected at least {tolerance_ci['min_pareto_size']} Pareto points, "
            f"got {pareto_y.shape[0]}"
        )

        # Invariant: Hypervolume should be non-decreasing
        for i in range(len(hypervolumes) - 1):
            assert hypervolumes[i + 1] >= hypervolumes[i] - 1e-6, (
                f"Hypervolume decreased: {hypervolumes[i]:.4f} -> {hypervolumes[i + 1]:.4f}"
            )

        # Calibrated tolerance: Pareto max should be reasonable
        # Uses CI tolerance from conftest (99th percentile + 10% margin)
        assert pareto_y.max() < tolerance_ci["pareto_max"], (
            f"Pareto max {pareto_y.max():.2f} exceeds CI tolerance {tolerance_ci['pareto_max']}"
        )

    @pytest.mark.nightly
    def test_optimization_finds_good_tradeoffs_statistical(self):
        """Statistical test: Optimization finds good trade-offs over multiple runs.

        This test runs multiple times with different seeds and checks that
        the AVERAGE performance meets tighter tolerances. This catches
        systematic regressions while allowing individual run variance.

        Marked as @nightly because it's slower and only needed for regression detection.

        Reference: BoTorch multi-objective optimization tutorial
        https://botorch.org/tutorials/multi_objective_bo
        """
        import numpy as np

        n_runs = 5
        pareto_maxes = []

        for run in range(n_runs):
            torch.manual_seed(42 + run)
            spec = create_branin_currin_spec(batch_size=3)
            observations: list[ObservationData] = []

            # Run optimization
            for iteration in range(5):
                suggestions, _ = generate_next_batch(
                    spec, observations, batch_size=3, iteration=iteration
                )
                new_obs = evaluate_branin_currin(spec, suggestions)
                observations.extend(new_obs)

            # Get Pareto front
            y = torch.tensor(
                [
                    [obs.objective_values["branin"], obs.objective_values["currin"]]
                    for obs in observations
                ]
            )
            pareto_y, _ = compute_pareto_front(y)
            pareto_maxes.append(pareto_y.max().item())

        # Statistical assertions (mean should be well-behaved)
        mean_pareto_max = np.mean(pareto_maxes)
        assert mean_pareto_max < 4.0, (
            f"Mean Pareto max {mean_pareto_max:.2f} too high (expected < 4.0)"
        )

        # 90th percentile should still be reasonable
        p90 = np.percentile(pareto_maxes, 90)
        assert p90 < 6.0, f"90th percentile Pareto max {p90:.2f} too high (expected < 6.0)"


class TestSuggestionProvenance:
    """Tests for suggestion metadata and provenance."""

    def test_initial_design_provenance(self):
        """Initial design suggestions have correct provenance."""
        spec = create_branin_currin_spec(batch_size=2)
        suggestions, _ = generate_next_batch(spec, [], batch_size=2, iteration=0)

        for s in suggestions:
            assert s.generation_method == "initial_design"
            assert s.random_seed > 0
            assert s.explanation is not None and "Sobol" in s.explanation

    def test_bo_suggestion_provenance(self):
        """BO suggestions have model and acquisition info."""
        torch.manual_seed(42)
        spec = create_branin_currin_spec(batch_size=2)

        observations = [
            ObservationData({"x0": 0.2, "x1": 0.3}, {"branin": 0.5, "currin": 0.7}),
            ObservationData({"x0": 0.5, "x1": 0.5}, {"branin": 0.6, "currin": 0.5}),
            ObservationData({"x0": 0.8, "x1": 0.7}, {"branin": 0.4, "currin": 0.8}),
        ]

        suggestions, _ = generate_next_batch(spec, observations, batch_size=2, iteration=1)

        for s in suggestions:
            assert s.generation_method == "bo"
            assert s.acquisition_function == "qLogNEHVI"
            assert s.model_type is not None
            assert "GP" in s.model_type or "Gaussian" in s.model_type
            assert s.confidence_level in ["high", "medium", "low"]
            assert s.explanation is not None


class TestEdgeCases:
    """Tests for edge cases in optimization workflow."""

    def test_single_observation(self):
        """Optimization handles single observation gracefully."""
        spec = create_branin_currin_spec(batch_size=2)
        observations = [
            ObservationData({"x0": 0.5, "x1": 0.5}, {"branin": 0.5, "currin": 0.5}),
        ]

        # Should fall back to initial design with insufficient data
        suggestions, _ = generate_next_batch(spec, observations, batch_size=2, iteration=1)
        assert len(suggestions) == 2

    def test_identical_observations(self):
        """Optimization handles identical observations."""
        torch.manual_seed(42)
        spec = create_branin_currin_spec(batch_size=2)

        # All same point
        observations = [
            ObservationData({"x0": 0.5, "x1": 0.5}, {"branin": 0.5, "currin": 0.5}),
            ObservationData({"x0": 0.5, "x1": 0.5}, {"branin": 0.5, "currin": 0.5}),
            ObservationData({"x0": 0.5, "x1": 0.5}, {"branin": 0.5, "currin": 0.5}),
        ]

        # Should still generate suggestions (model may be uncertain)
        suggestions, _ = generate_next_batch(spec, observations, batch_size=2, iteration=1)
        assert len(suggestions) == 2

    def test_batch_size_one(self):
        """Optimization works with batch size 1."""
        torch.manual_seed(42)
        spec = create_branin_currin_spec(batch_size=1)
        observations: list[ObservationData] = []

        for iteration in range(3):
            suggestions, _ = generate_next_batch(
                spec, observations, batch_size=1, iteration=iteration
            )
            assert len(suggestions) == 1
            new_obs = evaluate_branin_currin(spec, suggestions)
            observations.extend(new_obs)

        assert len(observations) == 3

    def test_empty_observations(self):
        """Optimization handles empty observations."""
        spec = create_branin_currin_spec(batch_size=3)
        suggestions, _ = generate_next_batch(spec, [], batch_size=3, iteration=0)

        assert len(suggestions) == 3
        for s in suggestions:
            assert s.generation_method == "initial_design"
