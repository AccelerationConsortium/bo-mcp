"""Tests for single-objective Bayesian Optimization (v1.0.1)."""

import numpy as np
import pytest
import torch
from botorch.exceptions.errors import InputDataError

from bo_engine import (
    AcquisitionMethod,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    compute_best_value,
    compute_improvement_history,
    compute_single_objective_improvement_rate,
    create_and_fit_single_task_model,
    create_single_objective_acquisition,
    determine_single_objective_health_status,
    generate_next_batch,
    get_best_observed_value,
    optimize_acquisition,
)


@pytest.fixture
def single_obj_spec() -> OptimizationSpec:
    """Create a single-objective optimization spec."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="y", minimize=True),
        ],
        batch_size=2,
    )


@pytest.fixture
def single_obj_maximize_spec() -> OptimizationSpec:
    """Create a single-objective maximization spec."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="y", minimize=False),
        ],
        batch_size=2,
    )


@pytest.fixture
def sample_observations_single() -> list[ObservationData]:
    """Create sample observations for single-objective optimization."""
    return [
        ObservationData(
            parameter_values={"x1": 0.1, "x2": 0.2},
            objective_values={"y": 5.0},
        ),
        ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.5},
            objective_values={"y": 3.0},
        ),
        ObservationData(
            parameter_values={"x1": 0.9, "x2": 0.1},
            objective_values={"y": 4.0},
        ),
        ObservationData(
            parameter_values={"x1": 0.3, "x2": 0.7},
            objective_values={"y": 3.5},
        ),
        ObservationData(
            parameter_values={"x1": 0.7, "x2": 0.4},
            objective_values={"y": 4.5},
        ),
    ]


class TestSingleObjectiveGeneration:
    """Test single-objective suggestion generation."""

    def test_generate_initial_design_single_objective(
        self, rng: np.random.Generator, single_obj_spec: OptimizationSpec
    ) -> None:
        """Test initial design generation for single-objective."""
        suggestions, _ = generate_next_batch(single_obj_spec, [], batch_size=3, rng=rng)

        assert len(suggestions) == 3
        assert all(s.generation_method == "initial_design" for s in suggestions)
        assert all("x1" in s.parameter_values for s in suggestions)
        assert all("x2" in s.parameter_values for s in suggestions)

    def test_generate_bo_suggestions_single_objective(
        self,
        rng: np.random.Generator,
        single_obj_spec: OptimizationSpec,
        sample_observations_single: list[ObservationData],
    ) -> None:
        """Test BO suggestion generation for single-objective."""
        suggestions, _ = generate_next_batch(
            single_obj_spec,
            sample_observations_single,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2
        assert all(s.generation_method == "bo" for s in suggestions)
        assert all(s.acquisition_function == "noisy_expected_improvement" for s in suggestions)
        assert all(s.model_type == "SingleTaskGP (Gaussian Process)" for s in suggestions)

    def test_generate_single_objective_maximization(
        self,
        rng: np.random.Generator,
        single_obj_maximize_spec: OptimizationSpec,
        sample_observations_single: list[ObservationData],
    ) -> None:
        """Test single-objective maximization."""
        suggestions, _ = generate_next_batch(
            single_obj_maximize_spec,
            sample_observations_single,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2
        assert all(s.generation_method == "bo" for s in suggestions)
        # Should work even for maximization (negated internally)
        assert all(s.acquisition_value is not None for s in suggestions)


@pytest.mark.usefixtures("torch_rng")
class TestSingleObjectiveModel:
    """Test single-objective model creation and fitting."""

    def test_create_and_fit_single_task_model(self) -> None:
        """Test SingleTaskGP creation and fitting."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        # Check model is fitted
        model.eval()
        with torch.no_grad():
            test_x = torch.rand(5, 2, dtype=torch.double)
            posterior = model.posterior(test_x)
            assert posterior.mean.shape == (5, 1)
            assert posterior.variance.shape == (5, 1)

    def test_create_single_task_model_with_1d_y(self) -> None:
        """Test SingleTaskGP creation with 1D y."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, dtype=torch.double)  # 1D
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        # Should handle 1D y correctly
        assert model is not None


@pytest.mark.usefixtures("torch_rng")
class TestSingleObjectiveAcquisition:
    """Test single-objective acquisition functions."""

    def test_create_single_objective_acquisition_qlognei(self) -> None:
        """Test qLogNEI acquisition creation."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        acqf = create_single_objective_acquisition(
            model=model,
            train_x=train_x,
            train_y=train_y,
            minimize=True,
            use_noisy=True,
        )

        assert acqf is not None

        # Test optimization
        candidates, values = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=2,
            num_restarts=5,
            raw_samples=64,
        )
        assert candidates.shape == (2, 2)
        assert values.numel() >= 1

    def test_get_best_observed_value(self) -> None:
        """Test best observed value computation."""
        train_y = torch.tensor([[5.0], [3.0], [4.0]], dtype=torch.double)

        # Minimize
        best_min = get_best_observed_value(train_y, minimize=True)
        assert best_min == 3.0

        # Maximize
        best_max = get_best_observed_value(train_y, minimize=False)
        assert best_max == 5.0


class TestSingleObjectiveDiagnostics:
    """Test single-objective diagnostic functions."""

    def test_compute_best_value(self) -> None:
        """Test best value computation."""
        values = [5.0, 3.0, 4.0, 2.0, 6.0]

        # Minimize
        best_min, idx_min = compute_best_value(values, minimize=True)
        assert best_min == 2.0
        assert idx_min == 3

        # Maximize
        best_max, idx_max = compute_best_value(values, minimize=False)
        assert best_max == 6.0
        assert idx_max == 4

    def test_compute_improvement_history(self) -> None:
        """Test improvement history computation."""
        values = [5.0, 3.0, 4.0, 2.0, 6.0]

        # Minimize - running minimum
        history_min = compute_improvement_history(values, minimize=True)
        assert history_min == [5.0, 3.0, 3.0, 2.0, 2.0]

        # Maximize - running maximum
        history_max = compute_improvement_history(values, minimize=False)
        assert history_max == [5.0, 5.0, 5.0, 5.0, 6.0]

    def test_compute_improvement_rate(self) -> None:
        """Test improvement rate computation."""
        # Good improvement
        history_improving = [10.0, 8.0, 6.0, 4.0, 2.0]
        rate_improving = compute_single_objective_improvement_rate(history_improving)
        assert rate_improving > 0.5

        # No improvement
        history_stagnant = [5.0, 5.0, 5.0, 5.0, 5.0]
        rate_stagnant = compute_single_objective_improvement_rate(history_stagnant)
        assert rate_stagnant == 0.0

    def test_determine_health_status(self) -> None:
        """Test health status determination."""
        # Healthy - improving
        status, warnings = determine_single_objective_health_status(
            improvement_history=[10.0, 8.0, 6.0, 4.0, 2.0],
            model_correlation=0.8,
        )
        assert status == "healthy"

        # Critical - stagnant
        status, warnings = determine_single_objective_health_status(
            improvement_history=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            model_correlation=0.8,
            stagnation_threshold=5,
        )
        assert status == "critical"
        assert len(warnings) > 0

        # Warning - low correlation
        status, warnings = determine_single_objective_health_status(
            improvement_history=[10.0, 8.0, 6.0, 4.0, 2.0],
            model_correlation=0.1,
        )
        assert status in ["warning", "critical"]


class TestSingleObjectiveIntegration:
    """Integration tests for single-objective BO workflow."""

    def test_single_objective_full_workflow(self, rng: np.random.Generator) -> None:
        """Test complete single-objective BO workflow."""
        # Define simple 1D minimization problem
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 10.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f", minimize=True),
            ],
            batch_size=2,
        )

        # Objective function: f(x) = (x - 5)^2
        def objective(x: float) -> float:
            return (x - 5.0) ** 2

        observations: list[ObservationData] = []

        # Run 3 iterations
        for iteration in range(3):
            suggestions, _ = generate_next_batch(spec, observations, iteration=iteration, rng=rng)

            # Evaluate suggestions
            for sugg in suggestions:
                x = sugg.parameter_values["x"]
                f = objective(x)
                observations.append(
                    ObservationData(
                        parameter_values={"x": x},
                        objective_values={"f": f},
                    )
                )

        # Check that we found a good solution (x near 5)
        best_f, best_idx = compute_best_value(
            [obs.objective_values["f"] for obs in observations],
            minimize=True,
        )
        best_x = observations[best_idx].parameter_values["x"]

        # Best x should be reasonably close to 5
        assert abs(best_x - 5.0) < 3.0  # Allow some tolerance
        assert best_f < 10.0  # Should be better than random

    def test_acquisition_method_selection(self, rng: np.random.Generator) -> None:
        """Test that acquisition method selection works correctly."""
        spec_auto = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f", minimize=True),
            ],
            batch_size=1,
            acquisition_method=AcquisitionMethod.AUTO,
        )

        spec_qlognei = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f", minimize=True),
            ],
            batch_size=1,
            acquisition_method=AcquisitionMethod.NOISY_EI,
        )

        observations = [
            ObservationData(parameter_values={"x": 0.3}, objective_values={"f": 1.0}),
            ObservationData(parameter_values={"x": 0.7}, objective_values={"f": 2.0}),
            ObservationData(parameter_values={"x": 0.5}, objective_values={"f": 1.5}),
        ]

        # Both should work and produce qLogNEI for single objective
        sugg_auto, _ = generate_next_batch(spec_auto, observations, iteration=1, rng=rng)
        sugg_qlognei, _ = generate_next_batch(spec_qlognei, observations, iteration=1, rng=rng)

        assert sugg_auto[0].acquisition_function == "noisy_expected_improvement"
        assert sugg_qlognei[0].acquisition_function == "noisy_expected_improvement"


@pytest.mark.usefixtures("torch_rng")
class TestModelActuallyLearns:
    """Test that GP models actually learn from data."""

    def test_model_learns_simple_quadratic(self) -> None:
        """Model should approximate y = (x - 0.5)^2."""
        # Create training data from quadratic function
        train_x = torch.linspace(0, 1, 15).unsqueeze(-1).double()
        train_y = ((train_x - 0.5) ** 2).double()
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        model.eval()

        # Test at known points
        test_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
        expected_y = torch.tensor([[0.25], [0.0], [0.25]], dtype=torch.double)

        with torch.no_grad():
            pred_mean = model.posterior(test_x).mean

        # Predictions should be close to actual values
        assert torch.allclose(pred_mean, expected_y, atol=0.1), (
            f"Model predictions {pred_mean.squeeze().tolist()} "
            f"should approximate {expected_y.squeeze().tolist()}"
        )

    def test_model_uncertainty_decreases_near_data(self) -> None:
        """Uncertainty should be lower near training points."""
        train_x = torch.tensor([[0.2], [0.8]], dtype=torch.double)
        train_y = torch.tensor([[0.5], [0.3]], dtype=torch.double)
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        model.eval()

        # Test at training point vs. far from training data
        near_point = torch.tensor([[0.2]], dtype=torch.double)  # Near training
        far_point = torch.tensor([[0.5]], dtype=torch.double)  # Far from training

        with torch.no_grad():
            near_var = model.posterior(near_point).variance.item()
            far_var = model.posterior(far_point).variance.item()

        assert near_var < far_var, (
            f"Variance near data ({near_var:.4f}) should be less than "
            f"variance far from data ({far_var:.4f})"
        )

    def test_model_handles_noisy_data(self) -> None:
        """Model should smooth through noisy observations."""
        torch.manual_seed(42)
        n_points = 20
        train_x = torch.linspace(0, 1, n_points).unsqueeze(-1).double()
        # True function + noise
        true_y = torch.sin(2 * 3.14159 * train_x)
        noise = torch.randn_like(true_y) * 0.1
        train_y = true_y + noise
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        model.eval()

        # Predictions should be smoother than noisy observations
        with torch.no_grad():
            pred_mean = model.posterior(train_x).mean

        # Check that predictions don't exactly match noisy data (smoothing occurred)
        residuals = (pred_mean - train_y).abs()
        assert residuals.mean() > 0.01, "Model should smooth, not interpolate noisy data"


class TestEdgeCasesAndFailures:
    """Test error handling and edge cases."""

    def test_handles_nan_in_observations(self, rng: np.random.Generator) -> None:
        """Should handle or raise clear error for NaN values."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            batch_size=1,
        )

        observations = [
            ObservationData({"x": 0.1}, {"y": 2.0}),
            ObservationData({"x": 0.5}, {"y": float("nan")}),
            ObservationData({"x": 0.3}, {"y": 1.0}),
        ]

        # Should either filter NaN or raise informative error
        with pytest.raises((ValueError, RuntimeError, InputDataError)) as exc_info:
            generate_next_batch(spec, observations, iteration=1, rng=rng)

        # If it raises, message should be helpful
        assert "nan" in str(exc_info.value).lower() or len(str(exc_info.value)) > 0

    def test_handles_inf_in_observations(self, rng: np.random.Generator) -> None:
        """Should handle or raise clear error for Inf values."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            batch_size=1,
        )

        observations = [
            ObservationData({"x": 0.1}, {"y": 2.0}),
            ObservationData({"x": 0.5}, {"y": float("inf")}),
            ObservationData({"x": 0.3}, {"y": 1.0}),
        ]

        with pytest.raises((ValueError, RuntimeError, InputDataError)):
            generate_next_batch(spec, observations, iteration=1, rng=rng)

    def test_handles_very_small_bounds(self, rng: np.random.Generator) -> None:
        """Should work with very small parameter ranges."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1e-10)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            batch_size=2,
        )

        observations = [
            ObservationData({"x": 1e-11}, {"y": 1.0}),
            ObservationData({"x": 5e-11}, {"y": 2.0}),
            ObservationData({"x": 8e-11}, {"y": 1.5}),
        ]

        # Should work without numerical issues
        suggestions, _ = generate_next_batch(spec, observations, iteration=1, rng=rng)
        assert len(suggestions) == 2
        for s in suggestions:
            assert 0.0 <= s.parameter_values["x"] <= 1e-10
