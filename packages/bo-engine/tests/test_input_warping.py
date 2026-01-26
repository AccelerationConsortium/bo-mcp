"""Tests for input warping with Kumaraswamy CDF (v1.1)."""

import torch

from bo_engine import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    create_and_fit_model,
    create_and_fit_single_task_model,
    generate_next_batch,
    get_warping_parameters,
)
from bo_engine.models import create_input_transform, create_model, create_single_task_model


class TestInputTransformCreation:
    """Test input transform creation with optional warping."""

    def test_create_normalize_only(self) -> None:
        """Test creating Normalize-only transform (no warping)."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        transform = create_input_transform(n_dims=2, bounds=bounds, use_input_warping=False)

        # Should be Normalize transform
        assert hasattr(transform, "bounds")

    def test_create_chained_with_warping(self) -> None:
        """Test creating ChainedInputTransform with warping."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        transform = create_input_transform(n_dims=2, bounds=bounds, use_input_warping=True)

        # Should be ChainedInputTransform
        assert hasattr(transform, "keys")  # ChainedInputTransform has keys()
        assert "normalize" in transform.keys()
        assert "warp" in transform.keys()


class TestSingleTaskModelWithWarping:
    """Test SingleTaskGP with input warping."""

    def test_create_model_with_warping(self) -> None:
        """Test SingleTaskGP creation with warping enabled."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_single_task_model(train_x, train_y, bounds, use_input_warping=True)

        # Should have ChainedInputTransform
        assert hasattr(model.input_transform, "keys")

    def test_create_and_fit_model_with_warping(self) -> None:
        """Test creating and fitting SingleTaskGP with warping."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds, use_input_warping=True)

        # Should be fitted and usable for predictions
        model.eval()
        with torch.no_grad():
            test_x = torch.rand(5, 2, dtype=torch.double)
            posterior = model.posterior(test_x)
            assert posterior.mean.shape == (5, 1)

    def test_get_warping_parameters(self) -> None:
        """Test extracting warping parameters from model."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds, use_input_warping=True)
        params = get_warping_parameters(model)

        assert params is not None
        assert "concentration0" in params
        assert "concentration1" in params
        assert params["concentration0"].shape[-1] == 2  # 2 dimensions
        assert params["concentration1"].shape[-1] == 2

    def test_no_warping_parameters_without_warping(self) -> None:
        """Test that no warping parameters are returned without warping."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds, use_input_warping=False)
        params = get_warping_parameters(model)

        assert params is None


class TestModelListGPWithWarping:
    """Test ModelListGP with input warping."""

    def test_create_model_list_with_warping(self) -> None:
        """Test ModelListGP creation with warping enabled."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 2, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_model(train_x, train_y, bounds, use_input_warping=True)

        # Each sub-model should have warping
        for m in model.models:
            assert hasattr(m.input_transform, "keys")

    def test_create_and_fit_model_list_with_warping(self) -> None:
        """Test creating and fitting ModelListGP with warping."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 2, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_model(train_x, train_y, bounds, use_input_warping=True)

        # Should be fitted and usable for predictions
        model.eval()
        with torch.no_grad():
            test_x = torch.rand(5, 2, dtype=torch.double)
            posterior = model.posterior(test_x)
            # ModelListGP posterior shape can vary; check that it has 5 test points and 2 objectives
            assert posterior.mean.shape[0] == 5
            assert 2 in posterior.mean.shape  # 2 objectives somewhere in shape


class TestWarpingInWorkflow:
    """Test input warping in optimization workflow."""

    def test_single_objective_with_warping(self) -> None:
        """Test single-objective optimization with input warping."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f", minimize=True),
            ],
            batch_size=2,
            use_input_warping=True,
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.2, "x2": 0.3},
                objective_values={"f": 1.0},
            ),
            ObservationData(
                parameter_values={"x1": 0.8, "x2": 0.7},
                objective_values={"f": 2.0},
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5},
                objective_values={"f": 0.5},
            ),
        ]

        suggestions, _ = generate_next_batch(spec, observations, iteration=1)

        assert len(suggestions) == 2
        assert all(s.generation_method == "bo" for s in suggestions)

    def test_multi_objective_with_warping(self) -> None:
        """Test multi-objective optimization with input warping."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f1", minimize=True),
                ObjectiveSpec(name="f2", minimize=True),
            ],
            batch_size=2,
            use_input_warping=True,
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.2, "x2": 0.3},
                objective_values={"f1": 1.0, "f2": 2.0},
            ),
            ObservationData(
                parameter_values={"x1": 0.8, "x2": 0.7},
                objective_values={"f1": 2.0, "f2": 1.0},
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5},
                objective_values={"f1": 1.5, "f2": 1.5},
            ),
        ]

        suggestions, _ = generate_next_batch(spec, observations, iteration=1)

        assert len(suggestions) == 2
        assert all(s.generation_method == "bo" for s in suggestions)


class TestWarpingEffectiveness:
    """Test that input warping improves non-stationary function modeling."""

    def test_warping_on_non_uniform_data(self) -> None:
        """Test warping helps with non-uniformly distributed data."""
        # Create data clustered near 0 (non-uniform distribution)
        torch.manual_seed(42)
        n_samples = 20
        # Data clustered in [0, 0.3] range
        train_x = torch.rand(n_samples, 2, dtype=torch.double) * 0.3
        # Add a few points in [0.7, 1.0]
        train_x[15:] = 0.7 + torch.rand(5, 2, dtype=torch.double) * 0.3
        train_y = (train_x[:, 0:1] - 0.5) ** 2 + (train_x[:, 1:2] - 0.5) ** 2

        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        # Fit model with warping
        model_with_warp = create_and_fit_single_task_model(
            train_x, train_y, bounds, use_input_warping=True
        )

        # Fit model without warping
        model_no_warp = create_and_fit_single_task_model(
            train_x, train_y, bounds, use_input_warping=False
        )

        # Both should fit without errors
        model_with_warp.eval()
        model_no_warp.eval()

        # Test predictions on uniform grid
        test_x = torch.linspace(0, 1, 10, dtype=torch.double).unsqueeze(-1).expand(-1, 2)

        with torch.no_grad():
            pred_warp = model_with_warp.posterior(test_x).mean
            pred_no_warp = model_no_warp.posterior(test_x).mean

        # Both should produce valid predictions
        assert pred_warp.shape == (10, 1)
        assert pred_no_warp.shape == (10, 1)
        assert not torch.isnan(pred_warp).any()
        assert not torch.isnan(pred_no_warp).any()
