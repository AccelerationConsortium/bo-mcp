"""Unit tests for discrete and mixed acquisition optimization.

Tests the _optimize_discrete and _optimize_mixed paths in acquisition.py,
verifying that BoTorch's optimize_acqf_discrete and optimize_acqf_mixed
are correctly integrated.

References:
    - optimize_acqf_discrete:
      https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf_discrete
    - optimize_acqf_mixed:
      https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf_mixed
"""

import pytest
import torch

from bo_engine.acquisition import optimize_acquisition
from bo_engine.constants import MIXED_CATEGORICAL_COMBO_THRESHOLD
from bo_engine.models import create_and_fit_single_task_model
from bo_engine.transforms import (
    encode_categorical,
    enumerate_discrete_choices,
    get_bounds_tensor,
    get_n_dims,
)
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def categorical_spec_small() -> OptimizationSpec:
    """Small purely categorical spec: 2 params x 3 categories = 9 combos."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="color", type=ParameterType.CATEGORICAL, categories=["red", "green", "blue"]
            ),
            ParameterSpec(
                name="shape",
                type=ParameterType.CATEGORICAL,
                categories=["circle", "square", "triangle"],
            ),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        batch_size=2,
    )


@pytest.fixture
def mixed_spec_small() -> OptimizationSpec:
    """Small mixed spec: 1 continuous + 1 categorical (3 categories)."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(
                name="color", type=ParameterType.CATEGORICAL, categories=["red", "green", "blue"]
            ),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        batch_size=2,
    )


def _create_categorical_training_data(
    spec: OptimizationSpec,
    n_obs: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic training data for a categorical spec.

    Picks the first n_obs combinations from the enumeration and assigns
    synthetic objective values.
    """
    choices = enumerate_discrete_choices(spec)
    train_x = choices[:n_obs]
    # Synthetic objectives: sum of encoded values with some variation
    train_y = train_x.sum(dim=1, keepdim=True) + torch.randn(n_obs, 1, dtype=train_x.dtype) * 0.1
    return train_x, train_y


def _create_mixed_training_data(
    spec: OptimizationSpec,
    n_obs: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic training data for a mixed spec."""
    torch.manual_seed(42)
    values_list = [
        {"x1": 0.2, "color": "red"},
        {"x1": 0.5, "color": "green"},
        {"x1": 0.8, "color": "blue"},
        {"x1": 0.3, "color": "red"},
        {"x1": 0.7, "color": "green"},
    ]
    train_x = torch.stack([encode_categorical(v, spec) for v in values_list[:n_obs]])
    train_y = torch.randn(n_obs, 1, dtype=train_x.dtype)
    return train_x, train_y


# =============================================================================
# TestOptimizeDiscrete
# =============================================================================


class TestOptimizeDiscrete:
    """Test discrete acquisition optimization via optimize_acqf_discrete."""

    def test_unique_candidates_returned(self, categorical_spec_small: OptimizationSpec) -> None:
        """Batch of 2 from 9 choices should return 2 unique candidates.

        This is the core fix: optimize_acqf_discrete with unique=True ensures
        no duplicate suggestions within a batch.
        """
        torch.manual_seed(42)
        spec = categorical_spec_small
        train_x, train_y = _create_categorical_training_data(spec, n_obs=4)
        bounds = get_bounds_tensor(spec)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        from bo_engine.acquisition import create_single_objective_acquisition

        acqf = create_single_objective_acquisition(model=model, train_x=train_x, train_y=train_y)

        candidates, acq_values = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=2,
            spec=spec,
        )

        assert candidates.shape[0] == 2
        # Candidates must be different
        assert not torch.allclose(candidates[0], candidates[1])

    def test_x_avoid_is_respected(self, categorical_spec_small: OptimizationSpec) -> None:
        """X_avoid should exclude specified points from the choice set."""
        torch.manual_seed(42)
        spec = categorical_spec_small
        train_x, train_y = _create_categorical_training_data(spec, n_obs=4)
        bounds = get_bounds_tensor(spec)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        from bo_engine.acquisition import create_single_objective_acquisition

        acqf = create_single_objective_acquisition(model=model, train_x=train_x, train_y=train_y)

        candidates, _ = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=2,
            spec=spec,
            x_avoid=train_x,
        )

        # Candidates should not be any of the training points
        for i in range(candidates.shape[0]):
            for j in range(train_x.shape[0]):
                if torch.allclose(candidates[i], train_x[j], atol=1e-6):
                    pytest.fail(
                        f"Candidate {i} matches training point {j}, "
                        "but X_avoid should have excluded it."
                    )

    def test_batch_size_one_works(self, categorical_spec_small: OptimizationSpec) -> None:
        """batch_size=1 should return a single candidate."""
        torch.manual_seed(42)
        spec = categorical_spec_small
        train_x, train_y = _create_categorical_training_data(spec, n_obs=4)
        bounds = get_bounds_tensor(spec)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        from bo_engine.acquisition import create_single_objective_acquisition

        acqf = create_single_objective_acquisition(model=model, train_x=train_x, train_y=train_y)

        candidates, _ = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=1,
            spec=spec,
        )

        assert candidates.shape == (1, get_n_dims(spec))

    def test_candidates_are_valid_one_hot(self, categorical_spec_small: OptimizationSpec) -> None:
        """Each candidate should be a valid one-hot encoding."""
        torch.manual_seed(42)
        spec = categorical_spec_small
        train_x, train_y = _create_categorical_training_data(spec, n_obs=4)
        bounds = get_bounds_tensor(spec)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        from bo_engine.acquisition import create_single_objective_acquisition

        acqf = create_single_objective_acquisition(model=model, train_x=train_x, train_y=train_y)

        candidates, _ = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=2,
            spec=spec,
        )

        # Each candidate should have valid one-hot blocks
        # color: dims 0-2, shape: dims 3-5
        for i in range(candidates.shape[0]):
            color_block = candidates[i, :3]
            shape_block = candidates[i, 3:]

            assert torch.isclose(color_block.sum(), torch.tensor(1.0, dtype=color_block.dtype))
            assert torch.isclose(shape_block.sum(), torch.tensor(1.0, dtype=shape_block.dtype))
            assert ((color_block == 0.0) | (color_block == 1.0)).all()
            assert ((shape_block == 0.0) | (shape_block == 1.0)).all()


# =============================================================================
# TestOptimizeMixed
# =============================================================================


class TestOptimizeMixed:
    """Test mixed acquisition optimization via optimize_acqf_mixed."""

    def test_correct_output_shape(self, mixed_spec_small: OptimizationSpec) -> None:
        """Output should have shape (batch_size, n_dims)."""
        torch.manual_seed(42)
        spec = mixed_spec_small
        train_x, train_y = _create_mixed_training_data(spec, n_obs=5)
        bounds = get_bounds_tensor(spec)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        from bo_engine.acquisition import create_single_objective_acquisition

        acqf = create_single_objective_acquisition(model=model, train_x=train_x, train_y=train_y)

        candidates, _ = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=2,
            spec=spec,
            num_restarts=5,
            raw_samples=64,
        )

        n_dims = get_n_dims(spec)  # 1 continuous + 3 one-hot = 4
        assert candidates.shape == (2, n_dims)

    def test_categorical_dims_are_valid_one_hot(self, mixed_spec_small: OptimizationSpec) -> None:
        """Categorical dimensions should be valid one-hot encodings."""
        torch.manual_seed(42)
        spec = mixed_spec_small
        train_x, train_y = _create_mixed_training_data(spec, n_obs=5)
        bounds = get_bounds_tensor(spec)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        from bo_engine.acquisition import create_single_objective_acquisition

        acqf = create_single_objective_acquisition(model=model, train_x=train_x, train_y=train_y)

        candidates, _ = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=2,
            spec=spec,
            num_restarts=5,
            raw_samples=64,
        )

        # Categorical block: dims 1-3 (3 categories)
        for i in range(candidates.shape[0]):
            cat_block = candidates[i, 1:]
            assert torch.isclose(cat_block.sum(), torch.tensor(1.0, dtype=cat_block.dtype))
            assert ((cat_block == 0.0) | (cat_block == 1.0)).all()

    def test_continuous_dims_within_bounds(self, mixed_spec_small: OptimizationSpec) -> None:
        """Continuous dimensions should be within bounds."""
        torch.manual_seed(42)
        spec = mixed_spec_small
        train_x, train_y = _create_mixed_training_data(spec, n_obs=5)
        bounds = get_bounds_tensor(spec)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        from bo_engine.acquisition import create_single_objective_acquisition

        acqf = create_single_objective_acquisition(model=model, train_x=train_x, train_y=train_y)

        candidates, _ = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=2,
            spec=spec,
            num_restarts=5,
            raw_samples=64,
        )

        # Continuous dim: index 0, bounds [0, 1]
        for i in range(candidates.shape[0]):
            x1_val = candidates[i, 0].item()
            assert -1e-6 <= x1_val <= 1.0 + 1e-6, f"x1={x1_val} out of bounds"


# =============================================================================
# TestOptimizeMixedThreshold
# =============================================================================


class TestOptimizeMixedThreshold:
    """Test NotImplementedError for large mixed spaces."""

    def test_raises_for_large_mixed_space(self) -> None:
        """Mixed space exceeding threshold should raise NotImplementedError."""
        # Create a mixed spec with many categorical combos
        # 3 categoricals with 10 each = 1000 combos > 100 threshold
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(
                    name="cat1",
                    type=ParameterType.CATEGORICAL,
                    categories=[f"c{i}" for i in range(10)],
                ),
                ParameterSpec(
                    name="cat2",
                    type=ParameterType.CATEGORICAL,
                    categories=[f"c{i}" for i in range(10)],
                ),
                ParameterSpec(
                    name="cat3",
                    type=ParameterType.CATEGORICAL,
                    categories=[f"c{i}" for i in range(11)],
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        # Verify it's actually mixed and exceeds the threshold
        from bo_engine.transforms import (
            SearchSpaceType,
            classify_search_space,
            count_categorical_combinations,
        )

        assert classify_search_space(spec) == SearchSpaceType.MIXED
        assert count_categorical_combinations(spec) > MIXED_CATEGORICAL_COMBO_THRESHOLD

        # Create a dummy acquisition function
        torch.manual_seed(42)
        n_dims = get_n_dims(spec)
        train_x = torch.rand(5, n_dims, dtype=torch.float64)
        train_y = torch.rand(5, 1, dtype=torch.float64)
        bounds = get_bounds_tensor(spec)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        from bo_engine.acquisition import create_single_objective_acquisition

        acqf = create_single_objective_acquisition(model=model, train_x=train_x, train_y=train_y)

        with pytest.raises(NotImplementedError, match="not yet supported"):
            optimize_acquisition(
                acqf=acqf,
                bounds=bounds,
                batch_size=1,
                spec=spec,
            )


# =============================================================================
# TestContinuousFallback
# =============================================================================


class TestContinuousFallback:
    """Test that continuous specs still work correctly (no regression)."""

    def test_spec_none_uses_continuous(self) -> None:
        """When spec is None, should use the continuous path."""
        torch.manual_seed(42)
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        from bo_engine.acquisition import create_single_objective_acquisition

        acqf = create_single_objective_acquisition(model=model, train_x=train_x, train_y=train_y)

        candidates, _ = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=2,
            spec=None,
            num_restarts=5,
            raw_samples=64,
        )

        assert candidates.shape == (2, 2)
        assert torch.all(candidates >= -1e-6) and torch.all(candidates <= 1.0 + 1e-6)

    def test_continuous_spec_uses_continuous(self) -> None:
        """Continuous spec should route to _optimize_continuous."""
        torch.manual_seed(42)
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = get_bounds_tensor(spec)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        from bo_engine.acquisition import create_single_objective_acquisition

        acqf = create_single_objective_acquisition(model=model, train_x=train_x, train_y=train_y)

        candidates, _ = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=2,
            spec=spec,
            num_restarts=5,
            raw_samples=64,
        )

        assert candidates.shape == (2, 2)
