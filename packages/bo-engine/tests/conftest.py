"""Pytest configuration for bo-engine tests.

This module provides shared fixtures and pytest markers for the test suite.
"""

from typing import Any

import pytest
import torch

from bo_engine import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: Any) -> None:
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "slow: marks tests as slow (>30s)")
    config.addinivalue_line("markers", "gpu: marks tests that benefit from GPU")
    config.addinivalue_line("markers", "smoke: fast subset for CI")
    config.addinivalue_line("markers", "tutorial: tests reproducing BoTorch tutorial results")
    config.addinivalue_line("markers", "integration: integration tests spanning multiple modules")


# =============================================================================
# Parameterization Helpers
# =============================================================================

# Use these for tests that can run in "smoke" (fast) or "full" (comprehensive) mode
SMOKE_BATCH_SIZE = 1
SMOKE_ITERATIONS = 5
FULL_BATCH_SIZE = 4
FULL_ITERATIONS = 20


# =============================================================================
# Shared Fixtures - Device Management
# =============================================================================


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    """Standard dtype for tests."""
    return torch.float64


# =============================================================================
# Shared Fixtures - Optimization Specs
# =============================================================================


@pytest.fixture
def single_objective_spec() -> OptimizationSpec:
    """Create a standard single-objective optimization spec for testing."""
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
def multi_objective_spec() -> OptimizationSpec:
    """Create a standard multi-objective optimization spec for testing."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="f1", minimize=True),
            ObjectiveSpec(name="f2", minimize=True),
        ],
        batch_size=2,
    )


@pytest.fixture
def high_dim_spec() -> OptimizationSpec:
    """Create a high-dimensional optimization spec for TuRBO testing."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
            for i in range(20)
        ],
        objectives=[
            ObjectiveSpec(name="y", minimize=True),
        ],
        batch_size=4,
    )


# =============================================================================
# Shared Fixtures - Sample Data
# =============================================================================


@pytest.fixture
def sample_observations_single() -> list[ObservationData]:
    """Create sample single-objective observations."""
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
    ]


@pytest.fixture
def sample_observations_multi() -> list[ObservationData]:
    """Create sample multi-objective observations."""
    return [
        ObservationData(
            parameter_values={"x1": 0.1, "x2": 0.2},
            objective_values={"f1": 5.0, "f2": 2.0},
        ),
        ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.5},
            objective_values={"f1": 3.0, "f2": 4.0},
        ),
        ObservationData(
            parameter_values={"x1": 0.9, "x2": 0.1},
            objective_values={"f1": 4.0, "f2": 3.0},
        ),
    ]


# =============================================================================
# Shared Fixtures - Training Data Tensors
# =============================================================================


@pytest.fixture
def branin_training_data(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Create training data from Branin function for GP testing.

    Returns:
        Tuple of (train_x, train_y) tensors
    """
    from bo_engine.benchmarks import branin, branin_bounds

    torch.manual_seed(42)
    bounds = branin_bounds()
    # Generate Sobol points in [0, 1]^2 then scale to Branin bounds
    train_x_unit = torch.rand(10, 2, dtype=dtype)
    train_x = train_x_unit * (bounds[1] - bounds[0]) + bounds[0]
    train_y = branin(train_x).unsqueeze(-1)
    return train_x, train_y


@pytest.fixture
def hartmann6_training_data(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Create training data from Hartmann6 function for GP testing.

    Returns:
        Tuple of (train_x, train_y) tensors
    """
    from bo_engine.benchmarks import hartmann6

    torch.manual_seed(42)
    train_x = torch.rand(15, 6, dtype=dtype)
    train_y = hartmann6(train_x).unsqueeze(-1)
    return train_x, train_y


# =============================================================================
# Utility Functions for Tests
# =============================================================================


def assert_tensor_in_bounds(tensor: torch.Tensor, bounds: torch.Tensor, msg: str = "") -> None:
    """Assert that all values in tensor are within bounds.

    Args:
        tensor: Tensor of shape (n, d) to check
        bounds: Bounds tensor of shape (2, d) with [lower, upper]
        msg: Optional message for assertion error
    """
    lower, upper = bounds[0], bounds[1]
    assert (tensor >= lower - 1e-6).all() and (tensor <= upper + 1e-6).all(), (
        f"Tensor values outside bounds. {msg}"
    )


def generate_sobol_points(
    n: int, d: int, bounds: torch.Tensor, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Generate Sobol sequence points within bounds.

    Args:
        n: Number of points
        d: Dimensionality
        bounds: Bounds tensor of shape (2, d)
        dtype: Tensor dtype

    Returns:
        Tensor of shape (n, d) with Sobol points
    """
    from torch.quasirandom import SobolEngine

    sobol = SobolEngine(dimension=d, scramble=True)
    points = sobol.draw(n).to(dtype)
    # Scale to bounds
    return points * (bounds[1] - bounds[0]) + bounds[0]
