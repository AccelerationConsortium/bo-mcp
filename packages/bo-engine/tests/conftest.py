"""Pytest configuration for bo-engine tests.

This module provides shared fixtures and pytest markers for the test suite.

Test Categories:
    - smoke: Fast critical path tests (< 5s each)
    - slow: Long-running tests (> 30s, skip in PR CI)
    - nightly: Statistical/stochastic tests (run in nightly CI only)
    - deterministic: Tests requiring torch deterministic mode
    - tutorial: Tests reproducing BoTorch tutorial results

References:
    - BoTorch Reproducibility: https://botorch.org/docs/reproducibility
    - PyTorch Randomness: https://pytorch.org/docs/stable/notes/randomness.html
"""

import os
import random
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
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
    config.addinivalue_line("markers", "nightly: statistical tests for nightly CI")
    config.addinivalue_line("markers", "deterministic: tests requiring deterministic torch ops")


# =============================================================================
# Parameterization Helpers
# =============================================================================

# Use these for tests that can run in "smoke" (fast) or "full" (comprehensive) mode
SMOKE_BATCH_SIZE = 1
SMOKE_ITERATIONS = 5
FULL_BATCH_SIZE = 4
FULL_ITERATIONS = 20

# =============================================================================
# Calibrated Tolerances (from scripts/calibrate_test_tolerances.py)
# =============================================================================
# These tolerances were empirically determined via Monte Carlo simulation
# Run `uv run python scripts/calibrate_test_tolerances.py` to recalibrate

# Pareto max tolerance for Branin-Currin after 5 BO iterations
PARETO_MAX_TOLERANCE_CI = 3.1  # 99th percentile + 10% margin (CI must always pass)
PARETO_MAX_TOLERANCE_NIGHTLY = 1.5  # 95th percentile (statistical tests)

# Minimum Pareto front size (invariant - should always hold)
MIN_PARETO_SIZE = 2

# Hypervolume minimum threshold (loose, for regression detection)
MIN_HYPERVOLUME_THRESHOLD = 0.1


# =============================================================================
# Deterministic Mode Utilities
# =============================================================================


@contextmanager
def deterministic_mode(seed: int = 42) -> Generator[None]:
    """Context manager for deterministic torch operations.

    This enables PyTorch's deterministic mode and sets all relevant seeds.
    Use this for tests that require reproducibility across different runs.

    Note: Some operations may be slower in deterministic mode.
    Not all operations have deterministic implementations.

    Args:
        seed: Random seed to use

    Yields:
        None

    Example:
        with deterministic_mode(42):
            result = some_torch_operation()
    """
    # Save current state
    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    prev_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")

    try:
        # Configure deterministic mode
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        yield
    finally:
        # Restore previous state
        torch.use_deterministic_algorithms(prev_deterministic)
        if prev_cublas_config:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_cublas_config
        elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]


def set_all_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for:
    - torch (CPU)
    - torch.cuda (GPU)
    - Python's random module
    - numpy (if imported)

    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np

        np.random.seed(seed)  # noqa: NPY002
    except ImportError:
        pass


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
# Shared Fixtures - Reproducibility
# =============================================================================


@pytest.fixture
def seed() -> int:
    """Default seed for reproducible tests."""
    return 42


@pytest.fixture
def seeded(seed: int) -> int:
    """Fixture that sets torch seed and returns it.

    Use this for tests that need reproducibility but not full determinism.
    """
    set_all_seeds(seed)
    return seed


@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible NumPy random generator for stochastic tests.

    Use this for tests that call generate_next_batch() or other functions
    accepting an rng parameter.
    """
    return np.random.default_rng(42)


@pytest.fixture
def torch_rng() -> None:
    """Fixture that seeds torch for reproducible model fitting and tensor ops.

    Use this for tests that call torch.rand(), create_and_fit_model(), or
    other torch-level stochastic operations.
    """
    torch.manual_seed(42)


@pytest.fixture
def deterministic_seed() -> Generator[int]:
    """Fixture for tests requiring full deterministic mode.

    Enables torch deterministic algorithms and sets all seeds.
    Use for tests marked with @pytest.mark.deterministic.

    Note: Some operations may be slower or unavailable in this mode.
    """
    seed = 42
    with deterministic_mode(seed):
        yield seed


@pytest.fixture
def tolerance_ci() -> dict[str, float]:
    """Calibrated tolerances for CI tests (must always pass)."""
    return {
        "pareto_max": PARETO_MAX_TOLERANCE_CI,
        "min_pareto_size": MIN_PARETO_SIZE,
        "min_hypervolume": MIN_HYPERVOLUME_THRESHOLD,
    }


@pytest.fixture
def tolerance_nightly() -> dict[str, float]:
    """Calibrated tolerances for nightly tests (statistical)."""
    return {
        "pareto_max": PARETO_MAX_TOLERANCE_NIGHTLY,
        "min_pareto_size": MIN_PARETO_SIZE,
        "min_hypervolume": MIN_HYPERVOLUME_THRESHOLD,
    }


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
    """Create sample single-objective observations.

    Provides 5 observations for 2 parameters, comfortably above the minimum
    data requirement of n_params+1 = 3 for GP model fitting.
    """
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
            parameter_values={"x1": 0.7, "x2": 0.3},
            objective_values={"y": 4.5},
        ),
    ]


@pytest.fixture
def sample_observations_multi() -> list[ObservationData]:
    """Create sample multi-objective observations.

    Provides 5 observations for 2 parameters, comfortably above the minimum
    data requirement of n_params+1 = 3 for GP model fitting.
    """
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
        ObservationData(
            parameter_values={"x1": 0.3, "x2": 0.8},
            objective_values={"f1": 3.5, "f2": 3.5},
        ),
        ObservationData(
            parameter_values={"x1": 0.7, "x2": 0.4},
            objective_values={"f1": 4.5, "f2": 2.5},
        ),
    ]


# =============================================================================
# Shared Fixtures - Training Data Tensors
# =============================================================================


@pytest.fixture
def branin_training_data(dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create training data from Branin function for GP testing.

    The fixture depends on the ``seed`` fixture, so callers that parameterize
    the seed (e.g. via ``pytest.mark.parametrize("seed", [1, 2, 3])``) see
    distinct training sets. Seeding is routed through ``set_all_seeds`` so
    the full torch/numpy/random tuple is consistent.

    Returns:
        Tuple of (train_x, train_y) tensors
    """
    from bo_engine.benchmarks import branin, branin_bounds

    set_all_seeds(seed)
    bounds = branin_bounds()
    # Generate Sobol points in [0, 1]^2 then scale to Branin bounds
    train_x_unit = torch.rand(10, 2, dtype=dtype)
    train_x = train_x_unit * (bounds[1] - bounds[0]) + bounds[0]
    train_y = branin(train_x).unsqueeze(-1)
    return train_x, train_y


@pytest.fixture
def hartmann6_training_data(dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create training data from Hartmann6 function for GP testing.

    The fixture depends on the ``seed`` fixture, so callers that parameterize
    the seed (e.g. via ``pytest.mark.parametrize("seed", [1, 2, 3])``) see
    distinct training sets. Seeding is routed through ``set_all_seeds`` so
    the full torch/numpy/random tuple is consistent.

    Returns:
        Tuple of (train_x, train_y) tensors
    """
    from bo_engine.benchmarks import hartmann6

    set_all_seeds(seed)
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
    assert (tensor >= lower - 1e-6).all(), f"Tensor values below lower bound. {msg}"
    assert (tensor <= upper + 1e-6).all(), f"Tensor values above upper bound. {msg}"


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
