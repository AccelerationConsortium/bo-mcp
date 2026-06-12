"""Shared test fixtures for bo-engine-baybe tests."""

import pytest

from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


@pytest.fixture
def simple_spec() -> OptimizationSpec:
    """A simple 2D single-objective spec."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        batch_size=2,
    )


@pytest.fixture
def categorical_spec() -> OptimizationSpec:
    """A spec with categorical and continuous parameters."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="temp", type=ParameterType.CONTINUOUS, bounds=(200.0, 400.0)),
            ParameterSpec(
                name="solvent",
                type=ParameterType.CATEGORICAL,
                categories=["Water", "Ethanol", "DMF"],
            ),
        ],
        objectives=[ObjectiveSpec(name="yield", minimize=False)],
        batch_size=3,
    )


@pytest.fixture
def constrained_spec() -> OptimizationSpec:
    """A spec with a sum constraint."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x3", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        constraints=[
            ConstraintSpec(
                type=ConstraintType.SUM_EQUALS,
                parameters=["x1", "x2", "x3"],
                value=1.0,
            ),
        ],
        batch_size=2,
    )
