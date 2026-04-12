"""Tests for converters between bo-engine and BayBE types.

Verifies that OptimizationSpec is correctly mapped to BayBE's
SearchSpace, Objective, and Recommender types, including multi-objective
via ParetoObjective.

Reference: BayBE documentation — Parameter types and SearchSpace construction
https://emdgroup.github.io/baybe/stable/userguide/searchspace.html
"""

import math

from baybe.objectives import ParetoObjective, SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

from bo_engine_baybe.converters import (
    dataframe_to_suggestions,
    observations_to_dataframe,
    spec_to_objective,
    spec_to_parameters,
    spec_to_searchspace,
)


class TestSpecToParameters:
    def test_continuous_parameters(self, simple_spec: OptimizationSpec) -> None:
        params = spec_to_parameters(simple_spec)
        assert len(params) == 2
        assert all(isinstance(p, NumericalContinuousParameter) for p in params)
        assert params[0].name == "x1"
        assert params[1].name == "x2"

    def test_categorical_parameters(self, categorical_spec: OptimizationSpec) -> None:
        params = spec_to_parameters(categorical_spec)
        assert len(params) == 2
        assert isinstance(params[0], NumericalContinuousParameter)
        assert isinstance(params[1], CategoricalParameter)
        assert params[1].name == "solvent"

    def test_mixed_parameter_types(self, categorical_spec: OptimizationSpec) -> None:
        searchspace = spec_to_searchspace(categorical_spec)
        assert isinstance(searchspace, SearchSpace)


class TestSpecToObjective:
    def test_single_minimize(self, simple_spec: OptimizationSpec) -> None:
        objective = spec_to_objective(simple_spec)
        assert isinstance(objective, SingleTargetObjective)
        target = objective._target
        assert isinstance(target, NumericalTarget)
        assert target.name == "y"

    def test_single_maximize(self, categorical_spec: OptimizationSpec) -> None:
        objective = spec_to_objective(categorical_spec)
        assert isinstance(objective, SingleTargetObjective)
        target = objective._target
        assert isinstance(target, NumericalTarget)
        assert target.name == "yield"

    def test_multi_objective_returns_pareto(self) -> None:
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="y1", minimize=True),
                ObjectiveSpec(name="y2", minimize=False),
            ],
        )
        objective = spec_to_objective(spec)
        assert isinstance(objective, ParetoObjective)


class TestObservationsToDataframe:
    def test_basic_conversion(self, simple_spec: OptimizationSpec) -> None:
        observations = [
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.3},
                objective_values={"y": 1.2},
            ),
            ObservationData(
                parameter_values={"x1": 0.8, "x2": 0.1},
                objective_values={"y": 0.7},
            ),
        ]
        df = observations_to_dataframe(observations, simple_spec)
        assert len(df) == 2
        assert list(df.columns) == ["x1", "x2", "y"]
        assert math.isclose(df.iloc[0]["x1"], 0.5)
        assert math.isclose(df.iloc[1]["y"], 0.7)


class TestDataframeToSuggestions:
    def test_roundtrip(self, simple_spec: OptimizationSpec) -> None:
        observations = [
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.3},
                objective_values={"y": 1.2},
            ),
        ]
        df = observations_to_dataframe(observations, simple_spec)
        suggestions = dataframe_to_suggestions(df, simple_spec)
        assert len(suggestions) == 1
        assert math.isclose(suggestions[0]["x1"], 0.5)
        assert math.isclose(suggestions[0]["x2"], 0.3)
