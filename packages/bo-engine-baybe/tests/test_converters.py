"""Tests for converters between bo-engine and BayBE types.

Verifies that OptimizationSpec is correctly mapped to BayBE's
SearchSpace, Objective, and Recommender types, including multi-objective
via ParetoObjective.

Reference: BayBE documentation — Parameter types and SearchSpace construction
https://emdgroup.github.io/baybe/stable/userguide/searchspace.html
"""

import math
from typing import Any

import pandas as pd
import pytest
from baybe.constraints import (
    ContinuousLinearConstraint,
    DiscreteSumConstraint,
)
from baybe.objectives import ParetoObjective, SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

from bo_engine_baybe.converters import (
    classify_constraint_target,
    dataframe_to_suggestions,
    observations_to_dataframe,
    pending_points_to_dataframe,
    spec_to_constraints,
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

    def test_fractional_discrete_values(self) -> None:
        """Fractional numerical discrete grids round-trip without per-backend hacks.

        BayBE supports ``NumericalDiscreteParameter`` over arbitrary floats; the
        neutral ``ParameterSpec.values`` slot now accepts ``list[float]`` so a
        simplex grid like ``[0.0, 0.25, 0.5, 0.75, 1.0]`` does not have to be
        encoded as integers and rescaled in user code.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="fraction",
                    type=ParameterType.DISCRETE,
                    values=[0.0, 0.25, 0.5, 0.75, 1.0],
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        params = spec_to_parameters(spec)
        assert len(params) == 1
        assert isinstance(params[0], NumericalDiscreteParameter)
        assert tuple(params[0].values) == pytest.approx((0.0, 0.25, 0.5, 0.75, 1.0))

    def test_task_parameter_from_options(self) -> None:
        """``parameter_options['baybe'].role == 'task'`` emits a TaskParameter."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="lab",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B", "C"],
                    parameter_options={
                        "baybe": {"role": "task", "active_values": ["A"]},
                    },
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        params = spec_to_parameters(spec)
        assert isinstance(params[0], TaskParameter)

    def test_categorical_encoding_from_options(self) -> None:
        """Categorical encoding choice flows through parameter_options."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="solvent",
                    type=ParameterType.CATEGORICAL,
                    categories=["Water", "Ethanol", "DMF"],
                    parameter_options={"baybe": {"encoding": "INT"}},
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        params = spec_to_parameters(spec)
        assert isinstance(params[0], CategoricalParameter)
        assert params[0].encoding.value == "INT"


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


class TestPendingPointsToDataframe:
    """Pending parameter dicts to BayBE pending_experiments dataframe."""

    def test_strips_extra_columns(self, simple_spec: OptimizationSpec) -> None:
        """Objective columns leak in if a caller reuses ObservationData dicts."""
        pending = [{"x1": 0.5, "x2": 0.3, "y": 1.0}, {"x1": 0.1, "x2": 0.9, "y": 0.5}]
        df = pending_points_to_dataframe(pending, simple_spec)
        assert list(df.columns) == ["x1", "x2"]
        assert len(df) == 2

    def test_missing_parameter_raises(self, simple_spec: OptimizationSpec) -> None:
        with pytest.raises(ValueError, match="missing parameter columns"):
            pending_points_to_dataframe([{"x1": 0.5}], simple_spec)

    def test_non_dict_entry_raises(self, simple_spec: OptimizationSpec) -> None:
        bad_entry: list[Any] = [("x1", 0.5)]
        with pytest.raises(TypeError, match="must be dicts"):
            pending_points_to_dataframe(bad_entry, simple_spec)

    def test_empty_input_returns_empty_frame(self, simple_spec: OptimizationSpec) -> None:
        df = pending_points_to_dataframe([], simple_spec)
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestClassifyConstraintTarget:
    """Classify constraints by referenced parameter types."""

    def _spec_params(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="c", type=ParameterType.DISCRETE, values=[0.0, 0.5, 1.0]),
            ParameterSpec(name="d", type=ParameterType.CATEGORICAL, categories=["x", "y"]),
        ]

    def test_continuous_only(self) -> None:
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["a", "b"],
            value=1.0,
        )
        assert classify_constraint_target(constraint, self._spec_params()) == "continuous"

    def test_discrete_only(self) -> None:
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["c"],
            value=1.0,
        )
        assert classify_constraint_target(constraint, self._spec_params()) == "discrete"

    def test_hybrid(self) -> None:
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["a", "c"],
            value=1.0,
        )
        assert classify_constraint_target(constraint, self._spec_params()) == "hybrid"

    def test_categorical(self) -> None:
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["d"],
            value=1.0,
        )
        assert classify_constraint_target(constraint, self._spec_params()) == "categorical"


class TestSpecToConstraints:
    """Constraint mapping respects parameter classification."""

    def test_continuous_linear_constraint(self) -> None:
        params = [
            ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ]
        constraints = [
            ConstraintSpec(
                type=ConstraintType.SUM_EQUALS,
                parameters=["a", "b"],
                value=1.0,
            ),
        ]
        result = spec_to_constraints(constraints, params)
        assert result is not None
        assert isinstance(result[0], ContinuousLinearConstraint)
        assert result[0].operator == "="

    def test_discrete_sum_constraint(self) -> None:
        """Discrete numerical sum constraints map to BayBE DiscreteSumConstraint."""
        params = [
            ParameterSpec(
                name="x", type=ParameterType.DISCRETE, values=[0.0, 0.25, 0.5, 0.75, 1.0]
            ),
            ParameterSpec(
                name="y", type=ParameterType.DISCRETE, values=[0.0, 0.25, 0.5, 0.75, 1.0]
            ),
        ]
        constraints = [
            ConstraintSpec(
                type=ConstraintType.SUM_EQUALS,
                parameters=["x", "y"],
                value=1.0,
            ),
        ]
        result = spec_to_constraints(constraints, params)
        assert result is not None
        assert isinstance(result[0], DiscreteSumConstraint)

    def test_hybrid_constraint_raises(self) -> None:
        """BayBE has no shared linear model for mixed continuous/discrete constraints."""
        params = [
            ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="b", type=ParameterType.DISCRETE, values=[0.0, 0.5, 1.0]),
        ]
        constraints = [
            ConstraintSpec(
                type=ConstraintType.SUM_EQUALS,
                parameters=["a", "b"],
                value=1.0,
            ),
        ]
        with pytest.raises(ValueError, match="mixed continuous/discrete"):
            spec_to_constraints(constraints, params)

    def test_categorical_arithmetic_raises(self) -> None:
        params = [
            ParameterSpec(name="c", type=ParameterType.CATEGORICAL, categories=["x", "y"]),
        ]
        constraints = [
            ConstraintSpec(
                type=ConstraintType.SUM_EQUALS,
                parameters=["c"],
                value=1.0,
            ),
        ]
        with pytest.raises(ValueError, match="categorical"):
            spec_to_constraints(constraints, params)


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
