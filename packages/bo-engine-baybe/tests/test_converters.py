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

from bo_engine.constants import DISCRETE_ENUMERATION_MAX_POINTS
from bo_engine.types import (
    AcquisitionMethod,
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.constants import DEFAULT_MAX_CANDIDATES
from bo_engine_baybe.converters import (
    classify_constraint_target,
    dataframe_to_suggestions,
    observations_to_dataframe,
    pending_points_to_dataframe,
    spec_to_acquisition_function,
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


class TestDiscreteEnumerationLimits:
    """Grid semantics and enumeration caps for the discrete search space.

    BayBE's ``SearchSpace.from_product`` materializes the full Cartesian
    product of all discrete/categorical parameters into a DataFrame, which
    the BayBE user guide flags as a memory hazard for large product spaces:
    https://emdgroup.github.io/baybe/stable/userguide/searchspace.html
    The shared engine limit (``DISCRETE_ENUMERATION_MAX_POINTS``) is
    therefore enforced on the *product* across parameters — a per-parameter
    check alone would let two just-under-limit grids multiply into an
    unbuildable (tens-of-GB) frame.
    """

    @staticmethod
    def _spec(parameters: list[ParameterSpec]) -> OptimizationSpec:
        return OptimizationSpec(
            parameters=parameters,
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

    def test_bounds_only_discrete_materializes_integer_grid(self) -> None:
        """Bounds-only discrete means the integer grid over [lo, hi] (BoTorch parity)."""
        spec = self._spec([ParameterSpec(name="n", type=ParameterType.DISCRETE, bounds=(0.5, 4.2))])
        (param,) = spec_to_parameters(spec)
        assert isinstance(param, NumericalDiscreteParameter)
        assert tuple(param.values) == (1.0, 2.0, 3.0, 4.0)

    def test_bounds_spanning_fewer_than_two_integers_raise(self) -> None:
        spec = self._spec([ParameterSpec(name="n", type=ParameterType.DISCRETE, bounds=(0.1, 0.9))])
        with pytest.raises(ValueError, match="fewer than 2 integer values"):
            spec_to_parameters(spec)

    def test_single_grid_above_limit_raises(self) -> None:
        spec = self._spec(
            [
                ParameterSpec(
                    name="n",
                    type=ParameterType.DISCRETE,
                    bounds=(0.0, float(DISCRETE_ENUMERATION_MAX_POINTS)),
                )
            ]
        )
        with pytest.raises(ValueError, match="enumeration limit"):
            spec_to_searchspace(spec)

    def test_two_large_grids_subsample_instead_of_failing(self) -> None:
        """Two per-parameter-legal grids route through the bounded subsampler.

        Each parameter spans ~``DISCRETE_ENUMERATION_MAX_POINTS`` integers
        (passes the per-parameter check); their 10^8-row product previously
        raised the enumeration-limit error. The large-categorical safeguard
        now builds a bounded, deterministically subsampled discrete
        subspace instead — keeping explicit ``backend='baybe'`` usable
        while ``backend='auto'`` routes to a FULL backend via the DEGRADED
        capability report.
        """
        big = float(DISCRETE_ENUMERATION_MAX_POINTS - 1)
        spec = self._spec(
            [
                ParameterSpec(name="a", type=ParameterType.DISCRETE, bounds=(0.0, big)),
                ParameterSpec(name="b", type=ParameterType.DISCRETE, bounds=(0.0, big)),
            ]
        )
        searchspace = spec_to_searchspace(spec)
        n_rows = len(searchspace.discrete.exp_rep)
        assert 0 < n_rows <= DISCRETE_ENUMERATION_MAX_POINTS

    def test_product_at_limit_builds_expected_grid(self) -> None:
        """A product exactly at the cap still builds, with the full grid."""
        spec = self._spec(
            [
                ParameterSpec(name="a", type=ParameterType.DISCRETE, bounds=(0.0, 99.0)),
                ParameterSpec(name="b", type=ParameterType.DISCRETE, bounds=(0.0, 99.0)),
            ]
        )
        searchspace = spec_to_searchspace(spec)
        assert len(searchspace.discrete.exp_rep) == 100 * 100

    def test_values_and_categorical_mix_counted_in_product(self) -> None:
        """Explicit-values grids and category lists multiply into the same cap.

        The 20 000-combination product exceeds the candidate cap, so the
        safeguard subsamples rather than enumerating (or rejecting).
        """
        spec = self._spec(
            [
                ParameterSpec(
                    name="grid",
                    type=ParameterType.DISCRETE,
                    values=[float(i) for i in range(200)],
                ),
                ParameterSpec(
                    name="cat",
                    type=ParameterType.CATEGORICAL,
                    categories=[f"c{i}" for i in range(100)],
                ),
            ]
        )
        searchspace = spec_to_searchspace(spec)
        assert 0 < len(searchspace.discrete.exp_rep) <= DEFAULT_MAX_CANDIDATES


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

    def test_log_transform_chains_logarithmic_transformation(self) -> None:
        """``log_transform=True`` maps to BayBE's logarithmic target transformation.

        BayBE's modern target interface composes transformations onto
        ``NumericalTarget`` (``target.log()``); the converter must emit it
        so the user's log intent is honored instead of silently dropped.
        Reference: BayBE target transformations —
        https://emdgroup.github.io/baybe/stable/userguide/targets.html
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="rate", minimize=True, log_transform=True)],
        )
        objective = spec_to_objective(spec)
        assert isinstance(objective, SingleTargetObjective)
        target = objective._target
        assert isinstance(target, NumericalTarget)
        assert target.minimize is True
        assert "Logarithmic" in type(target.transformation).__name__

    def test_log_transform_with_maximize_rejected(self) -> None:
        """``log_transform=True`` + ``minimize=False`` is outside the contract.

        Mirrors the BoTorch model factory: the neutral
        :class:`bo_engine.types.ObjectiveSpec` contract restricts the flag
        to minimize objectives, so both backends must reject the maximize
        combination with a ``ValueError`` instead of diverging silently.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="rate", minimize=False, log_transform=True)],
        )
        with pytest.raises(ValueError, match="minimize=True"):
            spec_to_objective(spec)

    def test_without_log_transform_no_transformation_chained(self) -> None:
        """Plain objectives keep the identity transformation (regression guard)."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        objective = spec_to_objective(spec)
        assert isinstance(objective, SingleTargetObjective)
        target = objective._target
        assert isinstance(target, NumericalTarget)
        assert "Logarithmic" not in type(target.transformation).__name__


class TestSpecToAcquisitionFunction:
    """``spec.acquisition_method`` → BayBE acquisition-function name.

    The dispatch mirrors ``bo_engine.acquisition.create_acquisition``:
    the objective count selects the acquisition family and the requested
    method is consulted within it. BayBE acqf names per
    https://emdgroup.github.io/baybe/stable/userguide/acquisition.html
    """

    @staticmethod
    def _spec(method: AcquisitionMethod, n_objectives: int = 1) -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name=f"y{i}", minimize=True) for i in range(n_objectives)],
            acquisition_method=method,
        )

    def test_auto_keeps_baybe_default(self) -> None:
        assert spec_to_acquisition_function(self._spec(AcquisitionMethod.AUTO)) is None

    def test_single_objective_expected_improvement(self) -> None:
        assert (
            spec_to_acquisition_function(self._spec(AcquisitionMethod.EXPECTED_IMPROVEMENT))
            == "qLogEI"
        )

    def test_single_objective_noisy_ei(self) -> None:
        assert spec_to_acquisition_function(self._spec(AcquisitionMethod.NOISY_EI)) == "qLogNEI"

    def test_multi_objective_hypervolume(self) -> None:
        assert (
            spec_to_acquisition_function(
                self._spec(AcquisitionMethod.HYPERVOLUME_IMPROVEMENT, n_objectives=2)
            )
            == "qLogNEHVI"
        )

    def test_multi_objective_scalarized(self) -> None:
        assert (
            spec_to_acquisition_function(
                self._spec(AcquisitionMethod.SCALARIZED_MULTI_OBJ, n_objectives=2)
            )
            == "qLogNParEGO"
        )

    def test_family_mismatch_resolves_like_botorch(self) -> None:
        """A multi-objective-only method on a single-objective spec → noisy EI.

        Mirrors the BoTorch dispatch (``use_noisy = method != EI`` for
        single-objective specs) so ``backend="auto"`` comparisons stay
        like-for-like instead of erroring inside BayBE.
        """
        assert (
            spec_to_acquisition_function(self._spec(AcquisitionMethod.HYPERVOLUME_IMPROVEMENT))
            == "qLogNEI"
        )
        assert (
            spec_to_acquisition_function(
                self._spec(AcquisitionMethod.EXPECTED_IMPROVEMENT, n_objectives=2)
            )
            == "qLogNEHVI"
        )

    def test_unmappable_methods_fall_back_to_default(self) -> None:
        """cost_weighted_ei / multi_fidelity_kg have no BayBE equivalent.

        The capability layer reports them; the converter returns ``None``
        so an acknowledged run still produces suggestions with BayBE's
        default acquisition function.
        """
        assert spec_to_acquisition_function(self._spec(AcquisitionMethod.COST_WEIGHTED_EI)) is None
        assert spec_to_acquisition_function(self._spec(AcquisitionMethod.MULTI_FIDELITY_KG)) is None


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

    @staticmethod
    def _log_spec() -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="rate", minimize=True, log_transform=True)],
        )

    def test_log_transform_rejects_non_positive_targets(self) -> None:
        """Non-positive targets for a log objective fail loud at the boundary.

        ``log(y)`` is undefined for ``y <= 0``; without this guard the
        failure mode is a NaN cascade inside the GP fit. Mirrors the
        BoTorch model factory's construction-time check
        (``_assert_positive_for_log_transform``) so both backends raise
        the same clear ``ValueError``.
        """
        observations = [
            ObservationData(parameter_values={"x": 0.2}, objective_values={"rate": 1.5}),
            ObservationData(parameter_values={"x": 0.6}, objective_values={"rate": -0.1}),
        ]
        with pytest.raises(ValueError, match="strictly positive"):
            observations_to_dataframe(observations, self._log_spec())

    def test_log_transform_rejects_non_finite_targets(self) -> None:
        observations = [
            ObservationData(parameter_values={"x": 0.2}, objective_values={"rate": float("nan")}),
        ]
        with pytest.raises(ValueError, match="finite"):
            observations_to_dataframe(observations, self._log_spec())

    def test_log_transform_accepts_positive_targets(self) -> None:
        observations = [
            ObservationData(parameter_values={"x": 0.2}, objective_values={"rate": 0.001}),
            ObservationData(parameter_values={"x": 0.6}, objective_values={"rate": 150.0}),
        ]
        df = observations_to_dataframe(observations, self._log_spec())
        assert len(df) == 2


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


class TestDiscreteConstraintConstruction:
    """Only the arithmetic (SUM/PRODUCT) neutral types reach the discrete builder.

    The discrete dispatch map must contain exactly the SUM and PRODUCT
    types: SUM types build ``DiscreteSumConstraint``, PRODUCT types build
    ``DiscreteProductConstraint``, and any other member reaching the
    builder would be dead code implying support the spec cannot express.
    """

    def test_dispatch_map_contains_exactly_the_arithmetic_types(self) -> None:
        from bo_engine_baybe.converters import _DISCRETE_OPERATOR_MAP

        assert set(_DISCRETE_OPERATOR_MAP) == {
            ConstraintType.SUM_EQUALS,
            ConstraintType.SUM_LESS_THAN,
            ConstraintType.SUM_GREATER_THAN,
            ConstraintType.PRODUCT_EQUALS,
            ConstraintType.PRODUCT_LESS_THAN,
            ConstraintType.PRODUCT_GREATER_THAN,
        }

    def test_sum_types_build_discrete_sum_constraints(self) -> None:
        from baybe.constraints import DiscreteSumConstraint

        from bo_engine_baybe.converters import _build_discrete_constraint

        for constraint_type in (
            ConstraintType.SUM_EQUALS,
            ConstraintType.SUM_LESS_THAN,
            ConstraintType.SUM_GREATER_THAN,
        ):
            constraint = ConstraintSpec(type=constraint_type, parameters=["c1", "c2"], value=1.0)
            built = _build_discrete_constraint(constraint)
            assert isinstance(built, DiscreteSumConstraint)
