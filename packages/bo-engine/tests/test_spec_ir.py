"""Tests for the shared OptimizationSpec normalization helpers.

These tests verify that :mod:`bo_engine.spec_ir` provides a single
classification implementation that backend converters consume. The
contract test at the bottom feeds the same neutral spec through both
the classifier and the BayBE converter and asserts they agree — if the
BayBE converter's per-constraint dispatch diverged from the shared
classifier, the two would disagree on which constraints to construct.

Reference: BoTorch / BayBE both consume a common neutral spec
(``OptimizationSpec``); shared dispatch keeps capability reports and
converter construction in lock-step (see ``baybe_constraint_support``).
"""

from __future__ import annotations

import pytest

from bo_engine.spec_ir import (
    ConstraintTargetClass,
    classify_constraint_target,
    normalize_spec,
)
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _params() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ParameterSpec(name="c", type=ParameterType.DISCRETE, values=[0.0, 0.5, 1.0]),
        ParameterSpec(name="d", type=ParameterType.CATEGORICAL, categories=["x", "y"]),
    ]


class TestClassifyConstraintTarget:
    def test_continuous_only(self) -> None:
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["a", "b"],
            value=1.0,
        )
        assert classify_constraint_target(constraint, _params()) == ConstraintTargetClass.CONTINUOUS

    def test_discrete_only(self) -> None:
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["c"],
            value=1.0,
        )
        assert classify_constraint_target(constraint, _params()) == ConstraintTargetClass.DISCRETE

    def test_hybrid(self) -> None:
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["a", "c"],
            value=1.0,
        )
        assert classify_constraint_target(constraint, _params()) == ConstraintTargetClass.HYBRID

    def test_categorical(self) -> None:
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["d"],
            value=1.0,
        )
        target_class = classify_constraint_target(constraint, _params())
        assert target_class == ConstraintTargetClass.CATEGORICAL

    def test_unknown_when_reference_missing(self) -> None:
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["a", "ghost"],
            value=1.0,
        )
        assert classify_constraint_target(constraint, _params()) == ConstraintTargetClass.UNKNOWN

    def test_unknown_when_empty_parameter_list(self) -> None:
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=[],
            value=0.0,
        )
        assert classify_constraint_target(constraint, _params()) == ConstraintTargetClass.UNKNOWN

    def test_string_equality_preserved(self) -> None:
        """Legacy callers compare the classification against bare strings."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["a", "b"],
            value=1.0,
        )
        result = classify_constraint_target(constraint, _params())
        assert result == "continuous"


class TestNormalizeSpec:
    def test_pre_computes_constraint_classification(self) -> None:
        params = _params()
        constraints = [
            ConstraintSpec(
                type=ConstraintType.SUM_EQUALS,
                parameters=["a", "b"],
                value=1.0,
            ),
            ConstraintSpec(
                type=ConstraintType.SUM_EQUALS,
                parameters=["c"],
                value=0.5,
            ),
            ConstraintSpec(
                type=ConstraintType.SUM_EQUALS,
                parameters=["a", "c"],
                value=1.0,
            ),
        ]
        spec = OptimizationSpec(
            parameters=params,
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            constraints=constraints,
        )
        normalized = normalize_spec(spec)

        assert len(normalized.constraints) == 3
        assert normalized.constraints[0].target_class == ConstraintTargetClass.CONTINUOUS
        assert normalized.constraints[1].target_class == ConstraintTargetClass.DISCRETE
        assert normalized.constraints[2].target_class == ConstraintTargetClass.HYBRID
        assert normalized.spec is spec

    def test_no_constraints_yields_empty_tuple(self) -> None:
        spec = OptimizationSpec(
            parameters=_params(),
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        normalized = normalize_spec(spec)
        assert normalized.constraints == ()


class TestConverterContract:
    """Both engine converters must dispatch on the same shared classification.

    Feeds a multi-constraint spec through the BayBE converter and asserts
    its construction matches the shared :func:`classify_constraint_target`
    output. If BayBE re-implemented the dispatch divergent decisions would
    surface here.
    """

    @pytest.mark.parametrize(
        ("parameters", "constraint", "expected_class"),
        [
            pytest.param(
                [
                    ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                    ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ],
                ConstraintSpec(type=ConstraintType.SUM_EQUALS, parameters=["a", "b"], value=1.0),
                ConstraintTargetClass.CONTINUOUS,
                id="continuous_only",
            ),
            pytest.param(
                [
                    ParameterSpec(name="c", type=ParameterType.DISCRETE, values=[0.0, 0.5, 1.0]),
                ],
                ConstraintSpec(type=ConstraintType.SUM_EQUALS, parameters=["c"], value=0.5),
                ConstraintTargetClass.DISCRETE,
                id="discrete_only",
            ),
        ],
    )
    def test_baybe_dispatch_matches_shared_classifier(
        self,
        parameters: list[ParameterSpec],
        constraint: ConstraintSpec,
        expected_class: ConstraintTargetClass,
    ) -> None:
        baybe_converters = pytest.importorskip("bo_engine_baybe.converters")

        assert classify_constraint_target(constraint, parameters) == expected_class

        built = baybe_converters.spec_to_constraints([constraint], parameters)
        assert built is not None
        assert len(built) == 1
        if expected_class == ConstraintTargetClass.CONTINUOUS:
            # Continuous-only constraints map to ContinuousLinearConstraint.
            from baybe.constraints import ContinuousLinearConstraint

            assert isinstance(built[0], ContinuousLinearConstraint)
        elif expected_class == ConstraintTargetClass.DISCRETE:
            from baybe.constraints import DiscreteSumConstraint

            assert isinstance(built[0], DiscreteSumConstraint)

    def test_baybe_spec_to_searchspace_consumes_normalized_spec(self) -> None:
        """BayBE's searchspace builder must run the classifier exactly once.

        Wraps :func:`bo_engine.spec_ir.classify_constraint_target` with a
        counter; the converter should hit it via the normalized bundle
        rather than re-classifying inside ``spec_to_constraints``.
        """
        baybe_converters = pytest.importorskip("bo_engine_baybe.converters")
        spec_ir = pytest.importorskip("bo_engine.spec_ir")

        parameters = [
            ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ]
        spec = OptimizationSpec(
            parameters=parameters,
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            constraints=[
                ConstraintSpec(type=ConstraintType.SUM_EQUALS, parameters=["a", "b"], value=1.0),
            ],
        )
        original = spec_ir.classify_constraint_target
        calls: list[ConstraintSpec] = []

        def counting(constraint: ConstraintSpec, params: list[ParameterSpec]):
            calls.append(constraint)
            return original(constraint, params)

        spec_ir.classify_constraint_target = counting
        try:
            baybe_converters.spec_to_searchspace(spec)
        finally:
            spec_ir.classify_constraint_target = original

        # Exactly one classification — done by ``normalize_spec`` inside
        # the BayBE converter; the inner dispatch reuses that result.
        assert len(calls) == 1

    def test_botorch_and_baybe_agree_on_categorical_constraint_rejection(self) -> None:
        """Cross-backend contract: both backends classify the same constraint identically.

        BayBE refuses constraints that touch a categorical parameter, and
        BoTorch's native linear-constraint converter routes them to the
        projection bucket instead of the linear bucket. Both decisions
        must derive from the same shared
        :func:`classify_constraint_target` call so a future taxonomy
        change keeps the two engines in sync.
        """
        baybe_converters = pytest.importorskip("bo_engine_baybe.converters")
        from bo_engine.constraints import build_botorch_linear_constraints

        parameters = [
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="c", type=ParameterType.CATEGORICAL, categories=["a", "b"]),
        ]
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["x", "c"],
            value=1.0,
        )
        # Shared classifier puts categorical-touching constraints in the
        # CATEGORICAL bucket — that is the contract both backends honor.
        assert (
            classify_constraint_target(constraint, parameters) == ConstraintTargetClass.CATEGORICAL
        )

        # BayBE: rejected with a clear reason.
        with pytest.raises(ValueError, match="cannot express arithmetic constraint"):
            baybe_converters.spec_to_constraints([constraint], parameters)

        # BoTorch linear builder: routed to projection_constraints, not
        # inequality / equality buckets.
        spec = OptimizationSpec(
            parameters=parameters,
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            constraints=[constraint],
        )
        ineq, eq, projection = build_botorch_linear_constraints(spec)
        assert ineq == []
        assert eq == []
        assert projection == [constraint]
