"""Extended constraint surface on the BayBE backend (phases 4a + 4c).

Covers the newly reachable BayBE constraint classes — discrete products,
continuous/discrete cardinality (sparsity), interpoint linear semantics,
and the set-based label constraints (no-label-duplicates, linked
parameters, permutation invariance) — plus the completeness and
capability/converter symmetry gates that keep intake validation and
construction in lockstep. The deliberate exclusions (hybrid,
categorical-targeted arithmetic, discrete LINEAR) are pinned unchanged.

Reference: BayBE constraints userguide
(https://emdgroup.github.io/baybe/stable/userguide/constraints.html).
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from baybe.constraints import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
    DiscreteCardinalityConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteProductConstraint,
)

from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.backend import BayBEBackend
from bo_engine_baybe.converters import (
    baybe_constraint_support,
    spec_to_constraints,
)

_CONTINUOUS = [
    ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
    ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
]
_DISCRETE = [
    ParameterSpec(name="a", type=ParameterType.DISCRETE, values=[0.0, 1.0, 2.0]),
    ParameterSpec(name="b", type=ParameterType.DISCRETE, values=[0.0, 1.0, 2.0]),
]
_CATEGORICAL = [
    ParameterSpec(name="a", type=ParameterType.CATEGORICAL, categories=["x", "y", "z"]),
    ParameterSpec(name="b", type=ParameterType.CATEGORICAL, categories=["x", "y", "z"]),
]


def _spec(
    parameters: list[ParameterSpec],
    constraints: list[ConstraintSpec],
    **kwargs: Any,
) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=parameters,
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        constraints=constraints,
        **kwargs,
    )


class TestBuilders:
    def test_discrete_product_constraint(self) -> None:
        c = ConstraintSpec(type=ConstraintType.PRODUCT_LESS_THAN, parameters=["a", "b"], value=2.0)
        built = spec_to_constraints([c], _DISCRETE)
        assert built is not None
        assert isinstance(built[0], DiscreteProductConstraint)
        assert built[0].condition.threshold == pytest.approx(2.0)
        assert built[0].condition.operator == "<="

    def test_continuous_cardinality_constraint(self) -> None:
        c = ConstraintSpec(
            type=ConstraintType.CARDINALITY, parameters=["a", "b"], max_cardinality=1
        )
        built = spec_to_constraints([c], _CONTINUOUS)
        assert built is not None
        assert built[0] == ContinuousCardinalityConstraint(parameters=["a", "b"], max_cardinality=1)

    def test_discrete_cardinality_constraint(self) -> None:
        c = ConstraintSpec(
            type=ConstraintType.CARDINALITY,
            parameters=["a", "b"],
            min_cardinality=1,
            max_cardinality=1,
        )
        built = spec_to_constraints([c], _DISCRETE)
        assert built is not None
        assert built[0] == DiscreteCardinalityConstraint(
            parameters=["a", "b"], min_cardinality=1, max_cardinality=1
        )

    def test_interpoint_linear_constraint(self) -> None:
        c = ConstraintSpec(
            type=ConstraintType.SUM_LESS_THAN,
            parameters=["a", "b"],
            value=1.5,
            is_interpoint=True,
        )
        built = spec_to_constraints([c], _CONTINUOUS)
        assert built is not None
        assert isinstance(built[0], ContinuousLinearConstraint)
        assert built[0].is_interpoint is True

    @pytest.mark.parametrize(
        ("ctype", "expected_cls"),
        [
            (ConstraintType.NO_LABEL_DUPLICATES, DiscreteNoLabelDuplicatesConstraint),
            (ConstraintType.LINKED_PARAMETERS, DiscreteLinkedParametersConstraint),
            (ConstraintType.PERMUTATION_INVARIANCE, DiscretePermutationInvarianceConstraint),
        ],
    )
    def test_set_based_constraints(self, ctype: ConstraintType, expected_cls: type) -> None:
        c = ConstraintSpec(type=ctype, parameters=["a", "b"])
        built = spec_to_constraints([c], _CATEGORICAL)
        assert built is not None
        assert isinstance(built[0], expected_cls)


class TestSupportClassification:
    @pytest.mark.parametrize(
        ("constraint", "parameters", "fragment"),
        [
            # Deliberate exclusions preserved verbatim:
            (
                ConstraintSpec(
                    type=ConstraintType.LINEAR,
                    parameters=["a", "b"],
                    value=1.0,
                    coefficients=[1.0, 2.0],
                ),
                _DISCRETE,
                "no native discrete equivalent",
            ),
            (
                ConstraintSpec(type=ConstraintType.SUM_EQUALS, parameters=["a", "b"], value=1.0),
                [_CONTINUOUS[0], _DISCRETE[1]],
                "mixed continuous/discrete",
            ),
            (
                ConstraintSpec(type=ConstraintType.SUM_EQUALS, parameters=["a", "b"], value=1.0),
                _CATEGORICAL,
                "arithmetic constraint over categorical",
            ),
            # New-family rules:
            (
                ConstraintSpec(
                    type=ConstraintType.PRODUCT_EQUALS, parameters=["a", "b"], value=1.0
                ),
                _CONTINUOUS,
                "discrete-only",
            ),
            (
                ConstraintSpec(type=ConstraintType.NO_LABEL_DUPLICATES, parameters=["a", "b"]),
                _CONTINUOUS,
                "includes a continuous parameter",
            ),
            (
                ConstraintSpec(type=ConstraintType.CARDINALITY, parameters=["a", "b"]),
                _CONTINUOUS,
                "requires min_cardinality and/or max_cardinality",
            ),
            (
                ConstraintSpec(
                    type=ConstraintType.CARDINALITY,
                    parameters=["a", "b"],
                    min_cardinality=2,
                    max_cardinality=1,
                ),
                _CONTINUOUS,
                "min_cardinality <= max_cardinality",
            ),
            (
                ConstraintSpec(
                    type=ConstraintType.SUM_EQUALS,
                    parameters=["a", "b"],
                    value=1.0,
                    is_interpoint=True,
                ),
                _DISCRETE,
                "is_interpoint applies to continuous",
            ),
        ],
    )
    def test_rejections(
        self,
        constraint: ConstraintSpec,
        parameters: list[ParameterSpec],
        fragment: str,
    ) -> None:
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert not ok
        assert reason is not None
        assert fragment in reason
        # Converter/capability symmetry: what the report rejects, the
        # converter must refuse to build (and vice versa).
        with pytest.raises(ValueError, match=re.escape(fragment)):
            spec_to_constraints([constraint], parameters)


def _supported_case(ctype: ConstraintType) -> tuple[ConstraintSpec, list[ParameterSpec]]:
    """A (constraint, parameters) pair BayBE must support for each type."""
    if ctype in (
        ConstraintType.SUM_EQUALS,
        ConstraintType.SUM_LESS_THAN,
        ConstraintType.SUM_GREATER_THAN,
    ):
        return ConstraintSpec(type=ctype, parameters=["a", "b"], value=1.0), _CONTINUOUS
    if ctype == ConstraintType.LINEAR:
        return (
            ConstraintSpec(type=ctype, parameters=["a", "b"], value=1.0, coefficients=[1.0, 2.0]),
            _CONTINUOUS,
        )
    if ctype in (
        ConstraintType.PRODUCT_EQUALS,
        ConstraintType.PRODUCT_LESS_THAN,
        ConstraintType.PRODUCT_GREATER_THAN,
    ):
        return ConstraintSpec(type=ctype, parameters=["a", "b"], value=2.0), _DISCRETE
    if ctype == ConstraintType.CARDINALITY:
        return (
            ConstraintSpec(type=ctype, parameters=["a", "b"], max_cardinality=1),
            _CONTINUOUS,
        )
    return ConstraintSpec(type=ctype, parameters=["a", "b"]), _CATEGORICAL


class TestCompletenessGate:
    @pytest.mark.parametrize("ctype", list(ConstraintType))
    def test_every_type_has_support_and_converter_branch(self, ctype: ConstraintType) -> None:
        """No ConstraintType member may silently fall through the dispatch."""
        constraint, parameters = _supported_case(ctype)
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert ok, f"{ctype} unexpectedly unsupported: {reason}"
        built = spec_to_constraints([constraint], parameters)
        assert built is not None
        assert len(built) == 1


class TestRoutingSafety:
    @pytest.mark.parametrize(
        "ctype",
        [
            ConstraintType.PRODUCT_LESS_THAN,
            ConstraintType.CARDINALITY,
            ConstraintType.NO_LABEL_DUPLICATES,
        ],
    )
    def test_baybe_only_types_flip_botorch_unsupported(self, ctype: ConstraintType) -> None:
        """BayBE-only constraint types must veto BoTorch so 'auto' routes right."""
        constraint, parameters = _supported_case(ctype)
        spec = _spec(parameters, [constraint])
        assert BayBEBackend().validate_capabilities(spec).is_compatible
        botorch = BoTorchBackend().validate_capabilities(spec)
        assert not botorch.is_compatible

    def test_interpoint_flips_botorch_unsupported(self) -> None:
        spec = _spec(
            _CONTINUOUS,
            [
                ConstraintSpec(
                    type=ConstraintType.SUM_LESS_THAN,
                    parameters=["a", "b"],
                    value=1.5,
                    is_interpoint=True,
                )
            ],
        )
        assert BayBEBackend().validate_capabilities(spec).is_compatible
        assert not BoTorchBackend().validate_capabilities(spec).is_compatible

    def test_plain_linear_stays_supported_on_both(self) -> None:
        """Routing-safety regression: existing constraints keep routing as before."""
        spec = _spec(
            _CONTINUOUS,
            [ConstraintSpec(type=ConstraintType.SUM_LESS_THAN, parameters=["a", "b"], value=1.5)],
        )
        assert BayBEBackend().validate_capabilities(spec).is_compatible
        assert BoTorchBackend().validate_capabilities(spec).is_compatible


@pytest.mark.slow
class TestBehavioral:
    def test_no_label_duplicates_suggestions_are_distinct(self) -> None:
        """Suggestions honor DiscreteNoLabelDuplicatesConstraint end to end."""
        spec = _spec(
            _CATEGORICAL,
            [ConstraintSpec(type=ConstraintType.NO_LABEL_DUPLICATES, parameters=["a", "b"])],
            random_seed=3,
        )
        observations = [
            ObservationData(parameter_values={"a": "x", "b": "y"}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"a": "y", "b": "z"}, objective_values={"y": 0.5}),
        ]
        batch = BayBEBackend().generate_suggestions(
            spec=spec, observations=observations, batch_size=2, iteration=1
        )
        for suggestion in batch.suggestions:
            values = suggestion["parameter_values"]
            assert values["a"] != values["b"]

    def test_discrete_cardinality_bounds_nonzero_count(self) -> None:
        """Suggestions honor DiscreteCardinalityConstraint (at most 1 nonzero)."""
        spec = _spec(
            _DISCRETE,
            [
                ConstraintSpec(
                    type=ConstraintType.CARDINALITY,
                    parameters=["a", "b"],
                    max_cardinality=1,
                )
            ],
            random_seed=4,
        )
        observations = [
            ObservationData(parameter_values={"a": 0.0, "b": 1.0}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"a": 2.0, "b": 0.0}, objective_values={"y": 0.5}),
        ]
        batch = BayBEBackend().generate_suggestions(
            spec=spec, observations=observations, batch_size=2, iteration=1
        )
        for suggestion in batch.suggestions:
            values = suggestion["parameter_values"]
            nonzero = sum(1 for name in ("a", "b") if float(values[name]) != 0.0)
            assert nonzero <= 1


class TestCardinalityZeroPreconditions:
    """M-class: cardinality members must be able to take the value zero.

    BayBE's ``ContinuousCardinalityConstraint`` validates every member's
    bounds at search-space build for *any* cardinality bounds (an
    exception group when zero is excluded), so continuous members are
    checked unconditionally. A discrete grid without a zero value only
    breaks (filters to a silently empty subspace) when ``max_cardinality``
    can actually force a zero — max below the referenced-parameter count.
    """

    def test_continuous_bounds_excluding_zero_are_rejected(self) -> None:
        parameters = [
            ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.5, 1.0)),
            ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ]
        constraint = ConstraintSpec(
            type=ConstraintType.CARDINALITY, parameters=["a", "b"], max_cardinality=1
        )
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert not ok
        assert reason is not None
        assert "'a'" in reason
        assert "zero" in reason

    def test_discrete_grid_without_zero_is_rejected(self) -> None:
        parameters = [
            ParameterSpec(name="a", type=ParameterType.DISCRETE, values=[1.0, 2.0]),
            ParameterSpec(name="b", type=ParameterType.DISCRETE, values=[0.0, 1.0]),
        ]
        constraint = ConstraintSpec(
            type=ConstraintType.CARDINALITY, parameters=["a", "b"], max_cardinality=1
        )
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert not ok
        assert reason is not None
        assert "'a'" in reason
        assert "zero" in reason

    def test_zero_spanning_members_stay_supported(self) -> None:
        """Regression pin: the existing happy paths keep their SUPPORTED verdict."""
        continuous = ConstraintSpec(
            type=ConstraintType.CARDINALITY, parameters=["a", "b"], max_cardinality=1
        )
        ok, _reason = baybe_constraint_support(continuous, _CONTINUOUS)
        assert ok
        ok, _reason = baybe_constraint_support(continuous, _DISCRETE)
        assert ok

    def test_continuous_zero_excluded_bounds_rejected_even_without_forcing(self) -> None:
        """Continuous members need zero-spanning bounds for any max_cardinality.

        BayBE validates the bounds of every ``ContinuousCardinalityConstraint``
        member at search-space build regardless of whether the cardinality
        bounds can force a zero (verified against BayBE 0.15.0: min=1,
        max=2 over two zero-excluded parameters raises an exception group),
        so intake must reject the combination instead of deferring the
        crash to suggestion time.
        """
        parameters = [
            ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.5, 1.0)),
            ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0.5, 1.0)),
        ]
        constraint = ConstraintSpec(
            type=ConstraintType.CARDINALITY,
            parameters=["a", "b"],
            min_cardinality=1,
            max_cardinality=2,
        )
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert not ok
        assert reason is not None
        assert "zero" in reason
        with pytest.raises(ValueError, match="zero"):
            spec_to_constraints([constraint], parameters)

    def test_non_forcing_discrete_grid_without_zero_stays_supported(self) -> None:
        """Discrete grids without zero are fine while no zero can be forced.

        ``DiscreteCardinalityConstraint`` with max == parameter count never
        needs to zero a member, and BayBE builds the search space (verified
        against BayBE 0.15.0) — the unconditional rule applies to the
        continuous family only.
        """
        parameters = [
            ParameterSpec(name="a", type=ParameterType.DISCRETE, values=[1.0, 2.0]),
            ParameterSpec(name="b", type=ParameterType.DISCRETE, values=[1.0, 2.0]),
        ]
        constraint = ConstraintSpec(
            type=ConstraintType.CARDINALITY,
            parameters=["a", "b"],
            min_cardinality=1,
            max_cardinality=2,
        )
        ok, _reason = baybe_constraint_support(constraint, parameters)
        assert ok

    @pytest.mark.parametrize(
        ("min_cardinality", "max_cardinality"),
        [(0, 2), (None, 2), (0, None)],
    )
    def test_trivial_cardinality_bounds_are_rejected(
        self, min_cardinality: int | None, max_cardinality: int | None
    ) -> None:
        """min 0 with max == parameter count constrains nothing; BayBE refuses it.

        BayBE's ``CardinalityConstraint`` raises "No constraint ... is
        required" at construction for this combination (both families), so
        intake must classify it instead of deferring the ValueError to
        suggestion time.
        """
        constraint = ConstraintSpec(
            type=ConstraintType.CARDINALITY,
            parameters=["a", "b"],
            min_cardinality=min_cardinality,
            max_cardinality=max_cardinality,
        )
        for parameters in (_CONTINUOUS, _DISCRETE):
            ok, reason = baybe_constraint_support(constraint, parameters)
            assert not ok
            assert reason is not None
            assert "constrains nothing" in reason

    def test_bounds_only_discrete_grid_containing_zero_is_supported(self) -> None:
        parameters = [
            ParameterSpec(name="a", type=ParameterType.DISCRETE, bounds=(-1.0, 2.0)),
            ParameterSpec(name="b", type=ParameterType.DISCRETE, values=[0.0, 1.0]),
        ]
        constraint = ConstraintSpec(
            type=ConstraintType.CARDINALITY, parameters=["a", "b"], max_cardinality=1
        )
        ok, _reason = baybe_constraint_support(constraint, parameters)
        assert ok


class TestLinkedParameterDomainOverlap:
    """Linkage needs overlapping value domains, or the space filters to zero rows.

    ``DiscreteLinkedParametersConstraint`` keeps only rows whose
    referenced columns hold identical values (BayBE constraints
    userguide), so disjoint declared domains — or a numeric/label mix,
    which never compares equal — produce an empty discrete subspace and a
    guaranteed exhaustion failure at suggestion time. The capability
    layer must reject those shapes at intake.
    """

    def test_disjoint_categorical_domains_are_rejected(self) -> None:
        parameters = [
            ParameterSpec(name="a", type=ParameterType.CATEGORICAL, categories=["x", "y"]),
            ParameterSpec(name="b", type=ParameterType.CATEGORICAL, categories=["u", "v"]),
        ]
        constraint = ConstraintSpec(type=ConstraintType.LINKED_PARAMETERS, parameters=["a", "b"])
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert not ok
        assert reason is not None
        assert "disjoint" in reason
        with pytest.raises(ValueError, match="disjoint"):
            spec_to_constraints([constraint], parameters)

    def test_mixed_numeric_and_label_members_are_rejected(self) -> None:
        """A float grid value never equals a string label in the exp rep."""
        parameters = [
            ParameterSpec(name="a", type=ParameterType.DISCRETE, values=[1.0, 2.0]),
            ParameterSpec(name="b", type=ParameterType.CATEGORICAL, categories=["1", "2"]),
        ]
        constraint = ConstraintSpec(type=ConstraintType.LINKED_PARAMETERS, parameters=["a", "b"])
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert not ok
        assert reason is not None
        assert "numeric" in reason

    def test_overlapping_categorical_domains_build_the_intersection(self) -> None:
        """Partially overlapping domains stay supported and build non-empty."""
        from bo_engine_baybe.converters import spec_to_searchspace

        parameters = [
            ParameterSpec(name="a", type=ParameterType.CATEGORICAL, categories=["x", "y", "z"]),
            ParameterSpec(name="b", type=ParameterType.CATEGORICAL, categories=["y", "z", "w"]),
        ]
        constraint = ConstraintSpec(type=ConstraintType.LINKED_PARAMETERS, parameters=["a", "b"])
        ok, _reason = baybe_constraint_support(constraint, parameters)
        assert ok
        searchspace = spec_to_searchspace(_spec(parameters, [constraint]))
        exp = searchspace.discrete.exp_rep
        assert len(exp) == 2
        assert set(exp["a"]) == {"y", "z"}

    def test_overlapping_numeric_grids_stay_supported(self) -> None:
        parameters = [
            ParameterSpec(name="a", type=ParameterType.DISCRETE, values=[1.0, 2.0, 3.0]),
            ParameterSpec(name="b", type=ParameterType.DISCRETE, values=[2.0, 3.0, 4.0]),
        ]
        constraint = ConstraintSpec(type=ConstraintType.LINKED_PARAMETERS, parameters=["a", "b"])
        ok, _reason = baybe_constraint_support(constraint, parameters)
        assert ok

    def test_disjoint_domains_do_not_reject_the_vacuous_family_members(self) -> None:
        """no_label_duplicates / permutation_invariance are merely vacuous.

        Disjoint domains make those constraints filter nothing — the
        space stays intact, so they must not inherit the linkage
        rejection.
        """
        parameters = [
            ParameterSpec(name="a", type=ParameterType.CATEGORICAL, categories=["x", "y"]),
            ParameterSpec(name="b", type=ParameterType.CATEGORICAL, categories=["u", "v"]),
        ]
        for constraint_type in (
            ConstraintType.NO_LABEL_DUPLICATES,
            ConstraintType.PERMUTATION_INVARIANCE,
        ):
            constraint = ConstraintSpec(type=constraint_type, parameters=["a", "b"])
            ok, _reason = baybe_constraint_support(constraint, parameters)
            assert ok


class TestSetBasedEffectivePoolFeasibility:
    """Feasibility is judged on effective pools, not declared domains.

    BayBE builds the categorical/substance/custom product from the
    *active* values when ``active_values`` is set, and its
    no-label-duplicates / permutation-invariance constraints prune every
    row with a repeated value (permutation invariance additionally
    deduplicates permutations) — verified against BayBE 0.15.0. Declared
    domains that look valid can therefore still produce a 0-row search
    space, which must be an intake-time report rather than a first
    -suggestion exhaustion failure.
    """

    @staticmethod
    def _cat(name: str, categories: list[str], active: list[str] | None = None) -> ParameterSpec:
        options = {"baybe": {"active_values": active}} if active is not None else None
        return ParameterSpec(
            name=name,
            type=ParameterType.CATEGORICAL,
            categories=categories,
            parameter_options=options,
        )

    def test_linked_disjoint_active_values_are_rejected(self) -> None:
        """Declared domains overlap, active pools do not — reject at intake."""
        parameters = [
            self._cat("a", ["x", "y", "z"], active=["x"]),
            self._cat("b", ["x", "y", "z"], active=["y"]),
        ]
        constraint = ConstraintSpec(type=ConstraintType.LINKED_PARAMETERS, parameters=["a", "b"])
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert not ok
        assert reason is not None
        assert "disjoint" in reason
        with pytest.raises(ValueError, match="disjoint"):
            spec_to_constraints([constraint], parameters)

    def test_linked_disjoint_active_values_fail_capability_validation(self) -> None:
        """Backend-level proof: the failure is an intake report, not late exhaustion."""
        parameters = [
            self._cat("a", ["x", "y", "z"], active=["x"]),
            self._cat("b", ["x", "y", "z"], active=["y"]),
        ]
        constraint = ConstraintSpec(type=ConstraintType.LINKED_PARAMETERS, parameters=["a", "b"])
        result = BayBEBackend().validate_capabilities(_spec(parameters, [constraint]))
        assert not result.is_compatible

    def test_linked_overlapping_active_values_build_the_intersection(self) -> None:
        from bo_engine_baybe.converters import spec_to_searchspace

        parameters = [
            self._cat("a", ["x", "y", "z"], active=["x", "y"]),
            self._cat("b", ["x", "y", "z"], active=["y", "z"]),
        ]
        constraint = ConstraintSpec(type=ConstraintType.LINKED_PARAMETERS, parameters=["a", "b"])
        ok, _reason = baybe_constraint_support(constraint, parameters)
        assert ok
        searchspace = spec_to_searchspace(_spec(parameters, [constraint]))
        exp = searchspace.discrete.exp_rep
        assert exp.to_dict("records") == [{"a": "y", "b": "y"}]

    def test_no_label_duplicates_same_single_active_value_is_rejected(self) -> None:
        parameters = [
            self._cat("a", ["x", "y"], active=["x"]),
            self._cat("b", ["x", "y"], active=["x"]),
        ]
        constraint = ConstraintSpec(type=ConstraintType.NO_LABEL_DUPLICATES, parameters=["a", "b"])
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert not ok
        assert reason is not None
        assert "distinct" in reason
        with pytest.raises(ValueError, match="distinct"):
            spec_to_constraints([constraint], parameters)

    def test_no_label_duplicates_distinct_active_values_build(self) -> None:
        from bo_engine_baybe.converters import spec_to_searchspace

        parameters = [
            self._cat("a", ["x", "y"], active=["x"]),
            self._cat("b", ["x", "y"], active=["y"]),
        ]
        constraint = ConstraintSpec(type=ConstraintType.NO_LABEL_DUPLICATES, parameters=["a", "b"])
        ok, _reason = baybe_constraint_support(constraint, parameters)
        assert ok
        searchspace = spec_to_searchspace(_spec(parameters, [constraint]))
        assert len(searchspace.discrete.exp_rep) == 1

    def test_no_label_duplicates_declared_pools_smaller_than_members_rejected(self) -> None:
        """The hole exists without active_values: 3 slots over 2 shared labels."""
        parameters = [
            self._cat("p1", ["x", "y"]),
            self._cat("p2", ["x", "y"]),
            self._cat("p3", ["x", "y"]),
        ]
        constraint = ConstraintSpec(
            type=ConstraintType.NO_LABEL_DUPLICATES, parameters=["p1", "p2", "p3"]
        )
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert not ok
        assert reason is not None
        assert "distinct" in reason

    def test_permutation_invariance_same_single_active_value_is_rejected(self) -> None:
        parameters = [
            self._cat("a", ["x", "y"], active=["x"]),
            self._cat("b", ["x", "y"], active=["x"]),
        ]
        constraint = ConstraintSpec(
            type=ConstraintType.PERMUTATION_INVARIANCE, parameters=["a", "b"]
        )
        ok, reason = baybe_constraint_support(constraint, parameters)
        assert not ok
        assert reason is not None
        assert "distinct" in reason

    def test_permutation_invariance_exact_fit_stays_supported(self) -> None:
        """3 slots x 3 labels leaves exactly one all-distinct representative."""
        from bo_engine_baybe.converters import spec_to_searchspace

        parameters = [self._cat(n, ["x", "y", "z"]) for n in ("p1", "p2", "p3")]
        constraint = ConstraintSpec(
            type=ConstraintType.PERMUTATION_INVARIANCE, parameters=["p1", "p2", "p3"]
        )
        ok, _reason = baybe_constraint_support(constraint, parameters)
        assert ok
        searchspace = spec_to_searchspace(_spec(parameters, [constraint]))
        assert len(searchspace.discrete.exp_rep) == 1


class TestPermutationInvarianceLabelDedup:
    """Documented BayBE side effect: equal-label rows are filtered too.

    Row-count pin on a small grid (verified against BayBE 0.15.0): for
    two 3-category slots, the constraint keeps the 3 unordered pairs of
    distinct labels — permutation twins *and* the 3 equal-label rows are
    dropped.
    """

    def test_row_count_on_small_grid(self) -> None:
        import itertools

        import pandas as pd

        constraint = DiscretePermutationInvarianceConstraint(parameters=["a", "b"])
        frame = pd.DataFrame([{"a": a, "b": b} for a, b in itertools.product("xyz", repeat=2)])
        invalid = constraint.get_invalid(frame)
        assert len(frame) - len(invalid) == 3
        kept = frame.drop(index=invalid)
        assert not (kept["a"] == kept["b"]).any()
