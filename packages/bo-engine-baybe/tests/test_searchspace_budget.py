"""Large-categorical safeguard: budgeting, deterministic subsampling, warnings.

Pins the branch decision (below budget → ``from_product`` byte-for-byte;
above → bounded deterministic subsample), the sampler's determinism /
dedup / constraint-filtering / bounded-infeasibility contracts, the
byte-budget row derivation, parameter-object preservation, observed-row
union-in, and the DEGRADED capability report that keeps ``backend="auto"``
away from a subsampled BayBE run.

References: BayBE search-space userguide
(https://emdgroup.github.io/baybe/stable/userguide/searchspace.html —
``estimate_product_space_size`` / explicit candidate lists) and the
PostgreSQL protocol message-length limit
(https://www.postgresql.org/docs/current/protocol-message-formats.html)
behind the byte budget.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import pytest

from bo_engine.backend_base import CapabilityStatus
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
from bo_engine_baybe.constants import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_SEARCHSPACE_STATE_BYTES,
    MIN_VIABLE_SUBSAMPLE_CANDIDATES,
)
from bo_engine_baybe.converters import (
    resolve_searchspace_budget,
    spec_to_constraints,
    spec_to_parameters,
    spec_to_searchspace,
)
from bo_engine_baybe.searchspace_budget import (
    sample_discrete_candidates,
)
from bo_engine_baybe.state import _build_campaign, _rebuild_from_observations

if TYPE_CHECKING:
    from baybe.parameters.base import DiscreteParameter


def _discrete_params(spec: OptimizationSpec) -> list[DiscreteParameter]:
    """Build the spec's BayBE parameters, narrowed to the discrete family."""
    from baybe.parameters.base import DiscreteParameter

    return [p for p in spec_to_parameters(spec) if isinstance(p, DiscreteParameter)]


def _spec(
    parameters: list[ParameterSpec],
    constraints: list[ConstraintSpec] | None = None,
    baybe_options: dict[str, Any] | None = None,
) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=parameters,
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        constraints=constraints or [],
        backend_options={"baybe": baybe_options} if baybe_options is not None else None,
    )


def _grid(name: str, n: int) -> ParameterSpec:
    return ParameterSpec(
        name=name, type=ParameterType.DISCRETE, values=[float(i) for i in range(n)]
    )


def _cats(name: str, n: int) -> ParameterSpec:
    return ParameterSpec(
        name=name, type=ParameterType.CATEGORICAL, categories=[f"{name}_{i}" for i in range(n)]
    )


_SMALL_SPEC = _spec([_grid("a", 4), _cats("c", 3)])
# Three 500-category parameters: 1.25e8 combinations, OHE width 1500 —
# the incident shape at reduced scale.
_HUGE_SPEC = _spec([_cats("c1", 500), _cats("c2", 500), _cats("c3", 500)])


class TestPathPinning:
    def test_small_spec_builds_full_product(self) -> None:
        """Below-threshold specs keep the exact from_product grid (prime directive)."""
        searchspace = spec_to_searchspace(_SMALL_SPEC)
        assert len(searchspace.discrete.exp_rep) == 4 * 3

    def test_small_spec_budget_is_below_default(self) -> None:
        """Default-threshold sanity: representative small fixtures stay on the old path."""
        budget = resolve_searchspace_budget(_SMALL_SPEC)
        assert not budget.exceeds_budget
        assert budget.estimated_state_bytes < DEFAULT_MAX_SEARCHSPACE_STATE_BYTES

    def test_huge_spec_exceeds_budget(self) -> None:
        budget = resolve_searchspace_budget(_HUGE_SPEC)
        assert budget.exceeds_budget
        assert budget.n_candidates <= DEFAULT_MAX_CANDIDATES


class TestBranchDecision:
    def test_row_cap_triggers_even_when_bytes_fit(self) -> None:
        """The candidate cap is a second, independent trigger."""
        spec = _spec([_grid("a", 200), _grid("b", 100)])  # 20k rows, narrow comp rep
        budget = resolve_searchspace_budget(spec)
        assert budget.exceeds_budget
        assert budget.n_combinations == 20_000

    def test_byte_budget_derives_fewer_rows_for_wide_ohe(self) -> None:
        """Bound bytes, not rows: wide OHE reduces the row budget proportionally."""
        narrow = resolve_searchspace_budget(_spec([_grid("a", 200), _grid("b", 100)]))
        wide = resolve_searchspace_budget(_HUGE_SPEC)
        assert wide.n_candidates < narrow.n_candidates

    def test_option_overrides_lower_the_threshold(self) -> None:
        spec = _spec([_grid("a", 30), _grid("b", 30)], baybe_options={"max_candidates": 100})
        budget = resolve_searchspace_budget(spec)
        assert budget.exceeds_budget
        assert budget.n_candidates <= 100


class TestSampler:
    def test_determinism_across_calls(self) -> None:
        params = _discrete_params(_HUGE_SPEC)
        first = sample_discrete_candidates(_HUGE_SPEC, params, 50)
        second = sample_discrete_candidates(_HUGE_SPEC, params, 50)
        assert first.equals(second)

    def test_no_duplicate_rows(self) -> None:
        params = _discrete_params(_HUGE_SPEC)
        frame = sample_discrete_candidates(_HUGE_SPEC, params, 200)
        assert not frame.duplicated().any()

    def test_no_materialization_guard(self) -> None:
        """~1e12 combinations sample in seconds with exactly the budgeted rows.

        This is the test that would have caught the runtime half of the
        original incident (the ~190 s enumerate-everything recommend).
        """
        spec = _spec([_cats("c1", 10_000), _cats("c2", 10_000), _grid("g", 10_000)])
        started = time.monotonic()
        searchspace = spec_to_searchspace(spec)
        elapsed = time.monotonic() - started
        assert elapsed < 10.0
        assert len(searchspace.discrete.exp_rep) <= DEFAULT_MAX_CANDIDATES

    def test_constraint_filtering_uses_baybe_get_invalid(self) -> None:
        spec = _spec(
            [_grid("a", 200), _grid("b", 100)],
            constraints=[
                ConstraintSpec(type=ConstraintType.SUM_LESS_THAN, parameters=["a", "b"], value=50.0)
            ],
        )
        params = _discrete_params(spec)
        constraints = spec_to_constraints(list(spec.constraints), list(spec.parameters))
        frame = sample_discrete_candidates(spec, params, 100, baybe_constraints=constraints)
        assert ((frame["a"] + frame["b"]) <= 50.0).all()

    def test_near_infeasible_constraints_raise_typed_error(self) -> None:
        spec = _spec(
            [_grid("a", 200), _grid("b", 100)],
            constraints=[
                ConstraintSpec(type=ConstraintType.SUM_LESS_THAN, parameters=["a", "b"], value=-1.0)
            ],
        )
        params = _discrete_params(spec)
        constraints = spec_to_constraints(list(spec.constraints), list(spec.parameters))
        with pytest.raises(ValueError, match="feasible candidate"):
            sample_discrete_candidates(spec, params, 100, baybe_constraints=constraints)

    def test_minimum_viable_constant_is_positive(self) -> None:
        assert MIN_VIABLE_SUBSAMPLE_CANDIDATES > 0

    def test_tiny_space_under_tiny_budget_builds_instead_of_raising(self) -> None:
        """A whole space smaller than the viability floor is enumerable, not infeasible.

        With a byte budget of 1, a 3x3 grid (9 combinations, below
        ``MIN_VIABLE_SUBSAMPLE_CANDIDATES``) takes the subsample branch
        with a candidate target equal to the full space. The
        near-infeasibility floor guards constraint-driven sampling
        starvation only — an unconstrained tiny space must build (parity
        with the ``from_product`` result), not raise.
        """
        spec = _spec(
            [_grid("a", 3), _grid("b", 3)],
            baybe_options={"max_searchspace_state_bytes": 1},
        )
        assert resolve_searchspace_budget(spec).exceeds_budget
        searchspace = spec_to_searchspace(spec)
        assert 0 < len(searchspace.discrete.exp_rep) <= 9


class TestParameterPreservation:
    def test_int_encoding_survives_subsampling(self) -> None:
        """The subsampled subspace keeps the configured parameter objects."""
        spec = _spec(
            [
                ParameterSpec(
                    name="c1",
                    type=ParameterType.CATEGORICAL,
                    categories=[f"v{i}" for i in range(300)],
                    parameter_options={"baybe": {"encoding": "INT"}},
                ),
                _cats("c2", 300),
            ]
        )
        searchspace = spec_to_searchspace(spec)
        by_name = {p.name: p for p in searchspace.discrete.parameters}
        encoding = by_name["c1"].encoding
        assert encoding is not None
        assert encoding.value == "INT"

    def test_union_in_observed_and_pending_rows(self) -> None:
        observations = [
            ObservationData(
                parameter_values={"c1": "c1_7", "c2": "c2_8", "c3": "c3_9"},
                objective_values={"y": 1.0},
            )
        ]
        pending = [{"c1": "c1_17", "c2": "c2_18", "c3": "c3_19"}]
        searchspace = spec_to_searchspace(_HUGE_SPEC, observations, pending)
        exp = searchspace.discrete.exp_rep
        observed_present = (
            (exp["c1"] == "c1_7") & (exp["c2"] == "c2_8") & (exp["c3"] == "c3_9")
        ).any()
        pending_present = (
            (exp["c1"] == "c1_17") & (exp["c2"] == "c2_18") & (exp["c3"] == "c3_19")
        ).any()
        assert observed_present
        assert pending_present

    def test_measured_rows_marked_in_metadata_ledger(self) -> None:
        """The ledger marks unioned-in measured rows (already-measured contract)."""
        from baybe.campaign import _MEASURED

        from bo_engine_baybe.converters import observations_to_dataframe
        from bo_engine_baybe.state import _add_measurements

        observations = [
            ObservationData(
                parameter_values={"c1": "c1_7", "c2": "c2_8", "c3": "c3_9"},
                objective_values={"y": 1.0},
            )
        ]
        campaign = _build_campaign(_HUGE_SPEC, observations)
        _add_measurements(campaign, observations_to_dataframe(observations, _HUGE_SPEC), _HUGE_SPEC)
        assert campaign._searchspace_metadata[_MEASURED].sum() == 1


class TestRebuildDeterminism:
    def test_rebuild_regenerates_identical_candidate_set(self) -> None:
        observations = [
            ObservationData(
                parameter_values={"c1": "c1_1", "c2": "c2_2", "c3": "c3_3"},
                objective_values={"y": 0.5},
            )
        ]
        first = _rebuild_from_observations(_HUGE_SPEC, observations)
        second = _rebuild_from_observations(_HUGE_SPEC, observations)
        left = first.searchspace.discrete.exp_rep.sort_values(["c1", "c2", "c3"], ignore_index=True)
        right = second.searchspace.discrete.exp_rep.sort_values(
            ["c1", "c2", "c3"], ignore_index=True
        )
        assert left.equals(right)


class TestRoutingAndWarnings:
    def test_above_budget_reports_degraded(self) -> None:
        result = BayBEBackend().validate_capabilities(_HUGE_SPEC)
        assert result.is_compatible  # explicit backend='baybe' still runs
        reports = [r for r in result.option_reports if r.key == "searchspace_size"]
        assert reports
        assert reports[0].status == CapabilityStatus.DEGRADED

    def test_small_spec_has_no_budget_report(self) -> None:
        """Routing-safety: small-spec routing stays byte-identical."""
        result = BayBEBackend().validate_capabilities(_SMALL_SPEC)
        assert all(r.key != "searchspace_size" for r in result.option_reports)

    @pytest.mark.slow
    def test_generate_on_subsampled_space_is_bounded_and_warns(self) -> None:
        """End-to-end at reduced scale: warns, bounded time, bounded state.

        Thresholds are lowered via backend_options so the reporter's shape
        runs in CI seconds; the persisted state must stay under the byte
        budget it was derived from (plus estimate slack), proving the
        original 1.34 GB / >1 GiB-protocol failure class is closed.
        """
        byte_budget = 512 * 1024
        spec = OptimizationSpec(
            parameters=[_cats("c1", 60), _cats("c2", 60), _cats("c3", 60)],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            backend_options={"baybe": {"max_searchspace_state_bytes": byte_budget}},
            random_seed=9,
        )
        observations = [
            ObservationData(
                parameter_values={"c1": "c1_1", "c2": "c2_2", "c3": "c3_3"},
                objective_values={"y": 1.0},
            ),
            ObservationData(
                parameter_values={"c1": "c1_4", "c2": "c2_5", "c3": "c3_6"},
                objective_values={"y": 0.5},
            ),
        ]
        started = time.monotonic()
        batch = BayBEBackend().generate_suggestions(
            spec=spec, observations=observations, batch_size=1, iteration=1
        )
        assert time.monotonic() - started < 60.0
        assert any("subsampled" in w for w in batch.warnings)
        assert batch.method_info.get("searchspace_subsampled") is True
        assert batch.backend_state is not None
        state_bytes = len(json.dumps(batch.backend_state))
        # The state carries the campaign plus envelope overhead; it must sit
        # far below the incident scale and in the region the budget targets.
        assert state_bytes < 4 * byte_budget


class TestDeclaredTypeDtypes:
    """C-class: pool dtypes come from the declared parameter type.

    Numeric-string categorical labels (IDs, CAS-like codes) must stay
    object-dtype labels — try-float-first coercion turned them into a
    float column that BayBE's ``normalize_input_dtypes`` never converts
    back for categorical parameters, crashing the subspace merge (no
    observations) or silently producing an almost-all-NaN computational
    representation (observations unioned in).
    """

    @staticmethod
    def _numeric_string_spec() -> OptimizationSpec:
        return _spec(
            [
                ParameterSpec(
                    name="c1",
                    type=ParameterType.CATEGORICAL,
                    categories=[str(i) for i in range(300)],
                ),
                ParameterSpec(
                    name="c2",
                    type=ParameterType.CATEGORICAL,
                    categories=[str(i) for i in range(300)],
                ),
            ]
        )

    def test_build_succeeds_without_observations(self) -> None:
        spec = self._numeric_string_spec()
        assert resolve_searchspace_budget(spec).exceeds_budget
        searchspace = spec_to_searchspace(spec)
        assert len(searchspace.discrete.exp_rep) > 0

    def test_comp_rep_has_no_nan_with_observations_unioned(self) -> None:
        spec = self._numeric_string_spec()
        observations = [
            ObservationData(parameter_values={"c1": "7", "c2": "8"}, objective_values={"y": 1.0})
        ]
        searchspace = spec_to_searchspace(spec, observations)
        comp = searchspace.discrete.comp_rep
        assert not comp.isna().any().any()
        exp = searchspace.discrete.exp_rep
        observed = (exp["c1"] == "7") & (exp["c2"] == "8")
        assert int(observed.sum()) == 1

    def test_exp_rep_dtypes_match_from_product_sibling(self) -> None:
        """Subsample dtypes equal the from_product dtypes of a small sibling."""
        sibling = _spec(
            [
                ParameterSpec(
                    name="c1", type=ParameterType.CATEGORICAL, categories=["0", "1", "2"]
                ),
                ParameterSpec(
                    name="c2", type=ParameterType.CATEGORICAL, categories=["0", "1", "2"]
                ),
            ]
        )
        product_dtypes = spec_to_searchspace(sibling).discrete.exp_rep.dtypes
        sampled_dtypes = spec_to_searchspace(self._numeric_string_spec()).discrete.exp_rep.dtypes
        assert list(sampled_dtypes) == list(product_dtypes)


class TestCrossRowConstraintFiltering:
    """M-class: permutation invariance holds across top-up rounds and union-in."""

    @staticmethod
    def _perm_spec() -> OptimizationSpec:
        # Three slots over one shared 12-label pool: 12^3 = 1728
        # combinations but only C(12,3) = 220 canonical rows (BayBE keeps
        # one order per multiset and drops equal-label rows), so the
        # sampler can never fill a 500-row budget and the top-up loop is
        # guaranteed to run multiple rounds.
        labels = [f"v{i}" for i in range(12)]
        return _spec(
            [
                ParameterSpec(name=slot, type=ParameterType.CATEGORICAL, categories=labels)
                for slot in ("s1", "s2", "s3")
            ],
            constraints=[
                ConstraintSpec(
                    type=ConstraintType.PERMUTATION_INVARIANCE,
                    parameters=["s1", "s2", "s3"],
                )
            ],
            baybe_options={"max_candidates": 500},
        )

    def test_multi_round_frame_contains_no_constraint_violations(self) -> None:
        spec = self._perm_spec()
        params = _discrete_params(spec)
        constraints = spec_to_constraints(list(spec.constraints), list(spec.parameters))
        frame = sample_discrete_candidates(spec, params, 500, baybe_constraints=constraints)
        assert constraints is not None
        invalid = constraints[0].get_invalid(frame)
        assert len(invalid) == 0
        # All 220 canonical representatives are reachable and no more.
        assert len(frame) <= 220

    def test_unioned_observed_row_wins_as_canonical_representative(self) -> None:
        spec = self._perm_spec()
        params = _discrete_params(spec)
        constraints = spec_to_constraints(list(spec.constraints), list(spec.parameters))
        observed = {"s1": "v5", "s2": "v3", "s3": "v1"}
        observations = [ObservationData(parameter_values=observed, objective_values={"y": 1.0})]
        frame = sample_discrete_candidates(
            spec, params, 500, baybe_constraints=constraints, observations=observations
        )
        exact = (frame["s1"] == "v5") & (frame["s2"] == "v3") & (frame["s3"] == "v1")
        assert int(exact.sum()) == 1
        assert constraints is not None
        assert len(constraints[0].get_invalid(frame)) == 0
        # No other permutation of the observed multiset survives.
        multiset = frame[["s1", "s2", "s3"]].apply(lambda r: frozenset(r), axis=1)
        assert int((multiset == frozenset({"v5", "v3", "v1"})).sum()) == 1


class TestPrescreenSoundness:
    """M-class: the cheap prescreen is a sound upper bound or abstains."""

    def test_long_label_spec_makes_prescreen_abstain(self) -> None:
        """Labels beyond the fixed per-cell constant must not fake 'below budget'."""
        from bo_engine_baybe.searchspace_budget import cheap_budget_prescreen

        long_labels = ["x" * 1000 + f"_{i}" for i in range(50)]
        spec = _spec(
            [
                ParameterSpec(name="c1", type=ParameterType.CATEGORICAL, categories=long_labels),
                ParameterSpec(name="c2", type=ParameterType.CATEGORICAL, categories=long_labels),
            ],
            baybe_options={"max_searchspace_state_bytes": 5_000_000},
        )
        precise = resolve_searchspace_budget(spec)
        assert precise.exceeds_budget
        prescreen = cheap_budget_prescreen(spec)
        assert prescreen is None or prescreen.exceeds_budget == precise.exceeds_budget

    def test_fingerprint_kwargs_make_prescreen_abstain(self) -> None:
        """kwargs_fingerprint can exceed any fixed comp-width bound."""
        from bo_engine_baybe.searchspace_budget import cheap_budget_prescreen

        spec = _spec(
            [
                ParameterSpec(
                    name="mol",
                    type=ParameterType.CATEGORICAL,
                    categories=["water", "ethanol"],
                    parameter_options={
                        "baybe": {
                            "role": "substance",
                            "substance_data": {"water": "O", "ethanol": "CCO"},
                            "kwargs_fingerprint": {"fp_size": 16384},
                        }
                    },
                ),
                _cats("c", 3),
            ]
        )
        assert cheap_budget_prescreen(spec) is None

    def test_wide_unicode_labels_keep_prescreen_sound(self) -> None:
        """PEP 393 wide strings cost 2-4 bytes per character, not one.

        BayBE's precise estimate measures actual deep bytes
        (``sys.getsizeof`` per label), so a character-count per-cell bound
        under-estimates non-Latin-1 labels and the prescreen could claim
        "below budget" while construction subsamples — silently dropping
        the warning, the DEGRADED report, and the exhaustion marker. INT
        encoding keeps the computational width tiny so the label bytes
        dominate, which is exactly the shape that breaks a character-count
        bound.
        """
        from bo_engine_baybe.searchspace_budget import cheap_budget_prescreen

        labels = ["\U0001f9ea" * 40 + f"_{i}" for i in range(60)]
        parameters = [
            ParameterSpec(
                name=name,
                type=ParameterType.CATEGORICAL,
                categories=labels,
                parameter_options={"baybe": {"encoding": "INT"}},
            )
            for name in ("c1", "c2")
        ]
        unconstrained = resolve_searchspace_budget(_spec(parameters))
        tight = _spec(
            parameters,
            baybe_options={
                "max_searchspace_state_bytes": int(unconstrained.estimated_state_bytes * 0.9)
            },
        )
        precise = resolve_searchspace_budget(tight)
        assert precise.exceeds_budget
        prescreen = cheap_budget_prescreen(tight)
        assert prescreen is None or prescreen.exceeds_budget

    def test_prescreen_agrees_with_precise_estimate_on_suite_fixtures(self) -> None:
        """Property-style agreement: prescreen abstains or matches the decision."""
        from bo_engine_baybe.searchspace_budget import cheap_budget_prescreen

        fixtures = [
            _SMALL_SPEC,
            _spec([_grid("a", 4), _cats("c", 3)]),
            _spec([_cats("c1", 60), _cats("c2", 60)]),
            _spec([_grid("a", 10), _grid("b", 10), _cats("c", 5)]),
        ]
        for spec in fixtures:
            precise = resolve_searchspace_budget(spec)
            prescreen = cheap_budget_prescreen(spec)
            assert prescreen is None or prescreen.exceeds_budget == precise.exceeds_budget

    def test_suite_fixtures_stay_below_default_budget(self) -> None:
        """Every representative below-budget fixture keeps the from_product path."""
        fixtures = [
            _SMALL_SPEC,
            _spec([_grid("a", 4), _cats("c", 3)]),
            _spec([_grid("a", 10), _grid("b", 10), _cats("c", 5)]),
            _spec([_cats("c1", 10), _cats("c2", 10)]),
        ]
        for spec in fixtures:
            assert not resolve_searchspace_budget(spec).exceeds_budget


class TestUnionInPoolMembership:
    """Minor: out-of-pool union-in rows are dropped instead of NaN-poisoning."""

    def test_out_of_pool_observed_value_is_dropped(self) -> None:
        observations = [
            ObservationData(
                parameter_values={"c1": "NOT_A_LABEL", "c2": "c2_1", "c3": "c3_1"},
                objective_values={"y": 1.0},
            )
        ]
        searchspace = spec_to_searchspace(_HUGE_SPEC, observations)
        exp = searchspace.discrete.exp_rep
        assert not (exp["c1"] == "NOT_A_LABEL").any()
        assert not searchspace.discrete.comp_rep.isna().any().any()

    def test_numeric_label_observation_matches_string_categories(self) -> None:
        """A label submitted as a JSON number matches its declared string category.

        Union-in rows are coerced to the declared pool dtypes before
        membership matching, so the raw scalar ``7`` joins the frame as the
        canonical label ``"7"`` (one occurrence, no NaN columns) instead of
        being dropped as out-of-pool.
        """
        numeric_labels = [str(i) for i in range(300)]
        spec = _spec(
            [
                ParameterSpec(name="c1", type=ParameterType.CATEGORICAL, categories=numeric_labels),
                ParameterSpec(name="c2", type=ParameterType.CATEGORICAL, categories=numeric_labels),
            ]
        )
        observations = [
            ObservationData(parameter_values={"c1": 7, "c2": 11}, objective_values={"y": 1.0})
        ]
        searchspace = spec_to_searchspace(spec, observations)
        exp = searchspace.discrete.exp_rep
        assert int(((exp["c1"] == "7") & (exp["c2"] == "11")).sum()) == 1
        assert not searchspace.discrete.comp_rep.isna().any().any()

    def test_within_tolerance_observation_snaps_to_grid_value(self) -> None:
        """A measurement inside the declared tolerance joins as its grid value.

        Mirrors BayBE's own tolerance-based measurement matching
        (``NumericalDiscreteParameter.tolerance``, parameters userguide:
        https://emdgroup.github.io/baybe/stable/userguide/parameters.html):
        the canonical grid row must enter the candidate frame so the
        metadata ledger can mark it measured.
        """
        spec = _spec(
            [
                ParameterSpec(
                    name="a",
                    type=ParameterType.DISCRETE,
                    values=[float(i) for i in range(200)],
                    parameter_options={"baybe": {"tolerance": 0.05}},
                ),
                _grid("b", 200),
            ]
        )
        observations = [
            ObservationData(parameter_values={"a": 7.004, "b": 11.0}, objective_values={"y": 1.0})
        ]
        searchspace = spec_to_searchspace(spec, observations)
        exp = searchspace.discrete.exp_rep
        assert int(((exp["a"] == 7.0) & (exp["b"] == 11.0)).sum()) == 1
        assert not (exp["a"] == 7.004).any()

    def test_beyond_tolerance_observation_is_dropped(self) -> None:
        """A measurement outside the declared tolerance stays out of the frame."""
        spec = _spec(
            [
                ParameterSpec(
                    name="a",
                    type=ParameterType.DISCRETE,
                    values=[float(i) for i in range(200)],
                    parameter_options={"baybe": {"tolerance": 0.05}},
                ),
                _grid("b", 200),
            ]
        )
        observations = [
            ObservationData(parameter_values={"a": 7.4, "b": 11.0}, objective_values={"y": 1.0})
        ]
        searchspace = spec_to_searchspace(spec, observations)
        exp = searchspace.discrete.exp_rep
        assert not (exp["a"] == 7.4).any()
        assert not searchspace.discrete.comp_rep.isna().any().any()


class TestSeedHelperRobustness:
    """Minor: near-identical constraints must not break the canonical sort."""

    def test_optional_fields_do_not_raise_in_seed_derivation(self) -> None:
        from bo_engine_baybe.searchspace_budget import _spec_content_seed

        spec = _spec(
            [_grid("a", 4), _grid("b", 4)],
            constraints=[
                ConstraintSpec(
                    type=ConstraintType.LINEAR,
                    parameters=["a", "b"],
                    value=3.0,
                    coefficients=[1.0, 2.0],
                ),
                ConstraintSpec(type=ConstraintType.LINEAR, parameters=["a", "b"], value=3.0),
                ConstraintSpec(
                    type=ConstraintType.CARDINALITY,
                    parameters=["a", "b"],
                    max_cardinality=1,
                ),
                ConstraintSpec(
                    type=ConstraintType.CARDINALITY,
                    parameters=["a", "b"],
                    min_cardinality=1,
                    max_cardinality=1,
                ),
            ],
        )
        assert isinstance(_spec_content_seed(spec), int)


class TestBudgetOptionValidation:
    """Minor: invalid values for the two budget overrides are rejected at intake."""

    @pytest.mark.parametrize("field", ["max_searchspace_state_bytes", "max_candidates"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_values_are_rejected(self, field: str, value: int) -> None:
        import pydantic

        from bo_engine_baybe.options import BayBEBackendOptions

        with pytest.raises(pydantic.ValidationError):
            BayBEBackendOptions.model_validate({field: value})

    @pytest.mark.parametrize("field", ["max_searchspace_state_bytes", "max_candidates"])
    def test_invalid_values_surface_as_capability_report(self, field: str) -> None:
        spec = _spec([_grid("a", 4)], baybe_options={field: 0})
        result = BayBEBackend().validate_capabilities(spec)
        assert not result.is_compatible
        assert any(r.key == "backend_options.baybe" for r in result.unsupported)


@pytest.mark.slow
class TestSubsampledStateRoundTrip:
    def test_restore_then_generate_again_keeps_the_candidate_set(self) -> None:
        """Restore → generate-again identity on the subsampled path.

        The persisted campaign state must restore to the *same* bounded
        candidate set (and produce another batch) instead of silently
        rebuilding a different subsample per iteration.
        """
        spec = OptimizationSpec(
            parameters=[_cats("c1", 60), _cats("c2", 60), _cats("c3", 60)],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            backend_options={"baybe": {"max_candidates": 300}},
            random_seed=21,
        )
        observations = [
            ObservationData(
                parameter_values={"c1": "c1_1", "c2": "c2_2", "c3": "c3_3"},
                objective_values={"y": 1.0},
            ),
            ObservationData(
                parameter_values={"c1": "c1_4", "c2": "c2_5", "c3": "c3_6"},
                objective_values={"y": 0.5},
            ),
        ]
        backend = BayBEBackend()
        first = backend.generate_suggestions(
            spec=spec, observations=observations, batch_size=1, iteration=1
        )
        assert first.method_info.get("searchspace_subsampled") is True

        from bo_engine_baybe.state import _restore_or_build_campaign

        inner = backend.unwrap_state(first.backend_state)
        restored = _restore_or_build_campaign(spec, inner, observations=observations)
        rebuilt = _restore_or_build_campaign(spec, None, observations=observations)
        sort_cols = ["c1", "c2", "c3"]
        left = restored.searchspace.discrete.exp_rep.sort_values(sort_cols, ignore_index=True)
        right = rebuilt.searchspace.discrete.exp_rep.sort_values(sort_cols, ignore_index=True)
        assert left.equals(right)

        second = backend.generate_suggestions(
            spec=spec,
            observations=observations,
            batch_size=1,
            iteration=2,
            backend_state=first.backend_state,
        )
        assert len(second.suggestions) == 1


@pytest.mark.nightly
class TestSubsampledOptimizationQuality:
    def test_subsampled_baybe_beats_random_on_average(self) -> None:
        """Statistical: BO on the bounded subsample outperforms random sampling.

        Mean best-so-far after a short campaign must beat a random
        baseline drawn from the same subsampled candidate pool (multiple
        seeds; mean comparison per the stochastic-test guidance in
        TESTING.md).
        """
        import numpy as np

        def objective_value(row: dict[str, Any]) -> float:
            return float(
                (int(str(row["c1"]).split("_")[1]) - 20) ** 2
                + (int(str(row["c2"]).split("_")[1]) - 20) ** 2
            )

        seeds = [3, 5, 7]
        n_iterations = 6
        bo_bests: list[float] = []
        random_bests: list[float] = []
        for seed in seeds:
            spec = OptimizationSpec(
                parameters=[_cats("c1", 60), _cats("c2", 60)],
                objectives=[ObjectiveSpec(name="y", minimize=True)],
                backend_options={"baybe": {"max_candidates": 400}},
                random_seed=seed,
            )
            backend = BayBEBackend()
            observations: list[ObservationData] = []
            state: dict[str, Any] | None = None
            for iteration in range(1, n_iterations + 1):
                batch = backend.generate_suggestions(
                    spec=spec,
                    observations=observations,
                    batch_size=1,
                    iteration=iteration,
                    backend_state=state,
                )
                state = batch.backend_state
                values = batch.suggestions[0]["parameter_values"]
                observations.append(
                    ObservationData(
                        parameter_values=dict(values),
                        objective_values={"y": objective_value(values)},
                    )
                )
            bo_bests.append(min(o.objective_values["y"] for o in observations))

            rng = np.random.default_rng(seed)
            pool = spec_to_searchspace(spec).discrete.exp_rep
            picks = pool.iloc[rng.choice(len(pool), size=n_iterations, replace=False)]
            random_bests.append(min(objective_value(dict(row)) for _, row in picks.iterrows()))

        assert float(np.mean(bo_bests)) <= float(np.mean(random_bests))


class TestExhaustionSubsampleMarker:
    """The exhaustion error carries the subsample context for the E011 envelope."""

    def test_subsampled_campaign_marks_the_error(self) -> None:
        from bo_engine_baybe.backend import _search_space_exhausted_error

        campaign = _build_campaign(_HUGE_SPEC)
        err = _search_space_exhausted_error(_HUGE_SPEC, campaign, batch_size=1)
        budget = resolve_searchspace_budget(_HUGE_SPEC)
        assert err.subsampled is True
        assert err.n_full_combinations == budget.n_combinations
        assert err.max_candidates == budget.max_candidates
        assert err.n_total_combinations == len(campaign.searchspace.discrete.exp_rep)

    def test_below_budget_campaign_is_unmarked(self) -> None:
        from bo_engine_baybe.backend import _search_space_exhausted_error

        campaign = _build_campaign(_SMALL_SPEC)
        err = _search_space_exhausted_error(_SMALL_SPEC, campaign, batch_size=1)
        assert err.subsampled is False
        assert err.n_full_combinations is None


class TestPendingExclusionMask:
    """Exhaustion accounting stays aligned when pending rows repeat."""

    def test_duplicated_pending_rows_mark_exactly_their_candidates(self) -> None:
        """A duplicated pending row must not fan out or shift the mask.

        The mask is built by a left merge against ``exp_rep``; a repeated
        pending row would multiply the merge output beyond the candidate
        count and an index-aligned OR would then mark the wrong rows.
        """
        import pandas as pd

        from bo_engine_baybe.backend import _excluded_candidate_mask

        campaign = _build_campaign(_SMALL_SPEC)
        exp = campaign.searchspace.discrete.exp_rep
        pending_row = exp.iloc[[0]]
        pending = pd.concat([pending_row, pending_row], ignore_index=True)
        mask = _excluded_candidate_mask(campaign, pending)
        assert len(mask) == len(exp)
        assert bool(mask.iloc[0]) is True
        assert int(mask.sum()) == 1
