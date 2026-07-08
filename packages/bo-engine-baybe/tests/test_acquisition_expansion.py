"""Expanded acquisition-method mapping on the BayBE backend.

The neutral :class:`bo_engine.types.AcquisitionMethod` now spans the full
BayBE 0.15 acquisition family (UCB, PI, posterior statistics, Thompson
sampling, knowledge gradient, active learning / qNIPV, and the explicit
non-log variants). These tests pin the dispatch-table completeness for
both objective families, the abbreviation validity against the installed
BayBE (the ``BayBESubstanceEncoding`` pin-guard precedent), and the UCB
``acquisition_beta`` wiring.

References: BayBE acquisition userguide
(https://emdgroup.github.io/baybe/stable/userguide/acquisition.html) for
the abbreviation set, and ``baybe.acquisition.acqfs`` for the class
constraints (e.g. ``qTS.supports_batching`` is ``False``).
"""

from __future__ import annotations

from typing import Any

import pytest
from baybe import acquisition as baybe_acquisition
from baybe.acquisition import qUCB

from bo_engine.types import (
    AcquisitionMethod,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.backend import BayBEBackend
from bo_engine_baybe.converters import (
    _MULTI_OBJECTIVE_ACQF_MAP,
    _SINGLE_OBJECTIVE_ACQF_MAP,
    BAYBE_UNSUPPORTED_ACQUISITION,
    spec_to_acquisition_function,
)

_MAPPABLE_METHODS = [
    m
    for m in AcquisitionMethod
    if m != AcquisitionMethod.AUTO and m not in BAYBE_UNSUPPORTED_ACQUISITION
]


def _spec(
    method: AcquisitionMethod,
    n_objectives: int = 1,
    **kwargs: Any,
) -> OptimizationSpec:
    objectives = [ObjectiveSpec(name=f"y{i}", minimize=True) for i in range(n_objectives)]
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=objectives,
        acquisition_method=method,
        **kwargs,
    )


class TestDispatchTableCompleteness:
    @pytest.mark.parametrize("method", _MAPPABLE_METHODS)
    def test_single_objective_table_is_complete(self, method: AcquisitionMethod) -> None:
        assert spec_to_acquisition_function(_spec(method, n_objectives=1)) is not None

    @pytest.mark.parametrize("method", _MAPPABLE_METHODS)
    def test_multi_objective_table_is_complete(self, method: AcquisitionMethod) -> None:
        assert spec_to_acquisition_function(_spec(method, n_objectives=2)) is not None

    @pytest.mark.parametrize(
        "abbreviation",
        sorted(set(_SINGLE_OBJECTIVE_ACQF_MAP.values()) | set(_MULTI_OBJECTIVE_ACQF_MAP.values())),
    )
    def test_mapped_abbreviations_exist_in_installed_baybe(self, abbreviation: str) -> None:
        """Pin-guard: a BayBE rename of any mapped abbreviation fails CI."""
        assert hasattr(baybe_acquisition, abbreviation), (
            f"BayBE no longer exposes acquisition abbreviation {abbreviation!r}"
        )

    def test_auto_and_unsupported_return_none(self) -> None:
        assert spec_to_acquisition_function(_spec(AcquisitionMethod.AUTO)) is None
        for method in BAYBE_UNSUPPORTED_ACQUISITION:
            assert spec_to_acquisition_function(_spec(method)) is None


class TestUCBBeta:
    def test_explicit_beta_builds_qucb_instance(self) -> None:
        resolved = spec_to_acquisition_function(
            _spec(AcquisitionMethod.UPPER_CONFIDENCE_BOUND, acquisition_beta=0.7)
        )
        assert isinstance(resolved, qUCB)
        assert resolved.beta == pytest.approx(0.7)

    def test_unset_beta_resolves_to_abbreviation(self) -> None:
        resolved = spec_to_acquisition_function(_spec(AcquisitionMethod.UPPER_CONFIDENCE_BOUND))
        assert resolved == "qUCB"

    def test_beta_with_non_ucb_method_raises(self) -> None:
        with pytest.raises(ValueError, match="UCB acquisition family"):
            spec_to_acquisition_function(_spec(AcquisitionMethod.NOISY_EI, acquisition_beta=0.7))


class TestBayBECapabilityReports:
    def test_new_members_are_supported_on_baybe(self) -> None:
        for method in (
            AcquisitionMethod.UPPER_CONFIDENCE_BOUND,
            AcquisitionMethod.ACTIVE_LEARNING,
            AcquisitionMethod.KNOWLEDGE_GRADIENT,
            AcquisitionMethod.POSTERIOR_STANDARD_DEVIATION,
        ):
            result = BayBEBackend().validate_capabilities(_spec(method))
            assert result.is_compatible, f"{method} unexpectedly incompatible on BayBE"

    def test_beta_with_non_ucb_method_is_unsupported(self) -> None:
        result = BayBEBackend().validate_capabilities(
            _spec(AcquisitionMethod.NOISY_EI, acquisition_beta=0.7)
        )
        assert not result.is_compatible
        assert any(r.key == "acquisition_beta" for r in result.unsupported)

    def test_thompson_sampling_rejects_batched_specs(self) -> None:
        result = BayBEBackend().validate_capabilities(
            _spec(AcquisitionMethod.THOMPSON_SAMPLING, batch_size=2)
        )
        assert not result.is_compatible
        reports = [r for r in result.unsupported if r.key == "acquisition_method"]
        assert reports
        assert "batch" in reports[0].reason

    def test_thompson_sampling_single_batch_is_supported(self) -> None:
        result = BayBEBackend().validate_capabilities(
            _spec(AcquisitionMethod.THOMPSON_SAMPLING, batch_size=1)
        )
        assert result.is_compatible


@pytest.mark.slow
class TestBehavioralSmoke:
    """One build-and-recommend smoke per newly-reachable family.

    Invariant assertions only (a suggestion inside bounds), per the
    stochastic-test guidance in TESTING.md. Mirrors the acquisition
    userguide's ``BotorchRecommender(acquisition_function=...)`` examples
    (https://emdgroup.github.io/baybe/stable/userguide/acquisition.html).
    """

    @pytest.mark.parametrize(
        "method",
        [
            AcquisitionMethod.UPPER_CONFIDENCE_BOUND,
            AcquisitionMethod.POSTERIOR_STANDARD_DEVIATION,
            AcquisitionMethod.ACTIVE_LEARNING,
            AcquisitionMethod.THOMPSON_SAMPLING,
        ],
    )
    def test_generate_suggestion_with_new_method(self, method: AcquisitionMethod) -> None:
        from bo_engine.types import ObservationData

        spec = _spec(method, random_seed=11)
        observations = [
            ObservationData(parameter_values={"x": v}, objective_values={"y0": v**2})
            for v in (0.1, 0.4, 0.8)
        ]
        batch = BayBEBackend().generate_suggestions(
            spec=spec,
            observations=observations,
            batch_size=1,
            iteration=1,
        )
        assert len(batch.suggestions) == 1
        value = batch.suggestions[0]["parameter_values"]["x"]
        assert 0.0 <= float(value) <= 1.0


def _desirability_spec(method: AcquisitionMethod, **kwargs: Any) -> OptimizationSpec:
    """Two-objective desirability spec (single scalarized acquisition output)."""
    from bo_engine.types import ScalarizationMode

    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[
            ObjectiveSpec(name="y0", minimize=True, normalization_bounds=(0.0, 1.0)),
            ObjectiveSpec(name="y1", minimize=False, normalization_bounds=(0.0, 1.0)),
        ],
        acquisition_method=method,
        scalarization=ScalarizationMode.DESIRABILITY,
        **kwargs,
    )


class TestDesirabilityAcquisitionDispatch:
    """Desirability scalarizes to one output → single-objective acqf family.

    BayBE rejects hypervolume/ParEGO acquisition functions for a
    scalarized :class:`baybe.objectives.DesirabilityObjective`, so every
    explicit method must resolve through the single-objective table —
    and the capability reports must agree with the dispatch (the
    intake/construction symmetry contract).
    """

    @pytest.mark.parametrize("method", _MAPPABLE_METHODS)
    def test_explicit_methods_resolve_via_single_objective_family(
        self, method: AcquisitionMethod
    ) -> None:
        resolved = spec_to_acquisition_function(_desirability_spec(method))
        expected = _SINGLE_OBJECTIVE_ACQF_MAP[method]
        if isinstance(resolved, str):
            assert resolved == expected
        else:
            # Instance results (qNIPV on continuous spaces) carry the full
            # class; resolve the abbreviation through BayBE's own namespace.
            assert isinstance(resolved, getattr(baybe_acquisition, expected))

    @pytest.mark.parametrize("method", _MAPPABLE_METHODS)
    def test_capability_reports_agree_with_dispatch(self, method: AcquisitionMethod) -> None:
        """No family-fallback report fires for desirability specs."""
        spec = _desirability_spec(method)
        result = BayBEBackend().validate_capabilities(spec)
        family_reports = [
            r
            for r in result.unsupported
            if r.key == "acquisition_method" and "single-objective semantics" in r.reason
        ]
        assert not family_reports, f"{method} wrongly reported as family fallback"

    def test_ucb_beta_reaches_the_scalarized_objective(self) -> None:
        resolved = spec_to_acquisition_function(
            _desirability_spec(AcquisitionMethod.UPPER_CONFIDENCE_BOUND, acquisition_beta=0.9)
        )
        assert isinstance(resolved, qUCB)
        assert resolved.beta == pytest.approx(0.9)

    def test_thompson_sampling_batch_rule_applies_to_desirability(self) -> None:
        """qTS cannot batch; the rule must fire for the scalarized (single-output) path."""
        result = BayBEBackend().validate_capabilities(
            _desirability_spec(AcquisitionMethod.THOMPSON_SAMPLING, batch_size=2)
        )
        reports = [r for r in result.unsupported if r.key == "acquisition_method"]
        assert reports
        assert "batch" in reports[0].reason

    @pytest.mark.slow
    def test_desirability_with_explicit_ei_generates(self) -> None:
        """Desirability + explicit noisy-EI recommends instead of raising.

        The C-class regression this pins: the pre-fix dispatch handed a
        hypervolume acquisition to the scalarized objective and every
        explicit-method desirability campaign crashed at suggestion time.
        Mirrors the desirability example of the BayBE objectives userguide
        (https://emdgroup.github.io/baybe/stable/userguide/objectives.html).
        """
        from bo_engine.types import ObservationData

        spec = _desirability_spec(AcquisitionMethod.NOISY_EI, random_seed=13)
        observations = [
            ObservationData(
                parameter_values={"x": v},
                objective_values={"y0": v**2, "y1": 1.0 - v},
            )
            for v in (0.1, 0.4, 0.8)
        ]
        batch = BayBEBackend().generate_suggestions(
            spec=spec,
            observations=observations,
            batch_size=1,
            iteration=1,
        )
        assert len(batch.suggestions) == 1


class TestSingleObjectiveFamilyFallbackReports:
    """M-class symmetry: single-objective-only members on multi-objective specs.

    The dispatch falls back to the hypervolume default (sanctioned), but
    the combination must be *classified* — UNSUPPORTED by default,
    IGNORED once ``'acquisition_method'`` is acknowledged — so the
    request (and an attached ``acquisition_beta``) can never vanish
    silently.
    """

    def test_ucb_on_multi_objective_is_unsupported_by_default(self) -> None:
        result = BayBEBackend().validate_capabilities(
            _spec(AcquisitionMethod.UPPER_CONFIDENCE_BOUND, n_objectives=2)
        )
        assert not result.is_compatible
        reports = [r for r in result.unsupported if r.key == "acquisition_method"]
        assert reports
        assert "single-objective semantics" in reports[0].reason

    def test_acknowledged_fallback_downgrades_to_ignored(self) -> None:
        result = BayBEBackend().validate_capabilities(
            _spec(
                AcquisitionMethod.UPPER_CONFIDENCE_BOUND,
                n_objectives=2,
                acknowledge_degradations=("acquisition_method",),
            )
        )
        assert result.is_compatible
        assert any("hypervolume default" in w for w in result.warnings)

    def test_multi_objective_capable_members_stay_unreported(self) -> None:
        for method in (
            AcquisitionMethod.NOISY_EI,
            AcquisitionMethod.HYPERVOLUME_IMPROVEMENT,
            AcquisitionMethod.SCALARIZED_MULTI_OBJ,
        ):
            result = BayBEBackend().validate_capabilities(_spec(method, n_objectives=2))
            assert result.is_compatible, f"{method} wrongly reported on multi-objective"
