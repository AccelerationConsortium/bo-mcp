"""Expanded acquisition-method family on the BoTorch engine.

Covers the semantic members added for the full BayBE acquisition surface:
UCB (with the tunable ``acquisition_beta``), probability of improvement,
simple regret, the explicit non-log EI/NEI/NEHVI variants, and the
UNSUPPORTED-with-fallback policy for methods the BoTorch optimize path
cannot express (Thompson sampling, knowledge gradient, active learning,
posterior statistics).

References: BoTorch acquisition docs
(https://botorch.readthedocs.io/en/latest/acquisition.html — qUCB / qPI /
qSR / qEI / qNEI / qNEHVI) and the BayBE acquisition userguide
(https://emdgroup.github.io/baybe/stable/userguide/acquisition.html) whose
abbreviations define the semantic family this enum mirrors.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

from bo_engine.acquisition import (
    BOTORCH_UNSUPPORTED_ACQUISITION,
    create_acquisition,
)
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.constants import DEFAULT_UCB_BETA
from bo_engine.types import (
    AcquisitionMethod,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _single_task_model() -> tuple[SingleTaskGP, torch.Tensor, torch.Tensor]:
    train_x = torch.rand(6, 2, dtype=torch.float64)
    train_y = train_x.sum(dim=-1, keepdim=True)
    return SingleTaskGP(train_x, train_y), train_x, train_y


def _model_list() -> tuple[ModelListGP, torch.Tensor, torch.Tensor]:
    train_x = torch.rand(6, 2, dtype=torch.float64)
    train_y = torch.cat([train_x.sum(-1, keepdim=True), -train_x.prod(-1, keepdim=True)], dim=-1)
    models = [SingleTaskGP(train_x, train_y[:, i : i + 1]) for i in range(2)]
    return ModelListGP(*models), train_x, train_y


def _spec(method: AcquisitionMethod, **kwargs: Any) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        acquisition_method=method,
        **kwargs,
    )


_EXPECTED_SINGLE_CLASSES: dict[AcquisitionMethod, type] = {
    AcquisitionMethod.UPPER_CONFIDENCE_BOUND: qUpperConfidenceBound,
    AcquisitionMethod.PROBABILITY_OF_IMPROVEMENT: qProbabilityOfImprovement,
    AcquisitionMethod.SIMPLE_REGRET: qSimpleRegret,
    AcquisitionMethod.EXPECTED_IMPROVEMENT_NONLOG: qExpectedImprovement,
    AcquisitionMethod.NOISY_EI_NONLOG: qNoisyExpectedImprovement,
}


class TestSimpleMCFamily:
    @pytest.mark.parametrize(
        ("method", "expected_cls"),
        sorted(_EXPECTED_SINGLE_CLASSES.items(), key=lambda kv: kv[0].value),
    )
    def test_builds_expected_class(self, method: AcquisitionMethod, expected_cls: type) -> None:
        model, train_x, train_y = _single_task_model()
        acqf = create_acquisition(
            model=model,
            ref_point=None,
            train_x=train_x,
            train_y=train_y,
            n_objectives=1,
            maximize=True,
            method=method,
            beta=0.5 if method == AcquisitionMethod.UPPER_CONFIDENCE_BOUND else None,
        )
        assert isinstance(acqf, expected_cls)

    def test_ucb_default_beta_applied(self) -> None:
        model, train_x, train_y = _single_task_model()
        acqf = create_acquisition(
            model=model,
            ref_point=None,
            train_x=train_x,
            train_y=train_y,
            n_objectives=1,
            maximize=True,
            method=AcquisitionMethod.UPPER_CONFIDENCE_BOUND,
        )
        assert isinstance(acqf, qUpperConfidenceBound)
        # qUCB stores beta_prime = sqrt(beta * pi / 2) (BoTorch internal
        # reparameterization); assert via round-trip.
        expected = torch.sqrt(torch.tensor(DEFAULT_UCB_BETA * torch.pi / 2))
        assert torch.isclose(torch.as_tensor(acqf.beta_prime), expected)

    def test_beta_rejected_for_non_ucb_method(self) -> None:
        model, train_x, train_y = _single_task_model()
        with pytest.raises(ValueError, match="UCB acquisition"):
            create_acquisition(
                model=model,
                ref_point=None,
                train_x=train_x,
                train_y=train_y,
                n_objectives=1,
                maximize=True,
                method=AcquisitionMethod.NOISY_EI,
                beta=0.5,
            )

    def test_unsupported_methods_fall_back_to_family_default(self) -> None:
        model, train_x, train_y = _single_task_model()
        for method in sorted(BOTORCH_UNSUPPORTED_ACQUISITION, key=lambda m: m.value):
            acqf = create_acquisition(
                model=model,
                ref_point=None,
                train_x=train_x,
                train_y=train_y,
                n_objectives=1,
                maximize=True,
                method=method,
            )
            assert isinstance(acqf, qLogNoisyExpectedImprovement)


class TestMultiObjectiveNonLog:
    def test_hypervolume_nonlog_builds_qnehvi(self) -> None:
        model, train_x, train_y = _model_list()
        acqf = create_acquisition(
            model=model,
            ref_point=train_y.min(dim=0).values - 0.1,
            train_x=train_x,
            train_y=train_y,
            n_objectives=2,
            maximize_mask=torch.ones(2, dtype=torch.bool),
            method=AcquisitionMethod.HYPERVOLUME_IMPROVEMENT_NONLOG,
        )
        assert isinstance(acqf, qNoisyExpectedHypervolumeImprovement)
        assert not isinstance(acqf, qLogNoisyExpectedHypervolumeImprovement)


class TestTableCompleteness:
    @pytest.mark.parametrize(
        "method",
        [m for m in AcquisitionMethod if m != AcquisitionMethod.AUTO],
    )
    def test_every_member_resolves_single_objective(self, method: AcquisitionMethod) -> None:
        """No enum member may fall through the single-objective dispatch."""
        model, train_x, train_y = _single_task_model()
        acqf = create_acquisition(
            model=model,
            ref_point=None,
            train_x=train_x,
            train_y=train_y,
            n_objectives=1,
            maximize=True,
            method=method,
        )
        assert acqf is not None

    @pytest.mark.parametrize(
        "method",
        [m for m in AcquisitionMethod if m != AcquisitionMethod.AUTO],
    )
    def test_every_member_resolves_multi_objective(self, method: AcquisitionMethod) -> None:
        """No enum member may fall through the multi-objective dispatch."""
        model, train_x, train_y = _model_list()
        acqf = create_acquisition(
            model=model,
            ref_point=train_y.min(dim=0).values - 0.1,
            train_x=train_x,
            train_y=train_y,
            n_objectives=2,
            maximize_mask=torch.ones(2, dtype=torch.bool),
            method=method,
        )
        assert acqf is not None


class TestBoTorchBackendCapabilityReports:
    def test_unsupported_method_vetoes_backend(self) -> None:
        result = BoTorchBackend().validate_capabilities(_spec(AcquisitionMethod.THOMPSON_SAMPLING))
        assert not result.is_compatible
        assert any(r.key == "acquisition_method" for r in result.unsupported)

    def test_acknowledged_method_downgrades_to_ignored(self) -> None:
        result = BoTorchBackend().validate_capabilities(
            _spec(
                AcquisitionMethod.THOMPSON_SAMPLING,
                acknowledge_degradations=("acquisition_method",),
            )
        )
        assert result.is_compatible
        assert any("ignored" in w for w in result.warnings)

    def test_supported_method_stays_clean(self) -> None:
        result = BoTorchBackend().validate_capabilities(
            _spec(AcquisitionMethod.UPPER_CONFIDENCE_BOUND, acquisition_beta=0.4)
        )
        assert result.is_compatible

    def test_beta_with_non_ucb_method_is_unsupported(self) -> None:
        result = BoTorchBackend().validate_capabilities(
            _spec(AcquisitionMethod.NOISY_EI, acquisition_beta=0.4)
        )
        assert not result.is_compatible
        assert any(r.key == "acquisition_beta" for r in result.unsupported)


def _multi_spec(method: AcquisitionMethod, **kwargs: Any) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[
            ObjectiveSpec(name="y0", minimize=True),
            ObjectiveSpec(name="y1", minimize=True),
        ],
        acquisition_method=method,
        **kwargs,
    )


class TestSingleObjectiveFamilyFallbackReports:
    """M-class: single-objective-only members on multi-objective specs.

    The multi-objective dispatch falls back to the hypervolume default
    (sanctioned family behavior), but the combination must be classified
    at intake — the request and any attached ``acquisition_beta`` were
    previously discarded with no report on either backend.
    """

    def test_ucb_on_multi_objective_is_unsupported_by_default(self) -> None:
        result = BoTorchBackend().validate_capabilities(
            _multi_spec(AcquisitionMethod.UPPER_CONFIDENCE_BOUND, acquisition_beta=0.4)
        )
        assert not result.is_compatible
        reports = [r for r in result.unsupported if r.key == "acquisition_method"]
        assert reports
        assert "single-objective semantics" in reports[0].reason

    def test_acknowledged_fallback_downgrades_to_ignored(self) -> None:
        result = BoTorchBackend().validate_capabilities(
            _multi_spec(
                AcquisitionMethod.UPPER_CONFIDENCE_BOUND,
                acknowledge_degradations=("acquisition_method",),
            )
        )
        assert result.is_compatible
        assert any("hypervolume default" in w for w in result.warnings)

    def test_multi_objective_members_stay_unreported(self) -> None:
        for method in (
            AcquisitionMethod.NOISY_EI,
            AcquisitionMethod.HYPERVOLUME_IMPROVEMENT,
            AcquisitionMethod.SCALARIZED_MULTI_OBJ,
            AcquisitionMethod.HYPERVOLUME_IMPROVEMENT_NONLOG,
        ):
            result = BoTorchBackend().validate_capabilities(_multi_spec(method))
            assert result.is_compatible, f"{method} wrongly reported on multi-objective"

    def test_both_backends_classify_identically(self) -> None:
        """Routing honesty: neither backend honors UCB on a Pareto spec,
        so auto must reject (or run acknowledged) rather than silently
        landing on a backend that discards the request."""
        from bo_engine_baybe.backend import BayBEBackend

        spec = _multi_spec(AcquisitionMethod.UPPER_CONFIDENCE_BOUND, acquisition_beta=0.4)
        assert not BoTorchBackend().validate_capabilities(spec).is_compatible
        assert not BayBEBackend().validate_capabilities(spec).is_compatible


class TestSimpleMCWithOutcomeConstraintsReport:
    """M-class: the constrained reroute to NOISY_EI is DEGRADED, not silent."""

    @staticmethod
    def _constrained(method: AcquisitionMethod) -> OptimizationSpec:
        from bo_engine.types import OutcomeConstraintSpec

        return _spec(
            method,
            outcome_constraints=[OutcomeConstraintSpec(objective_name="y", threshold=1.0)],
        )

    def test_simple_mc_with_outcome_constraints_is_degraded(self) -> None:
        from bo_engine.backend_base import CapabilityStatus

        result = BoTorchBackend().validate_capabilities(
            self._constrained(AcquisitionMethod.UPPER_CONFIDENCE_BOUND)
        )
        assert result.is_compatible  # DEGRADED keeps the spec runnable
        degraded = [
            r
            for r in result.option_reports
            if r.key == "acquisition_method" and r.status == CapabilityStatus.DEGRADED
        ]
        assert degraded
        assert "NOISY_EI" in degraded[0].reason

    def test_noisy_ei_with_outcome_constraints_stays_clean(self) -> None:
        result = BoTorchBackend().validate_capabilities(
            self._constrained(AcquisitionMethod.NOISY_EI)
        )
        assert all(r.key != "acquisition_method" for r in result.option_reports), (
            "NOISY_EI needs no reroute report"
        )
