"""Contract tests for the transfer-learning advertisement surface.

``OptimizationSpec.transfer_learning`` exists, ``Feature.TRANSFER_LEARNING``
was in the BoTorch backend's static capability set, and
``validate_capabilities`` reported the option supported — but
``generate_next_batch`` never reads ``spec.transfer_learning``: prior
campaigns are never loaded into ``PriorTaskData`` and nothing dispatches
to ``generate_rgpe_suggestions``. A campaign requesting prior-campaign
transfer received ordinary GP suggestions with no warning.

The fix follows the multi-fidelity / SAASBO honesty precedent
(``test_multifidelity_advertisement_contract.py``,
``test_saasbo_advertisement_contract.py``): stop advertising and raise
loudly when a caller requests the un-wired path. BayBE's native
``TaskParameter`` mechanism (its ``validate_capabilities`` flips
``TRANSFER_LEARNING`` to SUPPORTED when a task parameter is declared)
remains the supported campaign-level transfer route.

References:
    - Feurer, Letham, Bakshy "Practical Transfer Learning for Bayesian
      Optimization" (https://arxiv.org/abs/1802.02219) — the RGPE method
      the standalone helpers implement.
    - BoTorch RGPE tutorial:
      https://botorch.org/docs/tutorials/meta_learning_with_rgpe/
"""

from __future__ import annotations

import pytest

from bo_engine.backend import Feature
from bo_engine.backend_base import CapabilityStatus
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.suggestions import (
    TransferLearningNotSupportedError,
    generate_next_batch,
)
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    TransferLearningSpec,
)


def _transfer_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        transfer_learning=TransferLearningSpec(prior_campaign_ids=["prior-1"]),
    )


class TestSupportedFeaturesDropsTransferLearning:
    """The static capability set must not advertise TRANSFER_LEARNING."""

    def test_botorch_backend_omits_transfer_learning(self) -> None:
        assert Feature.TRANSFER_LEARNING not in BoTorchBackend().supported_features


class TestSuggestionRouteRejectsTransferLearning:
    """``generate_next_batch`` must refuse the un-wired path with a typed error."""

    def test_transfer_learning_rejected(self) -> None:
        with pytest.raises(TransferLearningNotSupportedError):
            generate_next_batch(_transfer_spec(), [], batch_size=1)


class TestCapabilityReportVetoesTransferLearning:
    """The spec-aware capability report must not claim transfer support."""

    def test_transfer_learning_reported_unsupported(self) -> None:
        result = BoTorchBackend().validate_capabilities(_transfer_spec())
        assert not result.is_compatible
        # The feature and option surfaces share the key string
        # ("transfer_learning") — each must carry exactly one
        # UNSUPPORTED report with a reason.
        feature_reports = [
            r for r in result.feature_reports if r.key == str(Feature.TRANSFER_LEARNING)
        ]
        option_reports = [r for r in result.option_reports if r.key == "transfer_learning"]
        for reports in (feature_reports, option_reports):
            assert len(reports) == 1
            assert reports[0].status == CapabilityStatus.UNSUPPORTED
            assert reports[0].reason

    def test_option_report_agrees_with_feature_report(self) -> None:
        """The concrete spec option must be vetoed alongside the feature.

        Spec options and inferred features are reported separately; an
        option report claiming SUPPORTED next to an UNSUPPORTED feature
        report is a contradictory signal to clients inspecting either
        surface.
        """
        result = BoTorchBackend().validate_capabilities(_transfer_spec())
        unsupported_keys = [r.key for r in result.unsupported]
        assert "transfer_learning" in unsupported_keys
        supported_keys = {
            r.key for r in result.option_reports if r.status == CapabilityStatus.SUPPORTED
        }
        assert "transfer_learning" not in supported_keys

    def test_spec_without_transfer_stays_compatible(self) -> None:
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        assert BoTorchBackend().validate_capabilities(spec).is_compatible


class TestStandaloneRgpeHelpersStillImportable:
    """The bo_engine.transfer_learning helpers remain available."""

    def test_module_exposes_entry_points(self) -> None:
        import bo_engine.transfer_learning as tl

        assert hasattr(tl, "PriorTaskData")
        assert hasattr(tl, "create_rgpe_model")
        assert hasattr(tl, "generate_rgpe_suggestions")
