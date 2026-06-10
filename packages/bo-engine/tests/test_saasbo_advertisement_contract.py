"""Contract tests for the SAASBO advertisement surface.

``OptimizationSpec.saasbo_config`` exists, ``should_use_saasbo`` /
``generate_saasbo_suggestions`` exist in ``bo_engine.saasbo``, and the
capability surface advertised the option as supported — but the
suggestion pipeline never dispatched to SAASBO. A spec carrying
``saasbo_config`` silently ran a plain dense-ARD ``SingleTaskGP``
instead of the sparsity-inducing SAAS priors the caller asked for.

The fix mirrors the multi-fidelity precedent
(``MultiFidelityNotSupportedError``, pinned by
``test_multifidelity_advertisement_contract.py``): stop advertising and
raise loudly when a caller requests the un-wired path.

This module tests:

* Requesting ``saasbo_config`` raises the typed
  :class:`SAASBONotSupportedError` at the suggestion boundary.
* ``BoTorchBackend.validate_capabilities`` reports the option
  ``UNSUPPORTED`` (hard veto for ``backend="auto"`` routing and pinned
  ``backend="botorch"`` intakes).
* The standalone ``bo_engine.saasbo`` helpers remain importable so
  callers with a direct SAASBO workflow are not blocked.

References:
    - Eriksson & Jankowiak, "High-Dimensional Bayesian Optimization
      with Sparse Axis-Aligned Subspaces", UAI 2021
      (https://arxiv.org/abs/2103.00349)
    - BoTorch SAASBO tutorial:
      https://botorch.org/docs/tutorials/saasbo/
"""

from __future__ import annotations

import pytest

from bo_engine.backend_base import CapabilityStatus
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.saasbo import SAASBOConfig
from bo_engine.suggestions import (
    SAASBONotSupportedError,
    generate_next_batch,
)
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _saasbo_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        saasbo_config=SAASBOConfig(warmup_steps=8, num_samples=8, thinning=2),
    )


class TestSuggestionRouteRejectsSaasbo:
    """``generate_next_batch`` must refuse the un-wired path with a typed error."""

    def test_saasbo_config_rejected(self) -> None:
        with pytest.raises(SAASBONotSupportedError):
            generate_next_batch(_saasbo_spec(), [], batch_size=1)

    def test_spec_without_saasbo_unaffected(self) -> None:
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        suggestions, _ = generate_next_batch(spec, [], batch_size=1)
        assert len(suggestions) == 1


class TestCapabilityReportVetoesSaasbo:
    """The spec-aware capability report must not claim SAASBO support."""

    def test_saasbo_option_reported_unsupported(self) -> None:
        result = BoTorchBackend().validate_capabilities(_saasbo_spec())
        assert not result.is_compatible
        saasbo_reports = [r for r in result.option_reports if r.key == "saasbo_config"]
        assert len(saasbo_reports) == 1
        assert saasbo_reports[0].status == CapabilityStatus.UNSUPPORTED
        assert saasbo_reports[0].reason

    def test_spec_without_saasbo_stays_compatible(self) -> None:
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        assert BoTorchBackend().validate_capabilities(spec).is_compatible


class TestStandaloneSaasboHelpersStillImportable:
    """The bo_engine.saasbo helpers remain available for direct workflows."""

    def test_module_exposes_entry_points(self) -> None:
        import bo_engine.saasbo as saasbo

        assert hasattr(saasbo, "SAASBOConfig")
        assert hasattr(saasbo, "create_and_fit_saasbo_model")
        assert hasattr(saasbo, "generate_saasbo_suggestions")
        assert hasattr(saasbo, "should_use_saasbo")
