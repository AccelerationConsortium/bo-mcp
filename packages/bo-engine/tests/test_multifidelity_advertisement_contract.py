"""Contract tests for the multi-fidelity advertisement surface.

The audit flagged that ``AcquisitionMethod.MULTI_FIDELITY_KG``,
``Feature.MULTI_FIDELITY`` and the capability surface were lying to LLM
clients: each advertised multi-fidelity, but the suggestion pipeline did
not actually dispatch to BoTorch's qMFKG and did not construct a
``SingleTaskMultiFidelityGP``. The fix is to stop advertising and to
raise loudly when a caller requests the un-wired path.

This module tests:

* The BoTorch backend does not advertise ``Feature.MULTI_FIDELITY``.
* Requesting ``MULTI_FIDELITY_KG`` raises the typed
  :class:`MultiFidelityNotSupportedError` at the suggestion boundary.
* Setting ``spec.fidelity_parameter`` raises the same error.
* ``validate_capabilities`` mirrors the static exclusion: a
  multi-fidelity spec must be ``is_compatible=False`` so intake and
  ``backend="auto"`` reject at create time instead of failing later at
  suggestion time.
* The standalone ``bo_engine.multifidelity`` helpers remain importable so
  callers with a direct multi-fidelity workflow are not blocked.

References:
    - BoTorch qMFKG tutorial:
      https://botorch.org/tutorials/multi_fidelity_bo
    - Wu et al., "Multi-Fidelity Knowledge Gradient", NeurIPS 2019 —
      foundational reference for the qMFKG acquisition.
"""

from __future__ import annotations

import pytest

from bo_engine.backend import Feature
from bo_engine.backend_base import CapabilityStatus
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.suggestions import (
    MultiFidelityNotSupportedError,
    generate_next_batch,
)
from bo_engine.types import (
    AcquisitionMethod,
    FidelityParameterSpec,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


class TestSupportedFeaturesDropsMultiFidelity:
    """The static capability set must not advertise MULTI_FIDELITY."""

    def test_botorch_backend_omits_multi_fidelity(self) -> None:
        backend = BoTorchBackend()
        # Protocol contract: the property is an immutable frozenset
        # (set subtraction with dict views silently yields a plain set).
        assert isinstance(backend.supported_features, frozenset)
        assert Feature.MULTI_FIDELITY not in backend.supported_features
        # All other Feature members should remain (defensive sanity check
        # that we didn't accidentally drop more than the un-routed
        # entries; TRANSFER_LEARNING is excluded for the same reason —
        # see test_transfer_learning_advertisement_contract.py).
        for feature in Feature:
            if feature in (Feature.MULTI_FIDELITY, Feature.TRANSFER_LEARNING):
                continue
            assert feature in backend.supported_features


class TestCapabilityReportVetoesMultiFidelity:
    """The spec-aware report must mirror the static exclusion.

    Intake (``create_campaign``) and ``backend="auto"`` routing consult
    only ``validate_capabilities().is_compatible`` — without the veto a
    multi-fidelity campaign is accepted at create time and then fails at
    suggestion time with ``MultiFidelityNotSupportedError``.
    """

    def test_fidelity_parameter_reported_unsupported(self) -> None:
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            fidelity_parameter=FidelityParameterSpec(
                name="fidelity", bounds=(0.1, 1.0), target=1.0
            ),
        )
        result = BoTorchBackend().validate_capabilities(spec)
        assert not result.is_compatible
        unsupported_keys = [r.key for r in result.unsupported]
        # Both report surfaces must agree: the inferred feature AND the
        # concrete spec option are vetoed — a SUPPORTED option report
        # next to an UNSUPPORTED feature report is a contradictory signal.
        assert str(Feature.MULTI_FIDELITY) in unsupported_keys
        assert "fidelity_parameter" in unsupported_keys
        supported_keys = {
            r.key for r in result.option_reports if r.status == CapabilityStatus.SUPPORTED
        }
        assert "fidelity_parameter" not in supported_keys
        assert all(r.reason for r in result.unsupported)

    def test_mfkg_acquisition_method_reported_unsupported(self) -> None:
        # required_features() infers MULTI_FIDELITY from fidelity_parameter
        # only, so the acquisition-method route needs its own veto.
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            acquisition_method=AcquisitionMethod.MULTI_FIDELITY_KG,
        )
        result = BoTorchBackend().validate_capabilities(spec)
        assert not result.is_compatible
        assert "acquisition_method" in [r.key for r in result.unsupported]


class TestSuggestionRouteRejectsMultiFidelity:
    """``generate_next_batch`` must refuse the un-wired path with a typed error."""

    def _make_spec(self) -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

    def test_acquisition_method_mfkg_rejected(self) -> None:
        spec = self._make_spec()
        spec = OptimizationSpec(
            parameters=spec.parameters,
            objectives=spec.objectives,
            acquisition_method=AcquisitionMethod.MULTI_FIDELITY_KG,
        )
        with pytest.raises(MultiFidelityNotSupportedError):
            generate_next_batch(spec, [], batch_size=1)

    def test_fidelity_parameter_rejected(self) -> None:
        spec = self._make_spec()
        spec = OptimizationSpec(
            parameters=spec.parameters,
            objectives=spec.objectives,
            fidelity_parameter=FidelityParameterSpec(
                name="fidelity", bounds=(0.1, 1.0), target=1.0
            ),
        )
        with pytest.raises(MultiFidelityNotSupportedError):
            generate_next_batch(spec, [], batch_size=1)


class TestStandaloneMultiFidelityHelpersStillImportable:
    """The legacy bo_engine.multifidelity helpers remain available."""

    def test_module_imports(self) -> None:
        import bo_engine.multifidelity as mf

        # The constructor and the qMFKG factory both must remain reachable
        # so external scripts that drive multi-fidelity workflows directly
        # are not broken by the advertisement removal.
        assert hasattr(mf, "MultiFidelityConfig")
        assert hasattr(mf, "create_mfkg_acquisition")
