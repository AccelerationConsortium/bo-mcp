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
        assert Feature.MULTI_FIDELITY not in backend.supported_features
        # All other Feature members should remain (defensive sanity check that
        # we didn't accidentally drop more than the one entry).
        for feature in Feature:
            if feature is Feature.MULTI_FIDELITY:
                continue
            assert feature in backend.supported_features


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
