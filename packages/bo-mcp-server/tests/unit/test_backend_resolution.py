"""Backend auto-resolution consults :class:`BackendValidationResult`.

Covers TODO 1.69's "spec-aware ``resolve_backend_name``" requirement:
``backend="auto"`` must ask candidate backends whether the *concrete*
spec is compatible, not just whether the flat ``supported_features``
set covers the coarse feature requirements.
"""

from __future__ import annotations

from typing import Any

import pytest
from bo_engine.backend import Feature, SuggestionBatch
from bo_engine.backend_base import (
    BackendValidationResult,
    BaseBackend,
    CapabilityReport,
    CapabilityStatus,
)
from bo_engine.types import ObservationData, OptimizationSpec

from bo_mcp_server import backend as backend_module
from bo_mcp_server.backend import resolve_backend_name


class _RejectingBackend(BaseBackend):
    """Pretend backend that always reports the spec as unsupported.

    Even though its ``supported_features`` set is broad, the spec-aware
    capability report fails — exactly the scenario that pre-1.69
    selection missed (claim feature support broadly, then fail late
    during suggestion generation).
    """

    @property
    def name(self) -> str:
        return "rejecting"

    @property
    def supported_features(self) -> frozenset[Feature]:
        return frozenset(Feature)

    def validate_capabilities(self, spec: OptimizationSpec) -> BackendValidationResult:
        _ = spec
        return BackendValidationResult(
            backend=self.name,
            feature_reports=(
                CapabilityReport(
                    key="multi_objective",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason="forced rejection for test",
                ),
            ),
        )

    def generate_suggestions(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        batch_size: int,
        iteration: int,
        backend_state: dict[str, Any] | None = None,
        pending_points: list[dict[str, Any]] | None = None,
        progress_callback=None,
    ) -> SuggestionBatch:  # pragma: no cover — never invoked for rejected spec
        _ = (
            spec,
            observations,
            batch_size,
            iteration,
            backend_state,
            pending_points,
            progress_callback,
        )
        raise AssertionError("rejected backend should not be invoked")


@pytest.fixture
def patched_backend_cache(monkeypatch):
    """Register the rejecting backend as the env default for one test."""
    rejecting = _RejectingBackend()
    monkeypatch.setitem(backend_module._backends, "rejecting", rejecting)
    monkeypatch.setenv("BO_BACKEND", "rejecting")

    # Force only the env default to be discoverable so the fallback path
    # is actually exercised.
    monkeypatch.setattr(backend_module, "_get_available_backend_names", lambda: ["rejecting"])
    yield rejecting


def _simple_spec_dict() -> dict:
    return {
        "name": "Auto",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }


def test_auto_resolution_consults_validate_capabilities(patched_backend_cache):
    """When the env default reports unsupported, auto falls back to BoTorch.

    Pre-1.69 the selector only checked ``required <= supported_features``.
    With the rejecting backend advertising every feature, the old code
    would have happily selected it. The new code respects
    ``validate_capabilities`` and falls back.
    """
    resolved = resolve_backend_name("auto", _simple_spec_dict())
    assert resolved == "botorch"


def test_explicit_backend_name_bypasses_capability_check():
    """An explicit backend name is honored verbatim — capability checks are
    informational warnings, not gates."""
    resolved = resolve_backend_name("baybe", _simple_spec_dict())
    assert resolved == "baybe"


def test_auto_selects_env_default_when_compatible(monkeypatch):
    """``auto`` accepts the env default when it reports compatibility."""
    monkeypatch.setenv("BO_BACKEND", "botorch")
    resolved = resolve_backend_name("auto", _simple_spec_dict())
    assert resolved == "botorch"


def test_auto_routes_baybe_hybrid_constraint_to_botorch(monkeypatch):
    """TODO 1.64 — BayBE rejects hybrid constraints, so ``auto`` falls back.

    A constraint over a mixed continuous/discrete parameter set is not
    expressible as a BayBE ``ContinuousLinearConstraint`` (numerical_only)
    nor as a ``DiscreteSumConstraint`` (discrete-only), so BayBE returns
    ``CapabilityStatus.UNSUPPORTED`` for the constraint and the selector
    must route to BoTorch.
    """
    monkeypatch.setenv("BO_BACKEND", "baybe")
    spec_dict = {
        "name": "Hybrid",
        "parameters": [
            {"name": "a", "type": "continuous", "bounds": [0.0, 1.0]},
            {"name": "b", "type": "discrete", "values": [0.0, 0.5, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "constraints": [
            {"type": "sum_equals", "parameters": ["a", "b"], "value": 1.0},
        ],
    }
    resolved = resolve_backend_name("auto", spec_dict)
    assert resolved == "botorch"


def test_auto_prefers_full_support_over_degraded(monkeypatch):
    """A backend that fully honors the spec wins over one that ignores knobs.

    With ``BO_BACKEND=baybe`` and ``use_input_warping=True``, BayBE is
    *compatible* (the knob is mapped to ``IGNORED``, not ``UNSUPPORTED``),
    but BoTorch fully honors the warping. The selector must prefer the
    fully-supported backend so auto never silently drops options another
    installed backend could honor.
    """
    monkeypatch.setenv("BO_BACKEND", "baybe")
    spec_dict = {
        "name": "Warped",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "use_input_warping": True,
    }
    resolved = resolve_backend_name("auto", spec_dict)
    assert resolved == "botorch"


def test_auto_picks_env_default_when_both_fully_support(monkeypatch):
    """Within the FULL tier, ``BO_BACKEND`` env default still wins ties.

    A simple continuous spec has no advanced knobs, so both BoTorch and
    BayBE are FULL. The env default (BayBE) must win — this is the
    deploy-time preference and the original behavior of the selector.
    """
    monkeypatch.setenv("BO_BACKEND", "baybe")
    spec_dict = {
        "name": "Simple",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    resolved = resolve_backend_name("auto", spec_dict)
    assert resolved == "baybe"


def test_auto_falls_back_to_degraded_when_no_full_support(monkeypatch):
    """If only degraded backends exist, the selector still picks one.

    Simulated by patching the available-backends list to BayBE only.
    The caller acknowledges the degradation so BayBE reports ``IGNORED``
    (DEGRADED tier) instead of ``UNSUPPORTED`` (INCOMPATIBLE). With no
    fully-supporting candidate available, the selector picks the only
    DEGRADED backend rather than failing.
    """
    from bo_mcp_server import backend as backend_module

    monkeypatch.setenv("BO_BACKEND", "baybe")
    monkeypatch.setattr(backend_module, "_get_available_backend_names", lambda: ["baybe"])
    spec_dict = {
        "name": "Warped",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "use_input_warping": True,
        "acknowledge_degradations": ["use_input_warping"],
    }
    resolved = resolve_backend_name("auto", spec_dict)
    assert resolved == "baybe"
