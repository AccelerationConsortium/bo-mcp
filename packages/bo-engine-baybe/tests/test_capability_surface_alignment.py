"""Tests for TODO 8.15: ``supported_features`` ↔ runtime validation alignment.

The static ``supported_features`` set must contain only features
the backend supports *unconditionally* (no spec-shape preconditions).
Features that depend on spec shape live in ``conditional_features``
so discovery surfaces (``list_capabilities``, docs, agent prompts)
do not over-advertise. The runtime validation done by
:meth:`validate_capabilities` is the single source of truth for
whether a given spec actually unlocks a conditional feature.

References:
- Baybe TaskParameter behaviour:
  https://emdgroup.github.io/baybe/stable/userguide/parameters.html#TaskParameter
- The capability tristate (SUPPORTED / IGNORED / UNSUPPORTED)
  introduced in TODO 1.69 to replace the flat boolean set.
"""

from __future__ import annotations

import pytest
from bo_engine.backend import Feature
from bo_engine.backend_base import CapabilityStatus
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

from bo_engine_baybe.backend import BayBEBackend


def _spec_without_task_parameter() -> OptimizationSpec:
    """Build a single-objective spec that does not exercise TRANSFER_LEARNING.

    The two continuous parameters never trigger any conditional
    feature in BayBE, so any conditional feature appearing here
    indicates an alignment bug in :meth:`validate_capabilities`.
    """
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="y", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="z", minimize=False)],
    )


def _spec_with_baybe_task_parameter() -> OptimizationSpec:
    """Build a spec that unlocks BayBE's TRANSFER_LEARNING precondition.

    A categorical parameter tagged ``parameter_options['baybe'].role==
    'task'`` is the BayBE-native transfer-learning mechanism. The
    capability surface should report TRANSFER_LEARNING as SUPPORTED
    only when this precondition is met.
    """
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(
                name="experiment",
                type=ParameterType.CATEGORICAL,
                categories=["A", "B"],
                parameter_options={
                    "baybe": {"role": "task", "active_values": ["A"]},
                },
            ),
        ],
        objectives=[ObjectiveSpec(name="z", minimize=False)],
    )


class TestSupportedFeaturesIsUnconditional:
    """The static set must not advertise conditional features."""

    def test_transfer_learning_is_not_in_static_set(self) -> None:
        """BayBE only honours TRANSFER_LEARNING when a TaskParameter is present.

        Pre-8.15 the static set advertised TRANSFER_LEARNING
        unconditionally; LLM clients planning against
        ``list_capabilities`` would hit a late rejection on any
        spec without a TaskParameter.
        """
        backend = BayBEBackend()
        assert Feature.TRANSFER_LEARNING not in backend.supported_features

    def test_conditional_features_records_the_precondition(self) -> None:
        """The conditional surface must name TRANSFER_LEARNING with its precondition."""
        backend = BayBEBackend()
        conditional = backend.conditional_features
        assert Feature.TRANSFER_LEARNING in conditional
        reason = conditional[Feature.TRANSFER_LEARNING]
        # Must mention the activation knob so a planner can satisfy
        # the precondition without reading the source.
        assert "task" in reason.lower()


class TestRuntimeValidationReflectsConditionalSupport:
    """``validate_capabilities`` is the runtime source of truth."""

    def test_spec_without_task_parameter_does_not_unlock_transfer_learning(self) -> None:
        """A vanilla spec must not advertise TRANSFER_LEARNING at runtime either.

        Belt-and-braces: the static surface and the runtime surface
        agree about the absence of the feature.
        """
        backend = BayBEBackend()
        result = backend.validate_capabilities(_spec_without_task_parameter())
        assert Feature.TRANSFER_LEARNING not in result.supported_features

    def test_spec_with_task_parameter_unlocks_transfer_learning(self) -> None:
        """When the precondition is met, the runtime surface flips to SUPPORTED.

        Mirrors the positive half of the test strategy: for every
        conditional feature, a spec meeting the precondition reports
        it as SUPPORTED at runtime.
        """
        backend = BayBEBackend()
        result = backend.validate_capabilities(_spec_with_baybe_task_parameter())
        feature_keys = {r.key: r for r in result.feature_reports}
        tl_report = feature_keys.get(str(Feature.TRANSFER_LEARNING))
        assert tl_report is not None, (
            "TRANSFER_LEARNING should appear in feature_reports for a TaskParameter spec"
        )
        assert tl_report.status == CapabilityStatus.SUPPORTED


class TestListCapabilitiesExposesConditionalSurface:
    """``list_capabilities`` carries the conditional surface end-to-end."""

    def test_response_contains_conditional_features_mapping(self) -> None:
        """The operation must echo the backend's conditional surface."""
        # Local import to avoid any module-load-order surprises when
        # the BayBE plugin is auto-discovered.
        from bo_mcp_server.operations.list_capabilities import (
            list_capabilities_operation,
        )

        response = list_capabilities_operation()
        assert "conditional_features" in response
        assert isinstance(response["conditional_features"], dict)

    def test_list_capabilities_exposes_baybe_transfer_learning_precondition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When BayBE is the active backend, the operation must report the
        TRANSFER_LEARNING precondition through ``conditional_features``.

        F5 regression: the original test only asserted the response
        carried *some* ``conditional_features`` mapping, which passed
        even when the active backend was BoTorch (whose conditional
        surface is empty) or when BayBE's mapping accidentally lost
        the TRANSFER_LEARNING entry. This forces the active backend
        to BayBE via :data:`bo_mcp_server.backend._backends` so the
        assertion actually exercises the BayBE plugin.

        Mirrors the strategy used by ``test_backend_resolution.py``
        which monkeypatches the same registry.
        """
        from bo_mcp_server import backend as backend_module
        from bo_mcp_server.operations.list_capabilities import (
            list_capabilities_operation,
        )

        baybe = BayBEBackend()
        monkeypatch.setitem(backend_module._backends, "baybe", baybe)
        monkeypatch.setenv("BO_BACKEND", "baybe")

        response = list_capabilities_operation()
        assert response["backend"] == "baybe"
        # TRANSFER_LEARNING must be absent from the unconditional
        # surface (pre-fix it was incorrectly listed there).
        assert "transfer_learning" not in response["supported_features"]
        # …and present in the conditional surface with its precondition
        # string so callers can satisfy it before submitting a spec.
        conditional = response["conditional_features"]
        assert "transfer_learning" in conditional, conditional
        assert "task" in conditional["transfer_learning"].lower()
