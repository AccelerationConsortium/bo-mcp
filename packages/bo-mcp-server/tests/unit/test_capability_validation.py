"""Contracts of the shared resolve + capability-validation pipeline.

``bo_create_campaign`` and ``bo_validate_intake`` used to carry separate
copies of the resolve → stamp → capability-check pipeline, and their
rejection rendering had already drifted (bare reasons vs ``key: reason``
strings). Both operations now call
:mod:`bo_mcp_server.operations.capability_validation`; the parity test
here fails the moment either surface grows its own formatting again.
"""

from __future__ import annotations

import pytest

from bo_engine.backend_base import (
    BackendValidationResult,
    CapabilityReport,
    CapabilityStatus,
)
from bo_mcp_server.operations.capability_validation import capability_rejection_errors
from bo_mcp_server.operations.create_campaign import create_campaign_operation
from bo_mcp_server.operations.validate_intake import validate_intake_with_capabilities

OWNER_ID = "00000000-0000-0000-0000-000000000000"


def _baybe_incompatible_intake() -> dict:
    """BayBE-pinned spec with a BoTorch-only knob → capability rejection.

    ``use_input_warping`` is a degradable knob BayBE cannot honor; without
    an ``acknowledge_degradations`` entry the capability report is
    UNSUPPORTED and both surfaces must reject.
    """
    return {
        "name": "Parity fixture",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "backend": "baybe",
        "use_input_warping": True,
    }


def _botorch_incompatible_mixed_intake() -> dict:
    return {
        "name": "Direct arylation shaped",
        "parameters": [
            {
                "name": "base",
                "type": "categorical",
                "categories": [f"base_{index}" for index in range(4)],
            },
            {
                "name": "ligand",
                "type": "categorical",
                "categories": [f"ligand_{index}" for index in range(12)],
            },
            {
                "name": "solvent",
                "type": "categorical",
                "categories": [f"solvent_{index}" for index in range(4)],
            },
            {"name": "concentration", "type": "discrete", "values": [0.1, 0.2, 0.3]},
            {"name": "temperature", "type": "discrete", "values": [80.0, 100.0, 120.0]},
        ],
        "objectives": [{"name": "yield", "direction": "maximize"}],
        "backend": "botorch",
    }


@pytest.mark.asyncio
async def test_create_and_validate_render_identical_rejection(
    setup_database: None,
) -> None:
    """The same capability-rejected spec renders identically on both surfaces."""
    _ = setup_database
    create_response = await create_campaign_operation(
        intake_data=_baybe_incompatible_intake(), owner_id=OWNER_ID
    )
    validate_response = await validate_intake_with_capabilities(_baybe_incompatible_intake())

    assert create_response["success"] is False
    assert validate_response["valid"] is False
    assert create_response["errors"]
    assert create_response["errors"] == validate_response["errors"]
    assert create_response["field_errors"] == validate_response["field_errors"]


@pytest.mark.asyncio
async def test_validate_rejects_large_mixed_space_pinned_to_botorch() -> None:
    """The intake check reports the late-acquisition failure before creation."""
    response = await validate_intake_with_capabilities(_botorch_incompatible_mixed_intake())

    assert response["valid"] is False
    assert response["backend"] == "botorch"
    assert "parameters" in response["field_errors"]
    assert any("192" in error for error in response["errors"])


class TestCapabilityRejectionErrors:
    """Formatting contract of the shared rejection formatter."""

    @staticmethod
    def _result(*reports: CapabilityReport) -> BackendValidationResult:
        return BackendValidationResult(backend="baybe", option_reports=tuple(reports))

    def test_keyed_report_lands_in_both_outputs(self) -> None:
        result = self._result(
            CapabilityReport(
                key="use_input_warping",
                status=CapabilityStatus.UNSUPPORTED,
                reason="not supported",
            )
        )
        errors, field_errors = capability_rejection_errors(result)
        assert errors == ["use_input_warping: not supported"]
        assert field_errors == {"use_input_warping": ["not supported"]}

    def test_keyless_report_never_creates_empty_string_key(self) -> None:
        """Keyless reasons go to ``errors`` only — agents key ``field_errors``
        on dotted paths, so an anonymous ``""`` entry cannot be targeted."""
        result = self._result(
            CapabilityReport(
                key="",
                status=CapabilityStatus.UNSUPPORTED,
                reason="spec-level rejection",
            )
        )
        errors, field_errors = capability_rejection_errors(result)
        assert errors == ["spec-level rejection"]
        assert "" not in field_errors
        assert field_errors == {}

    def test_reasonless_report_is_skipped(self) -> None:
        result = self._result(
            CapabilityReport(key="turbo_config", status=CapabilityStatus.UNSUPPORTED)
        )
        errors, field_errors = capability_rejection_errors(result)
        assert errors == []
        assert field_errors == {}
