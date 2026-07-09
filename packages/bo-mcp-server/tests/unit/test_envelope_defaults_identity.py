"""Per-tool error-envelope defaults must be declared once, not copy-pasted.

``tool_boundary.py``'s boundary-caught failures and each tool's own
body-caught failures (shape checks, Pydantic validation) must render
the identical envelope shape for the same tool. That only holds by
construction if both call sites hold the same object, not two
equal-looking dict literals that can drift independently.
"""

from __future__ import annotations

from bo_mcp_server.tool_boundary import (
    _TOOL_ENVELOPE_OVERRIDES,
    CREATE_CAMPAIGN_ENVELOPE_EXTRA,
    SUBMIT_RESULTS_ENVELOPE_EXTRA,
    VALIDATE_INTAKE_ENVELOPE_EXTRA,
)
from bo_mcp_server.tools.create_campaign import _INTAKE_BOUNDARY_DEFAULTS
from bo_mcp_server.tools.submit_results import _RESULTS_BOUNDARY_DEFAULTS
from bo_mcp_server.tools.validate_intake import _VALIDATE_INTAKE_BOUNDARY_DEFAULTS


def test_create_campaign_envelope_defaults_are_the_same_object() -> None:
    assert _INTAKE_BOUNDARY_DEFAULTS is CREATE_CAMPAIGN_ENVELOPE_EXTRA
    assert _INTAKE_BOUNDARY_DEFAULTS is _TOOL_ENVELOPE_OVERRIDES["bo_create_campaign"]


def test_submit_results_envelope_defaults_are_the_same_object() -> None:
    assert _RESULTS_BOUNDARY_DEFAULTS is SUBMIT_RESULTS_ENVELOPE_EXTRA
    assert _RESULTS_BOUNDARY_DEFAULTS is _TOOL_ENVELOPE_OVERRIDES["bo_submit_results"]


def test_validate_intake_envelope_defaults_are_the_same_object() -> None:
    assert _VALIDATE_INTAKE_BOUNDARY_DEFAULTS is VALIDATE_INTAKE_ENVELOPE_EXTRA
    assert _VALIDATE_INTAKE_BOUNDARY_DEFAULTS is _TOOL_ENVELOPE_OVERRIDES["bo_validate_intake"]
