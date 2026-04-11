"""Validate intake tool wrapper for MCP."""

from typing import Any, Literal

from bo_mcp_server.domain import CampaignIntakeInput
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.validate_intake import validate_intake_operation
from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_validate_intake_response,
)
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_validate_intake")
async def validate_intake(
    intake_data: CampaignIntakeInput | dict[str, Any],
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
    """Validate a campaign specification without creating a campaign (dry-run).

    Workflow: Call before bo_create_campaign to check for errors without
    side effects. Fix any issues, then call bo_create_campaign.

    Args:
        intake_data: Campaign intake payload validated via CampaignIntakeInput.
            Includes campaign metadata, parameters, objectives, constraints,
            and optional execution settings (batch_size, max_iterations,
            initial_design_size, random_seed).
        verbosity: Response verbosity level. Options:
            - "minimal": ~20 tokens - valid/errors only
            - "standard": ~100 tokens - includes warnings and spec summary
            - "detailed": ~300+ tokens - full spec with all parameter details

    Returns:
        Dictionary shaped by verbosity:
            - minimal: valid, errors
            - standard: valid, errors, warnings, spec_summary
            - detailed: valid, errors, warnings, spec (validated CampaignSpec)

        On validation failure, returns:
            - valid: False
            - errors: List of validation messages
            - warnings: []
            - spec: None
    """
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    full_response = validate_intake_operation(intake_data)
    return format_validate_intake_response(full_response, verbosity_level)
