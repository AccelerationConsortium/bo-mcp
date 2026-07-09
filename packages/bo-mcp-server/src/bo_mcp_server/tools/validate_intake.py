"""Validate intake tool wrapper for MCP."""

from typing import Any, Literal, cast

from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.validate_intake import validate_intake_with_capabilities
from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_validate_intake_response,
)
from bo_mcp_server.server import mcp
from bo_mcp_server.tool_boundary import VALIDATE_INTAKE_ENVELOPE_EXTRA
from bo_mcp_server.tools.annotations import READ_ONLY
from bo_mcp_server.tools.common import IntakePayload, check_intake_shape
from bo_mcp_server.tools.response_models import ValidateIntakeResponse

# The tool boundary widens to ``Any`` for the same reason as
# ``bo_create_campaign``: the
# ``CampaignIntakeInput | dict[str, Any]`` union pre-1.53 made
# FastMCP attempt both union arms on every malformed payload, producing
# Pydantic-internal ``loc`` paths like
# ``function-after[validate_names_and_constraints(), CampaignIntakeInput]``
# that no agent can act on. Defer the shape check and full Pydantic
# validation to :func:`validate_intake_operation` so the structured
# ``field_errors`` envelope wins on every failure mode -- whether the
# outer shape is wrong, a scalar arg is missing, or an inner sub-
# field tripped a constraint.
_VALIDATE_INTAKE_BOUNDARY_DEFAULTS = VALIDATE_INTAKE_ENVELOPE_EXTRA


def _check_intake_shape(intake_data: object) -> dict[str, Any] | None:
    """Return a structured envelope iff ``intake_data`` is not object-shaped.

    Mirrors :func:`bo_mcp_server.tools.create_campaign._check_intake_shape`
    so the failure path for ``bo_validate_intake`` matches the
    create-campaign contract -- a single dotted ``intake_data`` key
    in ``field_errors`` instead of Pydantic-internal noise.
    """
    return check_intake_shape(intake_data, extra=_VALIDATE_INTAKE_BOUNDARY_DEFAULTS)


@mcp.tool(name="bo_validate_intake", annotations=READ_ONLY)
async def validate_intake(
    intake_data: IntakePayload,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> ValidateIntakeResponse:
    """Validate a campaign specification without creating a campaign (dry-run).

    Workflow: Call before bo_create_campaign to check for errors without
    side effects. Fix any issues, then call bo_create_campaign.

    Args:
        intake_data: Campaign intake payload validated via CampaignIntakeInput.
            Includes campaign metadata, parameters, objectives, constraints,
            and optional execution settings (batch_size, max_iterations,
            initial_design_size, random_seed). ``random_seed`` makes
            suggestions deterministic only within a fixed torch version,
            device, and ``torch.use_deterministic_algorithms`` setting; it is
            not a cross-version reproducibility guarantee.
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
    shape_error = _check_intake_shape(intake_data)
    if shape_error is not None:
        return cast(ValidateIntakeResponse, shape_error)

    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return cast(
            ValidateIntakeResponse,
            make_error_response(
                ErrorCode.VALIDATION_FAILED,
                message=(
                    f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed"
                ),
            ),
        )

    full_response = await validate_intake_with_capabilities(intake_data)
    return cast(
        ValidateIntakeResponse,
        format_validate_intake_response(full_response, verbosity_level),
    )
