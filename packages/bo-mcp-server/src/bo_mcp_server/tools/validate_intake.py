"""Validate intake tool wrapper for MCP."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from bo_mcp_server.domain.intake_models import INTAKE_INPUT_JSON_SCHEMA
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.field_errors import shape_envelope
from bo_mcp_server.operations.validate_intake import validate_intake_operation
from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_validate_intake_response,
)
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY

# The tool boundary widens to ``Any`` for the same reason as
# ``bo_create_campaign`` (see TODO 1.53 follow-up): the
# ``CampaignIntakeInput | dict[str, Any]`` union pre-1.53 made
# FastMCP attempt both union arms on every malformed payload, producing
# Pydantic-internal ``loc`` paths like
# ``function-after[validate_names_and_constraints(), CampaignIntakeInput]``
# that no agent can act on. Defer the shape check and full Pydantic
# validation to :func:`validate_intake_operation` so the structured
# ``field_errors`` envelope wins on every failure mode -- whether the
# outer shape is wrong, a scalar arg is missing, or an inner sub-
# field tripped a constraint.
IntakePayload = Annotated[
    Any,
    Field(
        description=(
            "Campaign intake specification. Validated against "
            "CampaignIntakeInput; validation failures are returned as a "
            "structured error envelope with ``field_errors`` keyed by "
            "dotted path."
        ),
        json_schema_extra={
            "type": "object",
            "properties": INTAKE_INPUT_JSON_SCHEMA.get("properties", {}),
            "required": INTAKE_INPUT_JSON_SCHEMA.get("required", []),
            "$defs": INTAKE_INPUT_JSON_SCHEMA.get("$defs", {}),
            "additionalProperties": False,
        },
    ),
]

_VALIDATE_INTAKE_BOUNDARY_DEFAULTS: dict[str, Any] = {
    "valid": False,
    "warnings": [],
    "spec": None,
}


def _check_intake_shape(intake_data: object) -> dict[str, Any] | None:
    """Return a structured envelope iff ``intake_data`` is not object-shaped.

    Mirrors :func:`bo_mcp_server.tools.create_campaign._check_intake_shape`
    so the failure path for ``bo_validate_intake`` matches the
    create-campaign contract -- a single dotted ``intake_data`` key
    in ``field_errors`` instead of Pydantic-internal noise.
    """
    if isinstance(intake_data, (Mapping, BaseModel)):
        return None
    return shape_envelope(
        "intake_data",
        f"Input should be an object, got {type(intake_data).__name__}",
        extra=_VALIDATE_INTAKE_BOUNDARY_DEFAULTS,
    )


@mcp.tool(name="bo_validate_intake", annotations=READ_ONLY)
async def validate_intake(
    intake_data: IntakePayload,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
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
        return shape_error

    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    full_response = validate_intake_operation(intake_data)
    return format_validate_intake_response(full_response, verbosity_level)
