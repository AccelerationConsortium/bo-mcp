"""Validate intake tool for MCP."""

import logging
from typing import Any, Literal

from pydantic import ValidationError

from bo_mcp_server.domain import (
    CampaignIntakeInput,
    CampaignSpec,
)
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel, format_validate_intake_response
from bo_mcp_server.server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
async def validate_intake(
    intake_data: CampaignIntakeInput,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
    """Validate campaign intake data and return validation result.

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
    intake_name = intake_data.name if isinstance(intake_data, CampaignIntakeInput) else "<no name>"
    logger.debug("Validating intake data: %s, verbosity=%s", intake_name, verbosity)

    # Validate verbosity parameter
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    try:
        intake = (
            intake_data
            if isinstance(intake_data, CampaignIntakeInput)
            else CampaignIntakeInput.model_validate(intake_data)
        )
        spec = CampaignSpec(**intake.model_dump())
    except ValidationError as e:
        errors: list[str] = []
        for error in e.errors():
            errors.append(f"{error['loc']}: {error['msg']}")
        return {"valid": False, "errors": errors, "warnings": [], "spec": None}

    # Add warnings
    warnings: list[str] = []
    if spec.n_objectives > 4:
        warnings.append(
            f"Many objectives ({spec.n_objectives}) may make Pareto front difficult to visualize"
        )

    if spec.n_parameters > 20:
        warnings.append(
            f"Many parameters ({spec.n_parameters}) may require more initial design points"
        )

    logger.info(
        "Intake validated successfully: name=%s, params=%d, objectives=%d",
        spec.name,
        spec.n_parameters,
        spec.n_objectives,
    )

    full_response = {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "spec": spec.to_dict(),
    }

    return format_validate_intake_response(full_response, verbosity_level)
