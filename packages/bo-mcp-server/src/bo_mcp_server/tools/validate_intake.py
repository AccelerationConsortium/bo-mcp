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
        intake_data: Dictionary containing campaign configuration:
            - name: Campaign name (required)
            - description: Campaign description (optional)
            - parameters: List of parameter definitions (required)
            - objectives: List of objective definitions (required)
            - constraints: List of constraint definitions (optional)
            - batch_size: Number of suggestions per batch (optional, default 1)
            - max_iterations: Maximum iterations (optional)
            - initial_design_size: Initial design points (optional)
            - random_seed: Random seed for reproducibility (optional)
        verbosity: Response verbosity level. Options:
            - "minimal": ~20 tokens - valid/errors only
            - "standard": ~100 tokens - includes warnings and spec summary
            - "detailed": ~300+ tokens - full spec with all parameter details

    Returns:
        Dictionary with:
            - valid: Boolean indicating if intake is valid
            - errors: List of error messages (if invalid)
            - warnings: List of warning messages
            - spec: Validated CampaignSpec as dict (if valid)
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
