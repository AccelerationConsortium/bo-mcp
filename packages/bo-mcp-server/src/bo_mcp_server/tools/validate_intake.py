"""Validate intake tool for MCP."""

import logging
from typing import Any

from pydantic import ValidationError

from bo_mcp_server.domain import (
    CampaignSpec,
    Constraint,
    ConstraintType,
    InputParameter,
    Objective,
    ParameterType,
)
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel, format_validate_intake_response
from bo_mcp_server.server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
async def validate_intake(
    intake_data: dict[str, Any],
    verbosity: str = "standard",
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
    logger.debug(
        "Validating intake data: %s, verbosity=%s",
        intake_data.get("name", "<no name>"),
        verbosity,
    )

    # Validate verbosity parameter
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    errors: list[str] = []
    warnings: list[str] = []

    # Validate required fields
    if "name" not in intake_data or not intake_data["name"]:
        errors.append("Campaign name is required")

    if "parameters" not in intake_data or not intake_data["parameters"]:
        errors.append("At least one parameter is required")

    if "objectives" not in intake_data or not intake_data["objectives"]:
        errors.append("At least one objective is required")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "spec": None}

    # Parse parameters
    try:
        parameters = _parse_parameters(intake_data["parameters"])
    except ValueError as e:
        errors.append(f"Parameter error: {e}")
        parameters = []

    # Parse objectives
    try:
        objectives = _parse_objectives(intake_data["objectives"])
    except ValueError as e:
        errors.append(f"Objective error: {e}")
        objectives = []

    # Parse constraints
    constraints: list[Constraint] = []
    if intake_data.get("constraints"):
        try:
            constraints = _parse_constraints(
                intake_data["constraints"],
                [p.name for p in parameters],
            )
        except ValueError as e:
            errors.append(f"Constraint error: {e}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "spec": None}

    # Build CampaignSpec
    try:
        spec = CampaignSpec(
            name=intake_data["name"],
            description=intake_data.get("description", ""),
            parameters=parameters,
            objectives=objectives,
            constraints=constraints,
            batch_size=intake_data.get("batch_size", 1),
            max_iterations=intake_data.get("max_iterations"),
            initial_design_size=intake_data.get("initial_design_size"),
            random_seed=intake_data.get("random_seed"),
        )
    except ValidationError as e:
        for error in e.errors():
            errors.append(f"{error['loc']}: {error['msg']}")
        return {"valid": False, "errors": errors, "warnings": warnings, "spec": None}

    # Add warnings
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


def _parse_parameters(params_data: list[dict[str, Any]]) -> list[InputParameter]:
    """Parse parameter definitions from intake data."""
    parameters = []

    for i, p in enumerate(params_data):
        if "name" not in p:
            msg = f"Parameter {i} missing name"
            raise ValueError(msg)

        if "type" not in p:
            msg = f"Parameter '{p['name']}' missing type"
            raise ValueError(msg)

        try:
            param_type = ParameterType(p["type"])
        except ValueError:
            msg = f"Parameter '{p['name']}' has invalid type '{p['type']}'"
            raise ValueError(msg) from None

        # Build parameter
        param = InputParameter(
            name=p["name"],
            type=param_type,
            bounds=tuple(p["bounds"]) if p.get("bounds") else None,
            values=p.get("values"),
            categories=p.get("categories"),
            description=p.get("description", ""),
        )
        parameters.append(param)

    return parameters


def _parse_objectives(objs_data: list[dict[str, Any]]) -> list[Objective]:
    """Parse objective definitions from intake data."""
    objectives = []

    for i, o in enumerate(objs_data):
        if "name" not in o:
            msg = f"Objective {i} missing name"
            raise ValueError(msg)

        if "direction" not in o:
            msg = f"Objective '{o['name']}' missing direction"
            raise ValueError(msg)

        if o["direction"] not in ("minimize", "maximize"):
            msg = f"Objective '{o['name']}' has invalid direction '{o['direction']}'"
            raise ValueError(msg)

        obj = Objective(
            name=o["name"],
            direction=o["direction"],
            unit=o.get("unit", ""),
            target=o.get("target"),
        )
        objectives.append(obj)

    return objectives


def _parse_constraints(
    constraints_data: list[dict[str, Any]],
    param_names: list[str],
) -> list[Constraint]:
    """Parse constraint definitions from intake data."""
    constraints = []

    for i, c in enumerate(constraints_data):
        if "type" not in c:
            msg = f"Constraint {i} missing type"
            raise ValueError(msg)

        try:
            constraint_type = ConstraintType(c["type"])
        except ValueError:
            msg = f"Constraint {i} has invalid type '{c['type']}'"
            raise ValueError(msg) from None

        if "parameters" not in c or not c["parameters"]:
            msg = f"Constraint {i} missing parameters"
            raise ValueError(msg)

        # Validate parameter references
        for param_name in c["parameters"]:
            if param_name not in param_names:
                msg = f"Constraint {i} references unknown parameter '{param_name}'"
                raise ValueError(msg)

        if "value" not in c:
            msg = f"Constraint {i} missing value"
            raise ValueError(msg)

        constraint = Constraint(
            type=constraint_type,
            parameters=c["parameters"],
            value=c["value"],
            coefficients=c.get("coefficients"),
        )
        constraints.append(constraint)

    return constraints
