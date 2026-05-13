"""Create campaign operation - protocol-neutral business logic."""

import logging
from typing import Any, Literal
from uuid import UUID, uuid4

from bo_mcp_server.backend import get_backend, resolve_backend_name
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    AcquisitionOptimizationConfig,
    Campaign,
    CampaignIntakeInput,
    CampaignSpec,
    CampaignStatus,
    Constraint,
    ConstraintType,
    InputParameter,
    Objective,
    ParameterType,
)
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.helpers import parse_verbosity
from bo_mcp_server.operations.validate_intake import validate_intake_operation
from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_create_campaign_response,
)
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    get_session,
)

logger = logging.getLogger(__name__)


def _build_spec_from_dict(data: dict[str, Any]) -> CampaignSpec:
    """Reconstruct CampaignSpec from dictionary."""
    parameters = [
        InputParameter(
            name=p["name"],
            type=ParameterType(p["type"]),
            bounds=p.get("bounds"),
            values=p.get("values"),
            categories=p.get("categories"),
            description=p.get("description", ""),
        )
        for p in data["parameters"]
    ]

    objectives = [
        Objective(
            name=o["name"],
            direction=o["direction"],
            unit=o.get("unit", ""),
            target=o.get("target"),
        )
        for o in data["objectives"]
    ]

    constraints = [
        Constraint(
            type=ConstraintType(c["type"]),
            parameters=c["parameters"],
            value=c["value"],
            coefficients=c.get("coefficients"),
        )
        for c in data.get("constraints", [])
    ]

    acquisition_optimization_raw = data.get("acquisition_optimization")
    if acquisition_optimization_raw is None:
        acquisition_optimization = None
    elif isinstance(acquisition_optimization_raw, AcquisitionOptimizationConfig):
        acquisition_optimization = acquisition_optimization_raw
    else:
        acquisition_optimization = AcquisitionOptimizationConfig.model_validate(
            acquisition_optimization_raw
        )

    return CampaignSpec(
        name=data["name"],
        description=data.get("description", ""),
        parameters=parameters,
        objectives=objectives,
        constraints=constraints,
        batch_size=data.get("batch_size", 1),
        max_iterations=data.get("max_iterations"),
        max_observations=data.get("max_observations"),
        convergence_tolerance=data.get("convergence_tolerance"),
        initial_design_size=data.get("initial_design_size"),
        random_seed=data.get("random_seed"),
        acquisition_optimization=acquisition_optimization,
        backend=data.get("backend", "botorch"),
    )


async def create_campaign_operation(
    intake_data: CampaignIntakeInput,
    owner_id: str,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
    """Create a new campaign from validated intake data.

    This contains the full business logic for campaign creation,
    independent of any transport protocol (MCP, REST, etc.).

    Args:
        intake_data: The campaign intake specification as a validated
            CampaignIntakeInput instance.
        owner_id: UUID string identifying the campaign owner.
        verbosity: Response detail level.

    Returns:
        Formatted response dictionary with campaign details.
    """
    logger.info(
        "Creating campaign for owner_id=%s, verbosity=%s",
        owner_id,
        verbosity,
    )
    logger.debug("Intake data: %s", intake_data)

    # Validate verbosity parameter
    verbosity_result = parse_verbosity(verbosity)
    if isinstance(verbosity_result, dict):
        return verbosity_result
    verbosity_level = verbosity_result

    validation = validate_intake_operation(intake_data)

    if not validation["valid"]:
        logger.warning(
            "Campaign creation failed validation: %s",
            validation["errors"],
        )
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="Intake validation failed",
            details={
                "validation_errors": validation["errors"],
            },
        )
        # Surface detailed validation errors in the backward-compat errors list
        # so callers (including agents) can inspect them without digging into
        # error.details.
        response["errors"] = validation["errors"]
        response["campaign_id"] = None
        response["spec_id"] = None
        response["warnings"] = validation.get("warnings", [])
        return response

    # Parse owner_id
    try:
        owner_uuid = UUID(owner_id)
    except ValueError:
        logger.warning("Invalid owner_id format: %s", owner_id)
        response = make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            message="Invalid owner_id format",
            details={"owner_id": owner_id},
        )
        response["campaign_id"] = None
        response["spec_id"] = None
        return response

    # Reconstruct CampaignSpec from validated data
    spec_data = validation["spec"]

    # Resolve "auto" backend to a concrete backend name
    raw_backend = spec_data.get("backend", "auto")
    spec_data["backend"] = resolve_backend_name(raw_backend, spec_data)

    spec = _build_spec_from_dict(spec_data)
    warnings: list[str] = validation.get("warnings", [])

    # Ask the backend whether it can handle this spec — surface warnings
    backend = get_backend(spec.backend)
    opt_spec = campaign_spec_to_optimization_spec(spec)
    backend_warnings = backend.validate_spec(opt_spec)
    warnings.extend(backend_warnings)

    # Generate IDs
    spec_id = uuid4()
    campaign_id = uuid4()

    # Create campaign entity
    campaign = Campaign(
        id=campaign_id,
        spec_id=spec_id,
        owner_id=owner_uuid,
        status=CampaignStatus.CREATED,
    )

    # Save to storage
    async with get_session() as session:
        spec_repo = CampaignSpecRepository(session)
        campaign_repo = CampaignRepository(session)

        await spec_repo.save(spec, spec_id)
        await campaign_repo.save(campaign)

    logger.info(
        "Campaign created successfully: campaign_id=%s, spec_id=%s, name=%s",
        campaign_id,
        spec_id,
        spec.name,
    )

    # Build full response
    full_response: dict[str, Any] = {
        "success": True,
        "campaign_id": str(campaign_id),
        "spec_id": str(spec_id),
        "campaign_name": spec.name,
        "warnings": warnings,
        "errors": [],
    }

    # Add detailed info for detailed verbosity
    if verbosity_level == VerbosityLevel.DETAILED:
        full_response["spec_summary"] = {
            "name": spec.name,
            "description": spec.description,
            "n_parameters": len(spec.parameters),
            "n_objectives": len(spec.objectives),
            "n_constraints": (len(spec.constraints) if spec.constraints else 0),
            "batch_size": spec.batch_size,
            "parameter_names": [p.name for p in spec.parameters],
            "objective_names": [o.name for o in spec.objectives],
            "initial_design_size": spec.initial_design_size,
        }

    return format_create_campaign_response(full_response, verbosity_level)
