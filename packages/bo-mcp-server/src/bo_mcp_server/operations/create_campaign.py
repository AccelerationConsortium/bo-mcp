"""Create campaign operation - protocol-neutral business logic."""

import logging
from typing import Any, Literal
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.backend import get_backend, resolve_backend_name
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    Campaign,
    CampaignIntakeInput,
    CampaignSpec,
    CampaignStatus,
)
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.idempotency import session_scope
from bo_mcp_server.operations.helpers import parse_verbosity
from bo_mcp_server.operations.validate_intake import validate_intake_operation
from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_create_campaign_response,
)
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
)

logger = logging.getLogger(__name__)


def _build_spec_from_dict(data: dict[str, Any]) -> CampaignSpec:
    """Canonically reconstruct CampaignSpec from a validated dict.

    The previous implementation manually unpacked a hand-picked subset of
    fields, dropping any advanced spec attributes (turbo_config,
    saasbo_config, outcome_constraints, etc.) silently. Routing through
    :meth:`CampaignSpec.model_validate` reuses the single source of truth
    for the schema so every field present in the dict round-trips into
    the persisted spec.
    """
    return CampaignSpec.model_validate(data)


async def create_campaign_operation(
    intake_data: CampaignIntakeInput,
    owner_id: str,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    *,
    session: AsyncSession | None = None,
) -> dict[str, Any]:
    """Create a new campaign from validated intake data.

    This contains the full business logic for campaign creation,
    independent of any transport protocol (MCP, REST, etc.).

    Args:
        intake_data: The campaign intake specification as a validated
            CampaignIntakeInput instance.
        owner_id: UUID string identifying the campaign owner.
        verbosity: Response detail level.
        session: Optional SQLAlchemy session to reuse. When supplied,
            all DB writes happen on that session and the caller owns
            the commit boundary — used by ``apply_idempotency``'s
            session-aware path so the campaign write and the cache
            finalize commit atomically. Omit to keep the operation
            self-contained.

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

    # Ask the backend whether it can handle this spec — surface warnings AND
    # enforce typed-option/feature capability. ``resolve_backend_name("auto",
    # ...)`` already routes around incompatible backends; the explicit-backend
    # path also has to fail-fast on UNSUPPORTED reports so misshaped BayBE
    # ``parameter_options`` / ``backend_options`` cannot reach the suggestion
    # path.
    backend = get_backend(spec.backend)
    opt_spec = campaign_spec_to_optimization_spec(spec)
    capabilities = backend.validate_capabilities(opt_spec)
    if not capabilities.is_compatible:
        unsupported_reports = [{"key": r.key, "reason": r.reason} for r in capabilities.unsupported]
        logger.warning(
            "Campaign creation rejected: backend %s reports %d unsupported items",
            spec.backend,
            len(unsupported_reports),
        )
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(
                f"Backend '{spec.backend}' cannot handle this spec: "
                + "; ".join(r["reason"] for r in unsupported_reports if r["reason"])
            ),
            details={
                "backend": spec.backend,
                "unsupported": unsupported_reports,
            },
        )
        response["campaign_id"] = None
        response["spec_id"] = None
        response["errors"] = [r["reason"] for r in unsupported_reports if r["reason"]]
        response["warnings"] = warnings
        return response
    warnings.extend(capabilities.warnings)

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

    # Save to storage — re-use the caller's session if one was passed,
    # otherwise open and commit our own.
    async with session_scope(session) as db:
        spec_repo = CampaignSpecRepository(db)
        campaign_repo = CampaignRepository(db)

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
