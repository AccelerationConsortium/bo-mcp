"""Create campaign operation - protocol-neutral business logic."""

import asyncio
import logging
from typing import Any, Literal
from uuid import UUID, uuid4

from bo_engine.backend_base import BackendValidationResult
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.backend import get_backend_async, resolve_backend_name
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
    with_response_metadata,
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


def _intake_validation_error_response(validation: dict[str, Any]) -> dict[str, Any]:
    """Build the structured-error response for a failed intake validation.

    Forwards both the legacy ``errors`` list (for backward compatibility)
    and the ``field_errors`` map so callers can
    address the offending fields directly.
    """
    field_errors = validation.get("field_errors", {})
    response = make_error_response(
        ErrorCode.VALIDATION_FAILED,
        message="Intake validation failed",
        details={
            "validation_errors": validation["errors"],
            "field_errors": field_errors,
        },
    )
    response["errors"] = validation["errors"]
    response["field_errors"] = field_errors
    response["campaign_id"] = None
    response["spec_id"] = None
    response["warnings"] = validation.get("warnings", [])
    return response


def _capability_error_response(
    spec: CampaignSpec,
    capabilities: BackendValidationResult,
    warnings: list[str],
) -> dict[str, Any]:
    """Build the structured-error response for a backend capability rejection.

    Capability reports are keyed by an opaque ``key`` (e.g.
    ``"acquisition_method"``, ``"backend_options.baybe.recommender"``)
    that already reads as a dotted path from the spec root, so we
    forward it as-is into the field_errors map.
    """
    unsupported_reports = [{"key": r.key, "reason": r.reason} for r in capabilities.unsupported]
    capability_field_errors: dict[str, list[str]] = {}
    for report in capabilities.unsupported:
        if report.reason:
            capability_field_errors.setdefault(report.key or "", []).append(report.reason)
    response = make_error_response(
        ErrorCode.VALIDATION_FAILED,
        message=(
            f"Backend '{spec.backend}' cannot handle this spec: "
            + "; ".join(r["reason"] for r in unsupported_reports if r["reason"])
        ),
        details={
            "backend": spec.backend,
            "unsupported": unsupported_reports,
            "field_errors": capability_field_errors,
        },
    )
    response["campaign_id"] = None
    response["spec_id"] = None
    response["errors"] = [r["reason"] for r in unsupported_reports if r["reason"]]
    response["field_errors"] = capability_field_errors
    response["warnings"] = warnings
    return response


@with_response_metadata
async def create_campaign_operation(
    intake_data: CampaignIntakeInput | dict[str, Any],
    owner_id: str,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    *,
    session: AsyncSession | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create a new campaign from validated intake data.

    This contains the full business logic for campaign creation,
    independent of any transport protocol (MCP, REST, etc.).

    Args:
        intake_data: Campaign intake specification. Accepts either a
            validated :class:`CampaignIntakeInput` (the REST path)
            or a raw dict (the MCP-boundary path, where validation
            is deferred to :func:`validate_intake_operation` so the
            failure path renders as our structured ``field_errors``
            envelope instead of a FastMCP ``ToolError``).
        owner_id: UUID string identifying the campaign owner.
        verbosity: Response detail level.
        session: Optional SQLAlchemy session to reuse. When supplied,
            all DB writes happen on that session and the caller owns
            the commit boundary — used by ``apply_idempotency``'s
            session-aware path so the campaign write and the cache
            finalize commit atomically. Omit to keep the operation
            self-contained.
        dry_run: If True, run validation + capability checks and return
            a preview without persisting the campaign. ``campaign_id``
            and ``spec_id`` are omitted from the preview because no
            entity is created; the response carries ``dry_run: True``
            and a ``preview`` block summarizing what *would* be written.

    Returns:
        Formatted response dictionary with campaign details.
    """
    logger.info(
        "Creating campaign for owner_id=%s, verbosity=%s, dry_run=%s",
        owner_id,
        verbosity,
        dry_run,
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
        return _intake_validation_error_response(validation)

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

    # Resolve "auto" backend to a concrete backend name. Offloaded to a
    # worker thread because resolving "auto" loads every candidate
    # backend (torch / baybe imports) on the first call.
    raw_backend = spec_data.get("backend", "auto")
    spec_data["backend"] = await asyncio.to_thread(resolve_backend_name, raw_backend, spec_data)

    spec = _build_spec_from_dict(spec_data)
    warnings: list[str] = validation.get("warnings", [])

    # Ask the backend whether it can handle this spec — surface warnings AND
    # enforce typed-option/feature capability. ``resolve_backend_name("auto",
    # ...)`` already routes around incompatible backends; the explicit-backend
    # path also has to fail-fast on UNSUPPORTED reports so misshaped BayBE
    # ``parameter_options`` / ``backend_options`` cannot reach the suggestion
    # path.
    backend = await get_backend_async(spec.backend)
    opt_spec = campaign_spec_to_optimization_spec(spec)
    capabilities = backend.validate_capabilities(opt_spec)
    if not capabilities.is_compatible:
        logger.warning(
            "Campaign creation rejected: backend %s reports %d unsupported items",
            spec.backend,
            len(capabilities.unsupported),
        )
        return _capability_error_response(spec, capabilities, warnings)
    warnings.extend(capabilities.warnings)

    if dry_run:
        logger.info(
            "Campaign create dry-run validated: name=%s, params=%d, objectives=%d",
            spec.name,
            len(spec.parameters),
            len(spec.objectives),
        )
        return {
            "success": True,
            "dry_run": True,
            "campaign_id": None,
            "spec_id": None,
            "campaign_name": spec.name,
            "warnings": warnings,
            "errors": [],
            "field_errors": {},
            "preview": {
                "name": spec.name,
                "backend": spec.backend,
                "n_parameters": len(spec.parameters),
                "n_objectives": len(spec.objectives),
                "n_constraints": (len(spec.constraints) if spec.constraints else 0),
                "batch_size": spec.batch_size,
            },
        }

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
    # otherwise open and commit our own. Arming the counter as an
    # ``after_commit`` hook (rather than calling it after the ``async
    # with`` block) keeps the metric truthful for both the
    # operation-owns-the-session path *and* the idempotency-owns-the-
    # session path, where the outer caller commits and a downstream
    # rollback would otherwise leave the counter inflated.
    from bo_mcp_server.metrics import record_campaign_created_after_commit

    async with session_scope(session) as db:
        spec_repo = CampaignSpecRepository(db)
        campaign_repo = CampaignRepository(db)

        await spec_repo.save(spec, spec_id)
        await campaign_repo.save(campaign)
        record_campaign_created_after_commit(db, spec.backend)

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
        "field_errors": {},
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
