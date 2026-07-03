"""Validate intake operation - protocol-neutral business logic."""

import asyncio
import logging
from typing import Any

from pydantic import ValidationError

from bo_mcp_server.backend_context import set_campaign_backend
from bo_mcp_server.domain import (
    CampaignIntakeInput,
    CampaignSpec,
)
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.field_errors import (
    field_error_messages,
    validation_errors_to_field_errors,
)

logger = logging.getLogger(__name__)


def validate_intake_operation(
    intake_data: CampaignIntakeInput | dict[str, Any],
) -> dict[str, Any]:
    """Validate a campaign intake specification without creating a campaign.

    Returns the full canonical response with all fields populated.
    Transport layers (MCP tool, HTTP route) apply verbosity formatting.

    Args:
        intake_data: Campaign intake payload. Accepts either a
            CampaignIntakeInput instance or a raw dict (validated internally).

    Returns:
        Dictionary with:
            - valid: Whether validation passed
            - errors: List of ``path: message`` validation strings (for
              backward-compatible callers).
            - field_errors: Dotted-path map ``dict[str, list[str]]`` so
              agents can target the offending field without re-parsing
              the flat ``errors`` list. Empty on success.
            - warnings: List of warning messages.
            - spec: Full CampaignSpec as dict (if valid), or None.
    """
    intake_name = intake_data.name if isinstance(intake_data, CampaignIntakeInput) else "<no name>"
    logger.debug("Validating intake data: %s", intake_name)

    try:
        intake = (
            intake_data
            if isinstance(intake_data, CampaignIntakeInput)
            else CampaignIntakeInput.model_validate(intake_data)
        )
        spec = CampaignSpec(**intake.model_dump())
    except ValidationError as e:
        field_errors = validation_errors_to_field_errors(e)
        errors = list(field_error_messages(field_errors))
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="Intake validation failed",
            details={
                "validation_errors": errors,
                "field_errors": field_errors,
            },
        )
        response["valid"] = False
        response["errors"] = errors
        response["field_errors"] = field_errors
        response["warnings"] = []
        response["spec"] = None
        return response

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

    return {
        "valid": True,
        "errors": [],
        "field_errors": {},
        "warnings": warnings,
        "spec": spec.to_dict(),
    }


async def validate_intake_with_capabilities(
    intake_data: CampaignIntakeInput | dict[str, Any],
) -> dict[str, Any]:
    """Validate an intake spec including backend capability checks.

    :func:`validate_intake_operation` only runs schema validation, so a
    spec could pass the dry-run and still be rejected by
    ``bo_create_campaign`` on capability grounds (BayBE substance / task
    / custom-descriptor rules). This variant mirrors the create path —
    resolve ``"auto"`` to a concrete backend, then ask that backend's
    :meth:`validate_capabilities` — so validate and create agree.

    Used by the standalone validate surfaces (MCP tool, REST route).
    ``create_campaign`` keeps calling the sync operation because it runs
    its own resolution + capability pass on the same data.
    """
    # Local import: create_campaign imports this module at import time.
    from bo_mcp_server.backend import get_backend_async, resolve_backend_name
    from bo_mcp_server.converters import campaign_spec_to_optimization_spec

    response = validate_intake_operation(intake_data)
    if not response.get("valid"):
        return response

    spec_data = dict(response["spec"])
    raw_backend = spec_data.get("backend", "auto")
    # Off-thread: resolving "auto" may cold-load candidate backends.
    resolved = await asyncio.to_thread(resolve_backend_name, raw_backend, spec_data)
    spec_data["requested_backend"] = raw_backend
    spec_data["backend"] = resolved
    set_campaign_backend(resolved)

    spec = CampaignSpec.model_validate(spec_data)
    backend = await get_backend_async(resolved)
    opt_spec = campaign_spec_to_optimization_spec(spec)
    capabilities = backend.validate_capabilities(opt_spec)

    response["backend"] = resolved
    response["spec"] = spec.to_dict()

    if not capabilities.is_compatible:
        capability_errors = [
            f"{report.key}: {report.reason}" if report.key else str(report.reason)
            for report in capabilities.unsupported
        ]
        field_errors: dict[str, list[str]] = response.get("field_errors", {})
        for report in capabilities.unsupported:
            if report.reason:
                field_errors.setdefault(report.key or "", []).append(report.reason)
        logger.info(
            "Intake capability validation failed for backend %s: %d unsupported",
            resolved,
            len(capabilities.unsupported),
        )
        response["valid"] = False
        response["errors"] = [*response.get("errors", []), *capability_errors]
        response["field_errors"] = field_errors
        response["spec"] = None
        return response

    response["warnings"] = [*response.get("warnings", []), *capabilities.warnings]
    return response
