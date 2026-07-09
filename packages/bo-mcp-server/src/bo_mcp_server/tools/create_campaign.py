"""Create campaign tool wrapper for MCP."""

from typing import Any, Literal, cast

from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.client import AuthenticationConfigurationError, resolve_mcp_user
from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.operations.create_campaign import (
    _intake_validation_error_response,
    create_campaign_operation,
)
from bo_mcp_server.operations.idempotency_wrapper import (
    canonical_create_campaign_payload,
)
from bo_mcp_server.operations.validate_intake import validate_intake_operation
from bo_mcp_server.response_formatter import attach_response_metadata
from bo_mcp_server.server import mcp
from bo_mcp_server.tool_boundary import CREATE_CAMPAIGN_ENVELOPE_EXTRA
from bo_mcp_server.tools.annotations import NON_IDEMPOTENT_MUTATION
from bo_mcp_server.tools.common import IntakePayload, check_intake_shape, mcp_identity_error
from bo_mcp_server.tools.response_models import CreateCampaignResponse
from bo_mcp_server.trace_context import bind_trace_id

_INTAKE_BOUNDARY_DEFAULTS = CREATE_CAMPAIGN_ENVELOPE_EXTRA


def _check_intake_shape(intake_data: object) -> dict[str, Any] | None:
    return check_intake_shape(intake_data, extra=_INTAKE_BOUNDARY_DEFAULTS)


def _mcp_identity_error(exc: AuthenticationConfigurationError) -> dict[str, Any]:
    return mcp_identity_error(exc)


def _validate_intake_before_identity(
    intake_data: IntakePayload,
    trace_id: str | None,
) -> dict[str, Any] | None:
    """Return an intake-validation envelope before requiring MCP identity, if invalid.

    Runs only the cheap, backend-free intake validation (container shape +
    schema) so a malformed payload yields an actionable ``field_errors``
    envelope even when MCP identity is unconfigured — and without paying for
    backend capability checks twice. Capability validation happens once, in the
    real create after identity resolves. Reuses
    :func:`_intake_validation_error_response` so the envelope is identical to
    the one the operation layer emits for the same failure.
    """
    with bind_trace_id(trace_id):
        shape_error = _check_intake_shape(intake_data)
        if shape_error is not None:
            return attach_response_metadata(shape_error)

        validation = validate_intake_operation(intake_data)
        if not validation["valid"]:
            return attach_response_metadata(_intake_validation_error_response(validation))

    return None


async def _create_campaign_for_owner(
    intake_data: IntakePayload,
    owner_id: str,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    idempotency_key: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Create a campaign for a concrete owner id.

    This helper intentionally remains importable for internal scripts/tests
    that exercise the operation layer directly. The registered MCP tool below
    resolves the owner internally and does not expose ``owner_id`` to agents.
    """
    # ``bind_trace_id`` wraps the entire wrapper body — including the
    # pre-operation shape check — so a malformed payload still emits an
    # envelope whose ``_metadata.trace_id`` echoes the bound workflow
    # id. Otherwise validation envelopes would be the only tool returns
    # that drop the trace, undermining the cookbook contract.
    with bind_trace_id(trace_id):
        shape_error = _check_intake_shape(intake_data)
        if shape_error is not None:
            return attach_response_metadata(shape_error)

        if dry_run:
            return await create_campaign_operation(
                intake_data=intake_data,
                owner_id=owner_id,
                verbosity=verbosity,
                dry_run=True,
            )

        # Canonical builder normalizes raw-dict and CampaignIntakeInput
        # inputs through the same validated model dump so the hash
        # matches what REST emits — the precondition for the audit's
        # "same cache namespace as the matching MCP tool" promise.
        request_payload = canonical_create_campaign_payload(
            intake_data=intake_data,
            owner_id=owner_id,
            verbosity=verbosity,
        )

        async def run(session: AsyncSession) -> dict[str, Any]:
            # Session-aware: the operation writes the new campaign on the
            # same session the idempotency cache finalizes on, so both
            # commit atomically.
            return await create_campaign_operation(
                intake_data=intake_data,
                owner_id=owner_id,
                verbosity=verbosity,
                session=session,
            )

        return await apply_idempotency(
            tool_name="bo_create_campaign",
            idempotency_key=idempotency_key,
            request_payload=request_payload,
            executor=run,
        )


async def create_campaign(
    intake_data: IntakePayload,
    owner_id: str,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    idempotency_key: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Compatibility helper for Python callers that already have an owner id."""
    return await _create_campaign_for_owner(
        intake_data=intake_data,
        owner_id=owner_id,
        verbosity=verbosity,
        idempotency_key=idempotency_key,
        dry_run=dry_run,
        trace_id=trace_id,
    )


@mcp.tool(name="bo_create_campaign", annotations=NON_IDEMPOTENT_MUTATION)
async def _create_campaign_tool(
    intake_data: IntakePayload,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    idempotency_key: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> CreateCampaignResponse:
    """Create a new optimization campaign from validated intake data.

    Workflow: Call once at the start. Use bo_validate_intake first to dry-run
    check your spec, and bo_list_capabilities to verify feature support.

    The MCP transport resolves the campaign owner internally from the current
    BO-MCP user identity. Agents must not provide database user ids.
    """
    validation_error = _validate_intake_before_identity(
        intake_data=intake_data,
        trace_id=trace_id,
    )
    if validation_error is not None:
        return cast(CreateCampaignResponse, validation_error)

    try:
        user = await resolve_mcp_user()
    except AuthenticationConfigurationError as exc:
        return cast(CreateCampaignResponse, _mcp_identity_error(exc))

    return cast(
        CreateCampaignResponse,
        await _create_campaign_for_owner(
            intake_data=intake_data,
            owner_id=str(user.id),
            verbosity=verbosity,
            idempotency_key=idempotency_key,
            dry_run=dry_run,
            trace_id=trace_id,
        ),
    )
