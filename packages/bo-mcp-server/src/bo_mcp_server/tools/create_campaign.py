"""Create campaign tool wrapper for MCP."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.client import AuthenticationConfigurationError, resolve_mcp_user
from bo_mcp_server.domain.intake_models import INTAKE_INPUT_JSON_SCHEMA
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.field_errors import shape_envelope
from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.operations.create_campaign import create_campaign_operation
from bo_mcp_server.operations.idempotency_wrapper import (
    canonical_create_campaign_payload,
)
from bo_mcp_server.response_formatter import attach_response_metadata
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import NON_IDEMPOTENT_MUTATION
from bo_mcp_server.trace_context import bind_trace_id

# The tool boundary is typed ``Any`` (not ``dict[str, Any]``) so that
# FastMCP's pre-call Pydantic validator cannot intercept the failure
# when callers send the wrong outer shape (e.g. a string instead of
# an object). Without this, FastMCP would raise an opaque ``ToolError``
# before our wrapper runs and agents would lose both the structured
# envelope and the dotted-path ``field_errors`` map. The shape check
# now lives inside the wrapper. Schema discoverability is preserved
# by splicing the full ``CampaignIntakeInput`` JSON schema into
# ``json_schema_extra`` so ``tools/list`` still advertises the rich
# nested structure.
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

_INTAKE_BOUNDARY_DEFAULTS: dict[str, Any] = {
    "campaign_id": None,
    "spec_id": None,
    "warnings": [],
}
_VALIDATION_ONLY_OWNER_ID = "00000000-0000-0000-0000-000000000000"


def _check_intake_shape(intake_data: object) -> dict[str, Any] | None:
    """Return a structured envelope iff ``intake_data`` is not object-shaped.

    The Pydantic models down the call chain produce per-field
    ``field_errors`` for malformed sub-fields, but their entry points
    expect a mapping. Anything else (string, number, list) would raise
    an opaque ``ToolError`` if we forwarded it. Catching the
    container-shape failure here keeps the agent-facing contract
    identical to inner-field failures.
    """
    if isinstance(intake_data, (Mapping, BaseModel)):
        return None
    return shape_envelope(
        "intake_data",
        f"Input should be an object, got {type(intake_data).__name__}",
        extra=_INTAKE_BOUNDARY_DEFAULTS,
    )


def _mcp_identity_error(exc: AuthenticationConfigurationError) -> dict[str, Any]:
    """Return a structured error when MCP cannot resolve a current user."""
    return attach_response_metadata(
        make_error_response(
            ErrorCode.INTERNAL_ERROR,
            message=str(exc),
            details={"transport": "mcp", "missing_identity": True},
        )
    )


async def _validate_intake_before_identity(
    intake_data: IntakePayload,
    verbosity: Literal["minimal", "standard", "detailed"],
    trace_id: str | None,
) -> dict[str, Any] | None:
    """Return a validation envelope before requiring MCP identity, if invalid."""
    with bind_trace_id(trace_id):
        shape_error = _check_intake_shape(intake_data)
        if shape_error is not None:
            return attach_response_metadata(shape_error)

        validation_result = await create_campaign_operation(
            intake_data=intake_data,
            owner_id=_VALIDATION_ONLY_OWNER_ID,
            verbosity=verbosity,
            dry_run=True,
        )
        if validation_result.get("success") is False:
            return validation_result

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
) -> dict[str, Any]:
    """Create a new optimization campaign from validated intake data.

    Workflow: Call once at the start. Use bo_validate_intake first to dry-run
    check your spec, and bo_list_capabilities to verify feature support.

    The MCP transport resolves the campaign owner internally from the current
    BO-MCP user identity. Agents must not provide database user ids.
    """
    validation_error = await _validate_intake_before_identity(
        intake_data=intake_data,
        verbosity=verbosity,
        trace_id=trace_id,
    )
    if validation_error is not None:
        return validation_error

    try:
        user = await resolve_mcp_user()
    except AuthenticationConfigurationError as exc:
        return _mcp_identity_error(exc)

    return await _create_campaign_for_owner(
        intake_data=intake_data,
        owner_id=str(user.id),
        verbosity=verbosity,
        idempotency_key=idempotency_key,
        dry_run=dry_run,
        trace_id=trace_id,
    )
