"""Create campaign tool wrapper for MCP."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.domain.intake_models import INTAKE_INPUT_JSON_SCHEMA
from bo_mcp_server.field_errors import shape_envelope
from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.operations.create_campaign import create_campaign_operation
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


def _payload_for_intake(intake_data: Any) -> dict[str, Any]:
    """Return a JSON-safe dict representation of the intake payload.

    Accepts either an already-validated :class:`CampaignIntakeInput` (the
    legacy in-process path used by tests pre-1.53 follow-up) or a raw
    dict (the MCP-boundary path).
    """
    if isinstance(intake_data, BaseModel):
        return intake_data.model_dump()
    return dict(intake_data)


def _check_intake_shape(intake_data: Any) -> dict[str, Any] | None:
    """Return a structured envelope iff ``intake_data`` is not object-shaped.

    The Pydantic models down the call chain produce per-field
    ``field_errors`` for malformed sub-fields, but their entry points
    expect a mapping. Anything else (string, number, list) would raise
    an opaque ``ToolError`` if we forwarded it. Catching the
    container-shape failure here keeps the agent-facing contract
    identical to inner-field failures.
    """
    if isinstance(intake_data, Mapping) or isinstance(intake_data, BaseModel):
        return None
    return shape_envelope(
        "intake_data",
        f"Input should be an object, got {type(intake_data).__name__}",
        extra=_INTAKE_BOUNDARY_DEFAULTS,
    )


@mcp.tool(name="bo_create_campaign", annotations=NON_IDEMPOTENT_MUTATION)
async def create_campaign(
    intake_data: IntakePayload,
    owner_id: str,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    idempotency_key: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Create a new optimization campaign from validated intake data.

    Workflow: Call once at the start. Use bo_validate_intake first to dry-run
    check your spec, and bo_list_capabilities to verify feature support.

    Args:
        intake_data: Campaign intake payload validated via CampaignIntakeInput.
        owner_id: UUID of the user creating the campaign.
        verbosity: Response verbosity level (minimal, standard, detailed).
        idempotency_key: Optional client-supplied key (recommended: UUIDv7
            per logical operation). If supplied and the same key was used
            in the last 24 hours with an identical payload, the prior
            response is returned verbatim with ``idempotency_replay:
            True``. Re-using a key with a different payload returns a
            ``VALIDATION_FAILED`` error with
            ``details.idempotency_conflict=True``.
        dry_run: If True, run intake validation and backend capability
            checks but do not persist the campaign. The response carries
            ``dry_run: True`` and a ``preview`` block. Dry-runs bypass
            the idempotency cache so the slot stays free for a real
            create. For pure schema validation, prefer
            ``bo_validate_intake`` (READ_ONLY annotation).
        trace_id: Optional workflow trace id. Bound for the duration of
            the call so audit events + response ``_metadata.trace_id``
            echo it. Use a stable id (e.g. UUIDv7) across the full
            multi-step workflow.

    Returns:
        Dictionary with success, campaign_id, spec_id, errors.
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

        request_payload = {
            "intake_data": _payload_for_intake(intake_data),
            "owner_id": owner_id,
            "verbosity": verbosity,
        }

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
