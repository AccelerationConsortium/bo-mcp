"""Create campaign tool wrapper for MCP."""

from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.domain import CampaignIntakeInput
from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.operations.create_campaign import create_campaign_operation
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import NON_IDEMPOTENT_MUTATION


def _payload_for_intake(intake_data: CampaignIntakeInput | dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe dict representation of the intake payload.

    Accepts either an already-validated :class:`CampaignIntakeInput` (the
    typed path used by the MCP transport) or a raw dict (the path used by
    tests that drive the tool function directly).
    """
    if isinstance(intake_data, BaseModel):
        return intake_data.model_dump()
    return dict(intake_data)


@mcp.tool(name="bo_create_campaign", annotations=NON_IDEMPOTENT_MUTATION)
async def create_campaign(
    intake_data: CampaignIntakeInput,
    owner_id: str,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    idempotency_key: str | None = None,
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

    Returns:
        Dictionary with success, campaign_id, spec_id, errors.
    """
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
