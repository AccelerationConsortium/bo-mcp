"""Shared API schema primitives."""

from typing import Final

from bo_mcp_server.client import RESPONSE_SCHEMA_VERSION, VerbosityLevel
from pydantic import BaseModel, Field

# Re-export the engine-side constant under a stable name. The REST
# response envelope advertises the same version the MCP path does, so
# clients can dispatch on a single integer regardless of transport.
API_RESPONSE_SCHEMA_VERSION: Final[int] = RESPONSE_SCHEMA_VERSION


class ResponseEnvelope(BaseModel):
    """Base class for REST success **envelopes**.

    Carries the top-level ``schema_version`` so REST clients can
    detect a backwards-incompatible envelope change without parsing
    the body. The default value tracks
    :data:`bo_mcp_server.response_formatter.RESPONSE_SCHEMA_VERSION`
    so a bump on the MCP path automatically flows through to REST
    responses too.

    Scope of the ``schema_version`` contract
    ----------------------------------------

    ``schema_version`` only stamps **envelope-shaped** responses —
    payloads that wrap the operation outcome with the
    ``success``/``errors``/``warnings``/``field_errors`` shape that
    the MCP and REST transports share (campaign create / list /
    query, suggestion generate / query / status update, result
    submit / query, capabilities, batch status, compare / transfer
    candidates, lifecycle actions).

    Single-resource GET endpoints — ``GET /api/campaigns/{id}``,
    ``GET /api/suggestions/{id}``, ``GET /api/results/{id}``, and
    the list/collection variants ``GET /api/suggestions/{id}`` /
    ``GET /api/results/{id}`` that return bare arrays — are explicitly
    OUT of this contract. They return resource representations,
    not envelopes; their shape is governed by the per-route response
    models (e.g. :class:`api.schemas.campaign.CampaignResponse`),
    which evolve independently. Mixing ``schema_version`` into bare
    resource views would also break the convention that a
    representation's keys map directly to the underlying entity.

    Bump rules for the envelope contract:

    * additive fields → keep the version unchanged (forward-compatible);
    * removed / renamed / re-typed fields → bump and document.
    """

    schema_version: int = Field(default=API_RESPONSE_SCHEMA_VERSION)


# Per-route response models that are intentionally NOT envelope-shaped
# (resource representations or bare-list collection views). The
# regression test ``test_rest_schema_version.test_resource_models_are_envelope_exempt``
# pins this allowlist so the exclusion is auditable in code review and
# a future regression that wraps these in envelopes (or invents a
# new resource view that bypasses the envelope) shows up as a diff.
RESOURCE_VIEW_MODEL_NAMES: Final[frozenset[str]] = frozenset(
    {
        # Bare resource representations returned from GET-by-id routes.
        "CampaignResponse",
        "SuggestionResponse",
        "ResultResponse",
        # Nested item models that compose into envelope responses but
        # are also returned in bare-list views.
        "SuggestionProvenance",
    }
)


__all__ = [
    "API_RESPONSE_SCHEMA_VERSION",
    "RESOURCE_VIEW_MODEL_NAMES",
    "ResponseEnvelope",
    "VerbosityLevel",
]
