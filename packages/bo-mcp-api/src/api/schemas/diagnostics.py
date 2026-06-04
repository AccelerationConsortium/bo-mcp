"""Diagnostics response schema."""

from pydantic import ConfigDict, Field

from api.schemas.common import ResponseEnvelope


class DiagnosticsResponse(ResponseEnvelope):
    """Campaign diagnostics response envelope.

    Mirrors the MCP-side ``DiagnosticsResponse`` (the one
    :class:`bo_mcp_server.response_formatter` model that is a
    passthrough). The declared fields pin the stable top-level keys an
    agent consumes — turning the previously opaque ``object`` in the
    OpenAPI schema into a real shape — while ``extra="allow"`` lets the
    deep, backend- and verbosity-dependent metric blocks (LOO-CV,
    hyperparameters, hypervolume history, per-section health/objective
    payloads) and the response ``_metadata`` envelope flow through
    verbatim.

    Like the sibling envelope responses (compare / transfer / batch
    status), the declared fields carry defaults so the REST shape stays
    stable across verbosity levels; the ``minimal`` projection simply
    leaves the standard-only keys at their defaults.
    """

    model_config = ConfigDict(extra="allow")

    success: bool
    campaign_status: str | None = None
    iteration: int | None = None
    n_results: int | None = None
    n_pending_suggestions: int | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
