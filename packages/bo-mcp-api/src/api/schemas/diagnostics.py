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

    The declared fields carry defaults only so the OpenAPI schema and
    construction stay convenient; the route serializes with
    ``response_model_exclude_unset=True`` so a verbosity projection that
    omits a declared key (the ``minimal`` projection drops
    ``campaign_status`` / ``n_pending_suggestions`` / ``warnings``) is
    echoed exactly — the model never re-introduces it as a default, so
    REST stays byte-equal to the MCP projection.
    """

    model_config = ConfigDict(extra="allow")

    success: bool
    campaign_status: str | None = None
    iteration: int | None = None
    n_results: int | None = None
    n_pending_suggestions: int | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
