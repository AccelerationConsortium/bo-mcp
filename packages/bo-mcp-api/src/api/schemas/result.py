"""Result schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from api.limits import MAX_BATCH_RESULTS
from api.schemas.common import ResponseEnvelope, VerbosityLevel
from bo_mcp_server.client import FiniteFloat, ResultMetadata

# ``extra="forbid"`` is applied to request schemas so typos / not-yet-supported
# keys raise 422 instead of being silently dropped. Response schemas remain
# permissive because the MCP response formatter splices a ``_metadata``
# envelope into every payload before it reaches the response model.
_FORBID_EXTRA: ConfigDict = ConfigDict(extra="forbid")


class ResultCreate(BaseModel):
    """Result creation input.

    The optional ``measurement_uncertainty`` mirrors
    :class:`bo_mcp_server.domain.ResultSubmissionInput` so REST callers
    can supply per-objective noise estimates (one stddev per declared
    objective). When omitted, the engine falls back to learned noise as
    if the field had been left out at MCP intake.

    ``objective_values`` uses the shared :data:`FiniteFloat` value type:
    NaN/±inf measurements would fail every subsequent model fit and
    cannot be deleted once persisted, so they are rejected with a 422
    at the schema boundary — same contract as MCP intake.
    """

    model_config = _FORBID_EXTRA

    parameter_values: dict[str, Any]
    objective_values: dict[str, FiniteFloat]
    suggestion_id: str | None = None
    measurement_uncertainty: dict[str, float] | None = None
    metadata: ResultMetadata = Field(default_factory=ResultMetadata)


class ResultBatchCreate(BaseModel):
    """Batch result creation request.

    ``results`` is bounded by :data:`api.limits.MAX_BATCH_RESULTS` so a
    single POST cannot pin a worker behind validating tens of
    thousands of rows.

    ``force`` mirrors the MCP ``bo_submit_results`` tool's override:
    when ``True`` it bypasses the exact-duplicate-coordinate check so
    an optimizer-requested replicate can be submitted without first
    rejecting the suggestion (which would not exclude the coordinates
    from future generation).
    """

    model_config = _FORBID_EXTRA

    results: list[ResultCreate] = Field(..., min_length=1, max_length=MAX_BATCH_RESULTS)
    source: str = Field(default="api", pattern="^(gui|file_upload|api)$")
    force: bool = False


class ResultResponse(BaseModel):
    """Result response schema.

    ``measurement_uncertainty`` echoes back the per-objective noise std
    that was supplied at submission, ``None`` when none was provided.
    """

    id: str
    campaign_id: str
    suggestion_id: str | None
    parameter_values: dict[str, Any]
    objective_values: dict[str, float]
    source: str
    submitted_by: str
    measurement_uncertainty: dict[str, float] | None = None
    created_at: datetime


class ResultQueryRequest(BaseModel):
    """Result query request with pagination."""

    model_config = _FORBID_EXTRA

    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD


class ResultQueryResponse(ResponseEnvelope):
    """Result query response with pagination envelope."""

    success: bool
    results: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    limit: int = 50
    offset: int = 0
    errors: list[str] = Field(default_factory=list)


class ResultSubmitResponse(ResponseEnvelope):
    """Response for result submission.

    ``field_errors`` mirrors the MCP envelope so REST callers can
    target the offending field by dotted path
    (e.g. ``results[5].objective_values``).

    ``idempotency_replay`` is ``True`` when the response was served
    from the idempotency cache instead of persisting a fresh batch —
    same marker the MCP tool exposes. Without it, REST clients that
    used an Idempotency-Key on a retry could not tell the cached
    reply from a brand-new insert and would have no way to surface
    that distinction to their users.
    """

    success: bool
    result_ids: list[str]
    errors: list[str]
    warnings: list[str]
    field_errors: dict[str, list[str]] = Field(default_factory=dict)
    idempotency_replay: bool = False
