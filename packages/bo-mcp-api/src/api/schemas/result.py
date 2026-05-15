"""Result schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ResultCreate(BaseModel):
    """Result creation input.

    The optional ``measurement_uncertainty`` mirrors
    :class:`bo_mcp_server.domain.ResultSubmissionInput` so REST callers
    can supply per-objective noise estimates (one stddev per declared
    objective). When omitted, the engine falls back to learned noise as
    if the field had been left out at MCP intake.
    """

    parameter_values: dict[str, Any]
    objective_values: dict[str, float]
    suggestion_id: str | None = None
    measurement_uncertainty: dict[str, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultBatchCreate(BaseModel):
    """Batch result creation request."""

    results: list[ResultCreate]
    source: str = Field(default="api", pattern="^(gui|file_upload|api)$")


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

    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    verbosity: str = "standard"


class ResultQueryResponse(BaseModel):
    """Result query response with pagination envelope."""

    success: bool
    results: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    limit: int = 50
    offset: int = 0
    errors: list[str] = Field(default_factory=list)


class ResultSubmitResponse(BaseModel):
    """Response for result submission.

    ``field_errors`` mirrors the MCP envelope so REST callers can
    target the offending field by dotted path
    (e.g. ``results[5].objective_values``).
    """

    success: bool
    result_ids: list[str]
    errors: list[str]
    warnings: list[str]
    field_errors: dict[str, list[str]] = Field(default_factory=dict)
