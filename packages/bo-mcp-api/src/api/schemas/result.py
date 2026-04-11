"""Result schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ResultCreate(BaseModel):
    """Result creation input."""

    parameter_values: dict[str, Any]
    objective_values: dict[str, float]
    suggestion_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultBatchCreate(BaseModel):
    """Batch result creation request."""

    results: list[ResultCreate]
    source: str = Field(default="api", pattern="^(gui|file_upload|api)$")


class ResultResponse(BaseModel):
    """Result response schema."""

    id: str
    campaign_id: str
    suggestion_id: str | None
    parameter_values: dict[str, Any]
    objective_values: dict[str, float]
    source: str
    submitted_by: str
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
    """Response for result submission."""

    success: bool
    result_ids: list[str]
    errors: list[str]
    warnings: list[str]
