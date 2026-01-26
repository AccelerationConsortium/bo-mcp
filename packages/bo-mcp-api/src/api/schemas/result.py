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


class ResultSubmitResponse(BaseModel):
    """Response for result submission."""

    success: bool
    result_ids: list[str]
    errors: list[str]
    warnings: list[str]
