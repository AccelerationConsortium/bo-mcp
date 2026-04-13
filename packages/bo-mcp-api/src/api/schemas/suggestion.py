"""Suggestion schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SuggestionProvenance(BaseModel):
    """Suggestion provenance schema."""

    iteration: int
    batch_index: int
    acquisition_value: float | None = None
    model_uncertainty: float | None = None
    generation_method: str
    # Enhanced provenance fields
    acquisition_function: str | None = None
    model_type: str | None = None
    random_seed: int | None = None
    model_version: int | None = None
    confidence_level: str | None = None
    explanation: str | None = None


class SuggestionResponse(BaseModel):
    """Suggestion response schema."""

    id: str
    campaign_id: str
    parameter_values: dict[str, Any]
    status: str
    provenance: SuggestionProvenance
    created_at: datetime


class SuggestionsGenerateResponse(BaseModel):
    """Response for suggestion generation."""

    success: bool
    suggestions: list[SuggestionResponse]
    iteration: int | None = None
    errors: list[str]


class SuggestionStatusUpdateRequest(BaseModel):
    """Request to update a suggestion's status."""

    status: str = Field(pattern="^(accepted|rejected|expired)$")


class SuggestionStatusUpdateResponse(BaseModel):
    """Response for suggestion status update."""

    success: bool
    suggestion_id: str | None = None
    status: str | None = None
    previous_status: str | None = None
    errors: list[str] = []


class SuggestionQueryRequest(BaseModel):
    """Suggestion query request with filtering and pagination."""

    status_filter: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    verbosity: str = "standard"


class SuggestionQueryResponse(BaseModel):
    """Suggestion query response with pagination envelope."""

    success: bool
    suggestions: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    limit: int | None = None
    offset: int = 0
    errors: list[str] = Field(default_factory=list)


class SuggestionExplanationResponse(BaseModel):
    """Response for suggestion explanation."""

    success: bool
    explanation: str | None = None
    provenance: SuggestionProvenance | None = None
    errors: list[str]
