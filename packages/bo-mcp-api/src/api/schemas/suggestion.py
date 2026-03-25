"""Suggestion schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


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


class SuggestionExplanationResponse(BaseModel):
    """Response for suggestion explanation."""

    success: bool
    explanation: str | None = None
    provenance: SuggestionProvenance | None = None
    errors: list[str]
