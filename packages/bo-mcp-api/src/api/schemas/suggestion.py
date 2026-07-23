"""Suggestion schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from api.schemas.common import MutationEnvelope, ResponseEnvelope, VerbosityLevel

# ``extra="forbid"`` is applied to request schemas so typos / not-yet-supported
# keys raise 422 instead of being silently dropped. Response schemas remain
# permissive because the MCP response formatter splices a ``_metadata``
# envelope into every payload before it reaches the response model.
_FORBID_EXTRA: ConfigDict = ConfigDict(extra="forbid")

ManualSuggestionStatus = Literal["accepted", "rejected", "expired"]


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


class SuggestionsGenerateResponse(MutationEnvelope):
    """Response for suggestion generation.

    ``idempotency_replay`` is ``True`` when the response was served
    from the idempotency cache instead of running a fresh generation —
    same marker the MCP tool exposes, so REST clients can distinguish
    a retry's replayed batch from newly generated suggestions.
    """

    success: bool
    suggestions: list[SuggestionResponse] = Field(default_factory=list)
    iteration: int | None = None
    errors: list[str]
    idempotency_replay: bool = False


class SuggestionStatusUpdateRequest(BaseModel):
    """Request to update a suggestion's status."""

    model_config = _FORBID_EXTRA

    status: ManualSuggestionStatus = Field(
        description=(
            'Manual suggestion status transition. Use "accepted", "rejected", '
            'or "expired" here. Do not set "completed" directly; a suggestion '
            "becomes completed automatically when a result is submitted with "
            "its suggestion_id."
        ),
        examples=["accepted", "rejected", "expired"],
    )
    dry_run: bool = Field(
        default=False,
        description=(
            "Validate the transition and return a preview (from/to status) "
            "without committing — same semantics as the MCP "
            "bo_update_suggestion_status tool."
        ),
    )


class SuggestionStatusUpdateResponse(MutationEnvelope):
    """Response for suggestion status update."""

    success: bool
    suggestion_id: str | None = None
    status: str | None = None
    previous_status: str | None = None
    errors: list[str] = Field(default_factory=list)


class SuggestionQueryRequest(BaseModel):
    """Suggestion query request with filtering and pagination."""

    model_config = _FORBID_EXTRA

    status_filter: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD


class SuggestionQueryResponse(ResponseEnvelope):
    """Suggestion query response with pagination envelope."""

    success: bool
    suggestions: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    limit: int | None = None
    offset: int = 0
    errors: list[str] = Field(default_factory=list)


class SuggestionExplanationResponse(ResponseEnvelope):
    """Response for suggestion explanation."""

    success: bool
    explanation: str | None = None
    provenance: SuggestionProvenance | None = None
    errors: list[str]
