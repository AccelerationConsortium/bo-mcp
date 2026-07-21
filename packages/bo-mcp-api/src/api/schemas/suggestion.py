"""Suggestion schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from api.schemas.common import ResponseEnvelope, VerbosityLevel

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
    """Suggestion response schema.

    ``suggestion_id`` is the identity key: it is the same key the
    suggestion-query endpoint emits and the one result submission
    consumes, so its value can be copied into a
    ``POST /api/v1/results/{campaign_id}`` request without renaming.
    (Only the key copies over — the result request schema rejects the
    other suggestion fields.)
    """

    suggestion_id: str
    campaign_id: str
    parameter_values: dict[str, Any]
    status: str
    provenance: SuggestionProvenance
    created_at: datetime


class SuggestionsGenerateResponse(ResponseEnvelope):
    """Response for suggestion generation.

    ``idempotency_replay`` is ``True`` when the response was served
    from the idempotency cache instead of running a fresh generation —
    same marker the MCP tool exposes, so REST clients can distinguish
    a retry's replayed batch from newly generated suggestions.
    """

    success: bool
    suggestions: list[SuggestionResponse]
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


class SuggestionStatusUpdateResponse(ResponseEnvelope):
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


class SuggestionSummary(BaseModel):
    """One ``suggestions[]`` entry from the suggestion query endpoint.

    Superset of the minimal / standard / detailed projections built by
    the shared list-suggestions operation. ``suggestion_id`` and
    ``status`` are required — they are present at every verbosity —
    while the tier-dependent fields are optional.

    ``extra="allow"`` keeps the historical passthrough behaviour: keys
    the operation adds later still reach clients instead of being
    silently dropped, while the declared fields pin the identity key so
    a rename breaks loudly in tests.

    The query route serializes with ``response_model_exclude_unset``,
    so each verbosity keeps its exact historical wire shape instead of
    gaining ``null`` entries for every declared-but-absent optional
    field. A custom set-fields-only serializer is not an option here:
    a ``model_serializer`` replaces the model's serialization JSON
    schema with a bare object, erasing the ``required`` markers from
    OpenAPI.
    """

    model_config = ConfigDict(extra="allow")

    suggestion_id: str
    status: str
    parameter_values: dict[str, Any] | None = None
    iteration: int | None = None
    generation_method: str | None = None
    created_at: str | None = None
    batch_index: int | None = None
    acquisition_function: str | None = None
    acquisition_value: float | None = None
    model_uncertainty: float | None = None
    model_type: str | None = None
    confidence_level: str | None = None
    predicted_objectives: dict[str, Any] | None = None
    predicted_std: dict[str, Any] | None = None
    updated_at: str | None = None


class SuggestionQueryResponse(ResponseEnvelope):
    """Suggestion query response with pagination envelope.

    Serialized with ``response_model_exclude_unset``, so the route
    must set every field it wants on the wire — including
    ``schema_version``, which would otherwise be dropped as an unset
    default.
    """

    success: bool
    suggestions: list[SuggestionSummary] = Field(default_factory=list)
    total_count: int = 0
    limit: int | None = None
    offset: int = 0
    next_cursor: str | None = None
    errors: list[str] = Field(default_factory=list)


class SuggestionExplanationResponse(ResponseEnvelope):
    """Response for suggestion explanation."""

    success: bool
    explanation: str | None = None
    provenance: SuggestionProvenance | None = None
    errors: list[str]
