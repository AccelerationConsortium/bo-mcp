"""Campaign schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from api.schemas.common import VerbosityLevel
from api.schemas.intake import IntakeData

# ``extra="forbid"`` is applied to every request schema in this module so
# an unknown field — typically a typo or a not-yet-supported key — raises
# 422 at the transport boundary instead of being silently dropped. Response
# schemas remain permissive because the MCP response formatter splices a
# ``_metadata`` envelope into every payload before it reaches the
# response model.
_FORBID_EXTRA: ConfigDict = ConfigDict(extra="forbid")


class CampaignCreate(BaseModel):
    """Campaign creation request."""

    model_config = _FORBID_EXTRA

    intake: IntakeData


class CampaignResponse(BaseModel):
    """Campaign response schema."""

    id: str
    spec_id: str
    name: str
    description: str
    status: str
    iteration: int
    created_at: datetime
    updated_at: datetime
    n_parameters: int
    n_objectives: int


class CampaignCreateResponse(BaseModel):
    """Campaign creation response."""

    success: bool
    campaign_id: str | None = None
    spec_id: str | None = None
    warnings: list[str] = []
    errors: list[str]


class CampaignListResponse(BaseModel):
    """Campaign list response."""

    campaigns: list[CampaignResponse]
    total: int


class ValidateIntakeRequest(BaseModel):
    """Intake validation request (dry-run, no campaign created)."""

    model_config = _FORBID_EXTRA

    intake: IntakeData


class ValidateIntakeResponse(BaseModel):
    """Intake validation response."""

    valid: bool
    errors: list[str]
    warnings: list[str] = []
    spec_summary: dict[str, Any] | None = None


class CapabilitiesResponse(BaseModel):
    """Backend capabilities response."""

    backend: str
    supported_features: list[str]
    server_version: str


class CampaignQueryRequest(BaseModel):
    """Campaign query request with filtering and pagination."""

    model_config = _FORBID_EXTRA

    status: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    verbosity: str = "standard"


class CampaignQueryResponse(BaseModel):
    """Campaign query response with pagination envelope."""

    success: bool
    campaigns: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    limit: int = 20
    offset: int = 0
    errors: list[str] = Field(default_factory=list)


class CampaignLifecycleRequest(BaseModel):
    """Lifecycle action request."""

    model_config = _FORBID_EXTRA

    action: str = Field(pattern="^(pause|resume|terminate)$")


class CampaignLifecycleResponse(BaseModel):
    """Lifecycle action response."""

    success: bool
    campaign_id: str
    status: str | None = None
    previous_status: str | None = None
    errors: list[str] = Field(default_factory=list)


class BatchStatusRequest(BaseModel):
    """Batch status request."""

    model_config = _FORBID_EXTRA

    campaign_ids: list[str]
    verbosity: VerbosityLevel = VerbosityLevel.MINIMAL


class BatchStatusResponse(BaseModel):
    """Batch status response."""

    success: bool
    campaigns: dict[str, dict[str, Any]] = Field(default_factory=dict)
    failed_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CompareCampaignsRequest(BaseModel):
    """Campaign comparison request."""

    model_config = _FORBID_EXTRA

    campaign_ids: list[str]
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD


class CompareCampaignsResponse(BaseModel):
    """Campaign comparison response."""

    success: bool
    campaigns: list[dict[str, Any]] = Field(default_factory=list)
    comparison: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    n_campaigns: int | None = None
    best_performer: str | None = None
    recommendation: str | None = None


class TransferCandidatesRequest(BaseModel):
    """Transfer candidate discovery request.

    ``parameter_aliases`` mirrors the MCP tool's optional alias map so
    REST callers can bridge parameter-name drift across campaigns
    (e.g. ``{"temperature": ["temp_c", "temp_celsius"]}`` unifies all
    three names when computing parameter-set and bounds overlap).
    """

    model_config = _FORBID_EXTRA

    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_candidates: int = Field(default=5, ge=1)
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD
    parameter_aliases: dict[str, list[str]] | None = None


class TransferCandidatesResponse(BaseModel):
    """Transfer candidate discovery response."""

    success: bool
    target_campaign: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    overall_recommendation: str | None = None
    errors: list[str] = Field(default_factory=list)
    n_candidates: int | None = None
    top_candidate_id: str | None = None
    top_similarity: float | None = None
    recommendation: str | None = None
