"""Campaign schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from api.schemas.common import VerbosityLevel
from api.schemas.intake import IntakeData


class CampaignCreate(BaseModel):
    """Campaign creation request."""

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


class CampaignLifecycleRequest(BaseModel):
    """Lifecycle action request."""

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
    """Transfer candidate discovery request."""

    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_candidates: int = Field(default=5, ge=1)
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD


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
