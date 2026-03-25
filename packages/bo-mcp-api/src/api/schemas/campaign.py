"""Campaign schemas."""

from datetime import datetime

from pydantic import BaseModel

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
