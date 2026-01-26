"""Campaign schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from api.schemas.intake import IntakeData


class CampaignCreate(BaseModel):
    """Campaign creation request."""

    intake: IntakeData


class CampaignValidation(BaseModel):
    """Campaign validation response."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    spec: dict[str, Any] | None = None


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
    errors: list[str]


class CampaignListResponse(BaseModel):
    """Campaign list response."""

    campaigns: list[CampaignResponse]
    total: int
