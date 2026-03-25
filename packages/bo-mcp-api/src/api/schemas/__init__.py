"""API schemas."""

from api.schemas.campaign import (
    CampaignCreate,
    CampaignResponse,
)
from api.schemas.intake import IntakeData
from api.schemas.result import ResultCreate, ResultResponse
from api.schemas.suggestion import SuggestionResponse

__all__ = [
    "CampaignCreate",
    "CampaignResponse",
    "IntakeData",
    "ResultCreate",
    "ResultResponse",
    "SuggestionResponse",
]
