"""API schemas."""

from api.schemas.campaign import (
    BatchStatusRequest,
    BatchStatusResponse,
    CampaignCreate,
    CampaignCreateResponse,
    CampaignLifecycleRequest,
    CampaignLifecycleResponse,
    CampaignListResponse,
    CampaignResponse,
    CompareCampaignsRequest,
    CompareCampaignsResponse,
    TransferCandidatesRequest,
    TransferCandidatesResponse,
)
from api.schemas.common import VerbosityLevel
from api.schemas.intake import IntakeData
from api.schemas.result import ResultCreate, ResultResponse, ResultSubmitResponse
from api.schemas.suggestion import (
    SuggestionExplanationResponse,
    SuggestionResponse,
    SuggestionsGenerateResponse,
)

__all__ = [
    "BatchStatusRequest",
    "BatchStatusResponse",
    "CampaignCreate",
    "CampaignCreateResponse",
    "CampaignLifecycleRequest",
    "CampaignLifecycleResponse",
    "CampaignListResponse",
    "CampaignResponse",
    "CompareCampaignsRequest",
    "CompareCampaignsResponse",
    "IntakeData",
    "ResultCreate",
    "ResultResponse",
    "ResultSubmitResponse",
    "SuggestionExplanationResponse",
    "SuggestionResponse",
    "SuggestionsGenerateResponse",
    "TransferCandidatesRequest",
    "TransferCandidatesResponse",
    "VerbosityLevel",
]
