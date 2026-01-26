"""Domain models for the BO MCP service."""

from bo_mcp_server.domain.campaign import Campaign, CampaignStatus
from bo_mcp_server.domain.campaign_spec import (
    AcquisitionMethod,
    CampaignSpec,
    Constraint,
    ConstraintType,
    FidelityParameter,
    InputParameter,
    Objective,
    OutcomeConstraint,
    ParameterType,
    TransferLearningConfig,
)
from bo_mcp_server.domain.result import Result, ResultSource
from bo_mcp_server.domain.suggestion import Suggestion, SuggestionProvenance, SuggestionStatus
from bo_mcp_server.domain.user import User

__version__ = "0.1.0"

__all__ = [
    "AcquisitionMethod",
    "Campaign",
    "CampaignSpec",
    "CampaignStatus",
    "Constraint",
    "ConstraintType",
    "FidelityParameter",
    "InputParameter",
    "Objective",
    "OutcomeConstraint",
    "ParameterType",
    "Result",
    "ResultSource",
    "Suggestion",
    "SuggestionProvenance",
    "SuggestionStatus",
    "TransferLearningConfig",
    "User",
]
