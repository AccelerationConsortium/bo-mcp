"""Domain models for the BO MCP service."""

from bo_mcp_server.domain.campaign import Campaign, CampaignStatus
from bo_mcp_server.domain.campaign_spec import (
    AcquisitionMethod,
    Bounds,
    CampaignSpec,
    Constraint,
    ConstraintType,
    FidelityParameter,
    InputParameter,
    Objective,
    OutcomeConstraint,
    ParameterType,
    SaasboConfig,
    TransferLearningConfig,
    TurboConfig,
)
from bo_mcp_server.domain.event import Event, EventType
from bo_mcp_server.domain.intake_models import CampaignIntakeInput, ResultSubmissionInput
from bo_mcp_server.domain.result import Result, ResultSource
from bo_mcp_server.domain.suggestion import Suggestion, SuggestionProvenance, SuggestionStatus
from bo_mcp_server.domain.user import User

__version__ = "0.1.0"

__all__ = [
    "AcquisitionMethod",
    "Campaign",
    "Event",
    "EventType",
    "CampaignIntakeInput",
    "CampaignSpec",
    "CampaignStatus",
    "Bounds",
    "Constraint",
    "ConstraintType",
    "FidelityParameter",
    "InputParameter",
    "Objective",
    "OutcomeConstraint",
    "ParameterType",
    "Result",
    "ResultSubmissionInput",
    "ResultSource",
    "Suggestion",
    "SuggestionProvenance",
    "SuggestionStatus",
    "TransferLearningConfig",
    "TurboConfig",
    "SaasboConfig",
    "User",
]
