"""Tests that REST request schemas reject unknown fields.

Reference: FastAPI / Pydantic recommend ``extra='forbid'`` so unknown
fields raise ``ValidationError`` at the transport boundary instead of
being silently dropped. See
https://docs.pydantic.dev/latest/api/config/#pydantic.ConfigDict.extra
and the FastAPI extra-data tutorial:
https://fastapi.tiangolo.com/tutorial/body/.
"""

import pytest
from pydantic import ValidationError

from api.schemas.campaign import (
    BatchStatusRequest,
    CampaignCreate,
    CampaignLifecycleRequest,
    CampaignQueryRequest,
    CompareCampaignsRequest,
    TransferCandidatesRequest,
    ValidateIntakeRequest,
)
from api.schemas.result import ResultBatchCreate, ResultCreate, ResultQueryRequest
from api.schemas.suggestion import (
    SuggestionQueryRequest,
    SuggestionStatusUpdateRequest,
)

VALID_INTAKE = {
    "name": "T",
    "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
    "objectives": [{"name": "y", "direction": "minimize"}],
}


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (CampaignCreate, {"intake": VALID_INTAKE, "rogue": 1}),
        (ValidateIntakeRequest, {"intake": VALID_INTAKE, "rogue": 1}),
        (CampaignQueryRequest, {"limit": 10, "rogue": 1}),
        (CampaignLifecycleRequest, {"action": "pause", "rogue": 1}),
        (BatchStatusRequest, {"campaign_ids": ["x"], "rogue": 1}),
        (CompareCampaignsRequest, {"campaign_ids": ["x", "y"], "rogue": 1}),
        (TransferCandidatesRequest, {"similarity_threshold": 0.5, "rogue": 1}),
        (
            ResultCreate,
            {"parameter_values": {"x": 1.0}, "objective_values": {"y": 2.0}, "rogue": 1},
        ),
        (
            ResultBatchCreate,
            {"results": [], "source": "api", "rogue": 1},
        ),
        (ResultQueryRequest, {"limit": 5, "rogue": 1}),
        (SuggestionStatusUpdateRequest, {"status": "accepted", "rogue": 1}),
        (SuggestionQueryRequest, {"limit": 5, "rogue": 1}),
    ],
)
def test_request_schemas_reject_unknown_fields(schema: type, payload: dict) -> None:
    """Unknown fields must surface as ``extra_forbidden`` validation errors."""
    with pytest.raises(ValidationError) as exc_info:
        schema(**payload)
    types = {err["type"] for err in exc_info.value.errors()}
    assert "extra_forbidden" in types
