"""Tests that REST request schemas reject unknown fields.

Reference: FastAPI / Pydantic recommend ``extra='forbid'`` so unknown
fields raise ``ValidationError`` at the transport boundary instead of
being silently dropped. See
https://docs.pydantic.dev/latest/api/config/#pydantic.ConfigDict.extra
and the FastAPI extra-data tutorial:
https://fastapi.tiangolo.com/tutorial/body/.
"""

import pytest
from pydantic import BaseModel, ValidationError

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


def _resolve_json_schema_ref(model_schema: dict, field_schema: dict) -> dict:
    """Resolve a local Pydantic JSON-schema $ref used by enum fields."""
    ref = field_schema.get("$ref")
    if not ref:
        return field_schema

    prefix = "#/$defs/"
    assert ref.startswith(prefix)
    return model_schema["$defs"][ref.removeprefix(prefix)]


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
def test_request_schemas_reject_unknown_fields(schema: type[BaseModel], payload: dict) -> None:
    """Unknown fields must surface as ``extra_forbidden`` validation errors."""
    with pytest.raises(ValidationError) as exc_info:
        schema(**payload)
    types = {err["type"] for err in exc_info.value.errors()}
    assert "extra_forbidden" in types


def test_lifecycle_action_schema_advertises_allowed_workflow() -> None:
    """Lifecycle actions should be enum-like and explain campaign completion."""
    action_schema = CampaignLifecycleRequest.model_json_schema()["properties"]["action"]

    assert action_schema["enum"] == ["pause", "resume", "terminate"]
    assert "terminate" in action_schema["description"]
    assert "complete" in action_schema["description"]

    with pytest.raises(ValidationError) as exc_info:
        CampaignLifecycleRequest.model_validate({"action": "complete"})
    assert exc_info.value.errors()[0]["type"] == "literal_error"


def test_suggestion_status_schema_advertises_manual_transitions() -> None:
    """Suggestion status updates should make completion-by-result explicit."""
    status_schema = SuggestionStatusUpdateRequest.model_json_schema()["properties"]["status"]

    assert status_schema["enum"] == ["accepted", "rejected", "expired"]
    assert "completed" in status_schema["description"]
    assert "result" in status_schema["description"]

    with pytest.raises(ValidationError) as exc_info:
        SuggestionStatusUpdateRequest.model_validate({"status": "completed"})
    assert exc_info.value.errors()[0]["type"] == "literal_error"


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (CampaignQueryRequest, {"verbosity": "summary"}),
        (BatchStatusRequest, {"campaign_ids": ["x"], "verbosity": "summary"}),
        (CompareCampaignsRequest, {"campaign_ids": ["x", "y"], "verbosity": "summary"}),
        (TransferCandidatesRequest, {"verbosity": "summary"}),
        (ResultQueryRequest, {"verbosity": "summary"}),
        (SuggestionQueryRequest, {"verbosity": "summary"}),
    ],
)
def test_verbosity_schema_advertises_shared_enum(schema: type[BaseModel], payload: dict) -> None:
    """REST verbosity fields should match the operation-layer contract."""
    model_schema = schema.model_json_schema()
    verbosity_schema = _resolve_json_schema_ref(
        model_schema, model_schema["properties"]["verbosity"]
    )

    assert verbosity_schema["enum"] == ["minimal", "standard", "detailed"]
    assert model_schema["properties"]["verbosity"].get("default") in {"minimal", "standard"}
    assert "summary" not in verbosity_schema["enum"]

    with pytest.raises(ValidationError) as exc_info:
        schema.model_validate(payload)
    assert exc_info.value.errors()[0]["type"] == "enum"
