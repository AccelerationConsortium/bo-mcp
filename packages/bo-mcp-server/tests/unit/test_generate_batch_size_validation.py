"""Bounds validation for the generation ``batch_size`` request parameter.

Generation is the most expensive mutation (a q-batch acquisition solve
superlinear in q), yet ``batch_size`` previously reached the pipeline
unvalidated: ``0`` was silently replaced by the spec default via a
falsy-``or``, negatives produced opaque backend errors, and there was
no upper bound on any transport. The shared operation now rejects
anything outside ``[1, MAX_GENERATION_BATCH_SIZE]`` with a typed
``VALIDATION_FAILED`` envelope before touching the database or the
backend; the intake schema enforces the same ceiling.

Reference: OWASP API Security Top 10, API4 "Unrestricted Resource
Consumption" — expensive operations must bound caller-supplied sizes.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from bo_mcp_server.constants import MAX_GENERATION_BATCH_SIZE
from bo_mcp_server.domain.intake_models import CampaignIntakeInput
from bo_mcp_server.operations.generate_suggestions import generate_suggestions_operation

pytestmark = pytest.mark.usefixtures("setup_database")


@pytest.fixture(autouse=True)
def _backend_must_not_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any backend load during these tests is a validation-order bug."""

    async def _fail(*_args: object, **_kwargs: object) -> None:
        msg = "backend must not be invoked for an invalid batch_size"
        raise AssertionError(msg)

    monkeypatch.setattr("bo_mcp_server.operations.generate_suggestions.get_backend_async", _fail)


@pytest.mark.parametrize("bad_batch_size", [0, -3, MAX_GENERATION_BATCH_SIZE + 1])
@pytest.mark.asyncio
async def test_out_of_range_batch_size_returns_field_errors(bad_batch_size: int) -> None:
    result = await generate_suggestions_operation(
        campaign_id=str(uuid4()),
        batch_size=bad_batch_size,
    )

    assert result["success"] is False
    assert result["error"]["code"] == "E005"
    assert "batch_size" in result["field_errors"]
    assert str(MAX_GENERATION_BATCH_SIZE) in result["field_errors"]["batch_size"][0]


@pytest.mark.parametrize("bad_batch_size", [0, -3, MAX_GENERATION_BATCH_SIZE + 1])
@pytest.mark.asyncio
async def test_dry_run_rejects_out_of_range_batch_size_too(bad_batch_size: int) -> None:
    result = await generate_suggestions_operation(
        campaign_id=str(uuid4()),
        batch_size=bad_batch_size,
        dry_run=True,
    )

    assert result["success"] is False
    assert "batch_size" in result["field_errors"]


@pytest.mark.asyncio
async def test_valid_batch_size_passes_validation() -> None:
    """A well-formed request proceeds past validation to the campaign lookup."""
    result = await generate_suggestions_operation(
        campaign_id=str(uuid4()),
        batch_size=MAX_GENERATION_BATCH_SIZE,
    )

    # Unknown campaign, but the failure is the lookup — not batch_size.
    assert result["success"] is False
    assert result["error"]["code"] == "E002"


def test_intake_batch_size_rejects_values_above_the_shared_cap() -> None:
    payload = {
        "name": "over-cap",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "maximize"}],
        "batch_size": MAX_GENERATION_BATCH_SIZE + 1,
    }
    with pytest.raises(ValidationError) as exc_info:
        CampaignIntakeInput.model_validate(payload)
    assert any(e["loc"] == ("batch_size",) for e in exc_info.value.errors())
