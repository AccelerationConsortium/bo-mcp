"""TODO 1.62-1.67 review round 2 — explicit-backend creation enforces capabilities.

Verifies that ``create_campaign_operation`` rejects a spec whose explicit
backend reports ``validate_capabilities(...).is_compatible == False``,
not just the ``auto`` selection path.

Pre-fix, only ``backend.validate_spec`` ran — that produces warnings
only — so a campaign with invalid BayBE ``parameter_options`` slipped
through and crashed during the first suggestion call.
"""

from __future__ import annotations

import pytest

from bo_mcp_server.domain import CampaignIntakeInput
from bo_mcp_server.operations.create_campaign import create_campaign_operation

OWNER_ID = "00000000-0000-0000-0000-000000000000"


def _invalid_baybe_task_intake() -> dict:
    """Intake with an invalid TaskParameter ``active_values`` (``Z`` not in cats)."""
    return {
        "name": "Invalid task active values",
        "parameters": [
            {
                "name": "lab",
                "type": "categorical",
                "categories": ["A", "B"],
                "parameter_options": {
                    "baybe": {"role": "task", "active_values": ["A", "Z"]},
                },
            }
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "backend": "baybe",
    }


@pytest.mark.asyncio
async def test_explicit_baybe_rejected_when_capabilities_incompatible(
    setup_database: None,
) -> None:
    """Explicit ``backend="baybe"`` with invalid typed options fails at create."""
    _ = setup_database
    intake = CampaignIntakeInput.model_validate(_invalid_baybe_task_intake())
    response = await create_campaign_operation(intake_data=intake, owner_id=OWNER_ID)
    assert response["success"] is False
    assert response.get("campaign_id") is None
    # The unsupported reason mentions the offending key.
    joined = " ".join(response.get("errors") or [])
    assert "active_values" in joined or "['Z']" in joined
    details = response.get("error", {}).get("details", {})
    unsupported = details.get("unsupported", [])
    assert any("active_values" in r.get("key", "") for r in unsupported)


def _baybe_ignored_knob_intake() -> dict:
    """Intake with a BayBE-ignored BoTorch knob (``use_input_warping``).

    Pre-fix, ``required_features`` raised this to an UNSUPPORTED feature
    report and the new create-time enforcement rejected the campaign.
    Round-3 review demanded that BayBE-ignored knobs surface only as
    IGNORED warnings, not as UNSUPPORTED gates.
    """
    return {
        "name": "BayBE ignored knob",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "backend": "baybe",
        "use_input_warping": True,
    }


@pytest.mark.asyncio
async def test_explicit_baybe_accepts_ignored_knobs_with_warnings(
    setup_database: None,
) -> None:
    """A BayBE-ignored BoTorch knob is accepted with a warning, not rejected."""
    _ = setup_database
    intake = CampaignIntakeInput.model_validate(_baybe_ignored_knob_intake())
    response = await create_campaign_operation(intake_data=intake, owner_id=OWNER_ID)
    assert response["success"] is True
    assert response.get("campaign_id") is not None
    warnings = response.get("warnings") or []
    assert any("Input warping" in w or "input_warping" in w for w in warnings), warnings
