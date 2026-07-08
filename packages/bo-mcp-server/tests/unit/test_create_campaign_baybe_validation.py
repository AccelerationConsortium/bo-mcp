"""Explicit-backend creation enforces capability validation, not just warnings.

Verifies that ``create_campaign_operation`` rejects a spec whose explicit
backend reports ``validate_capabilities(...).is_compatible == False``,
not just the ``auto`` selection path.

Pre-fix, only ``backend.validate_spec`` ran — that produces warnings
only — so a campaign with invalid BayBE ``parameter_options`` slipped
through and crashed during the first suggestion call.

Companion coverage: a BayBE-degradable BoTorch knob (e.g.
``use_input_warping``) is rejected at create-time by default and only
accepted when the caller lists the field in ``acknowledge_degradations``.
"""

from __future__ import annotations

import pytest

from bo_mcp_server.domain import CampaignIntakeInput
from bo_mcp_server.operations.create_campaign import create_campaign_operation
from tests.factories import seed_owner


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
    response = await create_campaign_operation(intake_data=intake, owner_id=await seed_owner())
    assert response["success"] is False
    assert response.get("campaign_id") is None
    # The unsupported reason mentions the offending key.
    joined = " ".join(response.get("errors") or [])
    assert "active_values" in joined or "['Z']" in joined
    details = response.get("error", {}).get("details", {})
    unsupported = details.get("unsupported", [])
    assert any("active_values" in r.get("key", "") for r in unsupported)


def _valid_substance_intake() -> dict:
    """Intake for a valid solvent-screening substance campaign."""
    return {
        "name": "Solvent screening",
        "parameters": [
            {
                "name": "solvent",
                "type": "categorical",
                "categories": ["water", "ethanol", "methanol"],
                "parameter_options": {
                    "baybe": {
                        "role": "substance",
                        "substance_data": {"water": "O", "ethanol": "CCO", "methanol": "CO"},
                    }
                },
            }
        ],
        "objectives": [{"name": "yield", "direction": "maximize"}],
        "backend": "baybe",
    }


@pytest.mark.asyncio
async def test_explicit_baybe_accepts_valid_substance_spec(
    setup_database: None,
) -> None:
    """The inverse of the chem-missing rejection: a valid substance spec is created.

    Now that ``baybe[chem]`` is a default dependency, ``create_campaign_operation``
    must accept a ``role=substance`` campaign with valid SMILES and persist it
    (capability validation passes, no descriptor build happens until the first
    suggestion). Mirrors a BayBE solvent-screening ``SubstanceParameter`` spec.
    """
    _ = setup_database
    intake = CampaignIntakeInput.model_validate(_valid_substance_intake())
    response = await create_campaign_operation(intake_data=intake, owner_id=await seed_owner())
    assert response["success"] is True, response.get("errors")
    assert response.get("campaign_id") is not None


def _baybe_degradable_knob_intake(*, acknowledge: bool = False) -> dict:
    """Intake with a BayBE-degradable BoTorch knob (``use_input_warping``)."""
    intake: dict = {
        "name": "BayBE degradable knob",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "backend": "baybe",
        "use_input_warping": True,
    }
    if acknowledge:
        intake["acknowledge_degradations"] = ["use_input_warping"]
    return intake


@pytest.mark.asyncio
async def test_explicit_baybe_rejects_unacknowledged_degradable_knob(
    setup_database: None,
) -> None:
    """A BayBE-degradable knob is rejected at create-time without acknowledgement.

    Silently dropping a semantically load-bearing option produces wrong
    answers that look fine — the worst class of BO bug. Reject by default
    and force the caller to opt in.
    """
    _ = setup_database
    intake = CampaignIntakeInput.model_validate(_baybe_degradable_knob_intake())
    response = await create_campaign_operation(intake_data=intake, owner_id=await seed_owner())
    assert response["success"] is False
    assert response.get("campaign_id") is None
    joined = " ".join(response.get("errors") or [])
    assert "use_input_warping" in joined or "Input warping" in joined
    field_errors = response.get("field_errors") or {}
    assert "use_input_warping" in field_errors


@pytest.mark.asyncio
async def test_explicit_baybe_accepts_acknowledged_degradable_knob(
    setup_database: None,
) -> None:
    """Listing the knob in ``acknowledge_degradations`` opts into the degraded run."""
    _ = setup_database
    intake = CampaignIntakeInput.model_validate(
        _baybe_degradable_knob_intake(acknowledge=True),
    )
    response = await create_campaign_operation(intake_data=intake, owner_id=await seed_owner())
    assert response["success"] is True
    assert response.get("campaign_id") is not None
    warnings = response.get("warnings") or []
    assert any("Input warping" in w or "input_warping" in w for w in warnings), warnings
