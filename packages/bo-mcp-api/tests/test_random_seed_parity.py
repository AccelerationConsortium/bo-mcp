"""REST/MCP ``random_seed`` parity (TODO 1.66, step 1).

REST ``IntakeData`` previously defaulted ``random_seed`` to ``None``,
while MCP ``CampaignIntakeInput`` defaulted it to ``42``. The same
omitted field therefore produced deterministic MCP campaigns but
unseeded REST campaigns. Both transports now default to ``None`` so
omitting the field has the same effect everywhere.
"""

from __future__ import annotations

from bo_mcp_server.domain import CampaignIntakeInput

from api.schemas.intake import IntakeData


def _intake_dict() -> dict:
    return {
        "name": "Parity",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }


def test_rest_intake_defaults_random_seed_to_none() -> None:
    intake = IntakeData.model_validate(_intake_dict())
    assert intake.random_seed is None


def test_mcp_intake_defaults_random_seed_to_none() -> None:
    intake = CampaignIntakeInput.model_validate(_intake_dict())
    assert intake.random_seed is None


def test_rest_and_mcp_omitted_seed_agree() -> None:
    """The same omitted ``random_seed`` produces the same default on both transports."""
    rest = IntakeData.model_validate(_intake_dict())
    mcp = CampaignIntakeInput.model_validate(_intake_dict())
    assert rest.random_seed == mcp.random_seed
