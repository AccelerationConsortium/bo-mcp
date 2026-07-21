"""End-to-end tests for the ``field_errors`` envelope.

Agents that drive this server need a way to surgically correct failing
inputs without re-parsing the human-facing ``errors`` list. The
``field_errors`` map keys each finding by a dotted path so a 20-row
batch failure points at exactly the bad row + field.

Reference: this is the standard machine-readable shape adopted by
modern validation libraries (Pydantic ``ValidationError.errors()``,
JSON Schema ``instancePath``) — see
https://docs.pydantic.dev/latest/errors/usage_errors/ for the source
shape we lift the paths from.
"""

from typing import Any

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
from tests.factories import seed_owner


def _to_result_inputs(rows: list[dict[str, Any]]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in rows]


async def _build_campaign(batch_size: int = 3) -> tuple[str, list[dict[str, Any]], str]:
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Field Errors E2E",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    gen = await generate_suggestions(created["campaign_id"], batch_size=batch_size)
    return created["campaign_id"], gen["suggestions"], owner_id


@pytest.mark.usefixtures("setup_database")
class TestSubmitResultsFieldErrors:
    """``submit_results`` populates ``field_errors`` for row failures."""

    @pytest.mark.asyncio
    async def test_missing_objective_pins_row_and_field(self) -> None:
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign(batch_size=3)
        rows = [
            {
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
                "suggestion_id": suggestions[0]["suggestion_id"],
            },
            {
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
                "suggestion_id": suggestions[1]["suggestion_id"],
            },
            {
                "parameter_values": suggestions[2]["parameter_values"],
                "objective_values": {"wrong_name": 0.3},
                "suggestion_id": suggestions[2]["suggestion_id"],
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
            verbosity="detailed",
        )
        assert result["success"] is False
        # The detailed projection forwards the field_errors map verbatim.
        field_errors = result["field_errors"]
        assert "results[2].objective_values" in field_errors
        # Earlier (valid) rows must not appear in the map.
        assert "results[0].objective_values" not in field_errors
        assert "results[1].objective_values" not in field_errors

    @pytest.mark.asyncio
    async def test_negative_uncertainty_pins_objective_key(self) -> None:
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign(batch_size=2)
        rows = [
            {
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
                "measurement_uncertainty": {"y": 0.05},
                "suggestion_id": suggestions[0]["suggestion_id"],
            },
            {
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
                "measurement_uncertainty": {"y": -0.1},
                "suggestion_id": suggestions[1]["suggestion_id"],
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
            verbosity="detailed",
        )
        assert result["success"] is False
        field_errors = result["field_errors"]
        # The bad uncertainty key is addressable by its objective name.
        assert "results[1].measurement_uncertainty['y']" in field_errors
        assert any(
            "negative" in m.lower() for m in field_errors["results[1].measurement_uncertainty['y']"]
        )

    @pytest.mark.asyncio
    async def test_minimal_verbosity_still_carries_field_errors(self) -> None:
        """``minimal`` verbosity exposes the field_errors map for token-budget callers.

        The point of field_errors is that an agent on a tight token
        budget can correct the failing field without retrying the whole
        batch -- so the map has to be reachable at the lowest verbosity
        too.
        """
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign(batch_size=1)
        rows = [
            {
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"wrong": 0.0},
                "suggestion_id": suggestions[0]["suggestion_id"],
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
            verbosity="minimal",
        )
        assert result["success"] is False
        assert "results[0].objective_values" in result["field_errors"]


@pytest.mark.usefixtures("setup_database")
class TestValidateIntakeFieldErrors:
    """``validate_intake`` mirrors Pydantic ``loc`` into ``field_errors``."""

    def test_missing_name_pins_field(self) -> None:
        from bo_mcp_server.operations.validate_intake import validate_intake_operation

        intake = {
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        result = validate_intake_operation(intake)
        assert result["valid"] is False
        # ``name`` is the offending root-level required field.
        assert "name" in result["field_errors"]

    def test_nested_parameter_error(self) -> None:
        from bo_mcp_server.operations.validate_intake import validate_intake_operation

        intake = {
            "name": "Nested Path Test",
            "parameters": [
                {"name": "x", "type": "not_a_real_type", "bounds": [0, 1]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        result = validate_intake_operation(intake)
        assert result["valid"] is False
        # The bad parameter index is preserved in the dotted path so the
        # caller can edit ``parameters[0].type`` directly.
        keys = list(result["field_errors"])
        assert any(k.startswith("parameters[0]") for k in keys), keys

    def test_valid_intake_returns_empty_field_errors(self) -> None:
        from bo_mcp_server.operations.validate_intake import validate_intake_operation

        intake = {
            "name": "Valid",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        result = validate_intake_operation(intake)
        assert result["valid"] is True
        assert result["field_errors"] == {}


@pytest.mark.usefixtures("setup_database")
class TestCreateCampaignFieldErrors:
    """``create_campaign`` propagates field_errors from intake validation."""

    @pytest.mark.asyncio
    async def test_intake_validation_failure_surfaces_field_errors(self) -> None:
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = await seed_owner()
        intake = {
            # Empty name fails the ``min_length=1`` check on
            # ``CampaignIntakeInput.name``.
            "name": "",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        result = await create_campaign(intake, owner_id, verbosity="detailed")
        assert result["success"] is False
        assert "name" in result["field_errors"]
