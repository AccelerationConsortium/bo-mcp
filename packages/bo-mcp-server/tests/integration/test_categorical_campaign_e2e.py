"""End-to-end lifecycle tests for categorical-only BO campaigns.

Validates the complete intake -> suggest -> submit -> suggest flow with
purely categorical parameters, ensuring no duplicate suggestions in batches.

References:
    - BoTorch optimize_acqf_discrete:
      https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf_discrete
    - PR10: Fix duplicate batch suggestions for categorical/discrete parameters
"""

from uuid import uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput


def _to_result_inputs(results: list[dict]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


class TestCategoricalCampaignLifecycle:
    """E2E tests for categorical-only campaigns through the MCP tool layer."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_categorical_only(self, setup_database) -> None:
        """End-to-end with 2 categorical params: all suggestions in each batch are unique.

        Creates a campaign with 2 categorical parameters (3 categories each),
        runs initial design + 2 BO cycles, and verifies no duplicate suggestions.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        # Synthetic objective values
        strength_map = {
            ("steel", "none"): 50.0,
            ("steel", "chrome"): 70.0,
            ("steel", "zinc"): 65.0,
            ("aluminum", "none"): 40.0,
            ("aluminum", "chrome"): 55.0,
            ("aluminum", "zinc"): 50.0,
            ("titanium", "none"): 80.0,
            ("titanium", "chrome"): 95.0,
            ("titanium", "zinc"): 85.0,
        }

        intake_data = {
            "name": "Categorical E2E Test",
            "description": "Testing categorical-only BO lifecycle",
            "parameters": [
                {
                    "name": "material",
                    "type": "categorical",
                    "categories": ["steel", "aluminum", "titanium"],
                    "description": "Material type",
                },
                {
                    "name": "coating",
                    "type": "categorical",
                    "categories": ["none", "chrome", "zinc"],
                    "description": "Coating type",
                },
            ],
            "objectives": [
                {"name": "strength", "direction": "maximize", "unit": "MPa"},
            ],
            "batch_size": 2,
            "initial_design_size": 2,
            "random_seed": 42,
        }

        from bo_mcp_server.domain import CampaignIntakeInput

        create_result = await create_campaign(
            CampaignIntakeInput.model_validate(intake_data), owner_id
        )
        assert create_result["success"], f"Create failed: {create_result.get('errors')}"
        campaign_id = create_result["campaign_id"]

        # Run 3 cycles: initial design + 2 BO cycles
        for cycle in range(3):
            suggestions_result = await generate_suggestions(campaign_id)
            assert suggestions_result["success"], (
                f"Suggest failed cycle {cycle}: {suggestions_result.get('errors')}"
            )

            suggestions = suggestions_result["suggestions"]
            assert len(suggestions) == 2

            # Verify uniqueness within batch
            params_list = [s["parameter_values"] for s in suggestions]
            assert params_list[0] != params_list[1], (
                f"Duplicate suggestions in cycle {cycle}: {params_list[0]} == {params_list[1]}"
            )

            # Submit results
            results_to_submit = []
            for s in suggestions:
                pv = s["parameter_values"]
                key = (pv["material"], pv["coating"])
                results_to_submit.append(
                    {
                        "suggestion_id": s["id"],
                        "parameter_values": pv,
                        "objective_values": {"strength": strength_map[key]},
                    }
                )

            submit_result = await submit_results(
                campaign_id=campaign_id,
                results=_to_result_inputs(results_to_submit),
                submitted_by=owner_id,
                source="api",
            )
            assert submit_result["success"], (
                f"Submit failed cycle {cycle}: {submit_result.get('errors')}"
            )

    @pytest.mark.asyncio
    async def test_categorical_suggestions_are_valid(self, setup_database) -> None:
        """All returned category values must exist in the parameter spec."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Category Validation Test",
            "description": "Ensure returned categories are valid",
            "parameters": [
                {
                    "name": "metal",
                    "type": "categorical",
                    "categories": ["Cu", "Ag", "Au", "Pt"],
                },
                {
                    "name": "solvent",
                    "type": "categorical",
                    "categories": ["water", "ethanol", "DMSO"],
                },
            ],
            "objectives": [
                {"name": "yield", "direction": "maximize"},
            ],
            "batch_size": 2,
            "initial_design_size": 2,
            "random_seed": 42,
        }

        valid_metals = {"Cu", "Ag", "Au", "Pt"}
        valid_solvents = {"water", "ethanol", "DMSO"}

        from bo_mcp_server.domain import CampaignIntakeInput

        create_result = await create_campaign(
            CampaignIntakeInput.model_validate(intake_data), owner_id
        )
        assert create_result["success"]
        campaign_id = create_result["campaign_id"]

        # Initial design
        suggestions_result = await generate_suggestions(campaign_id)
        assert suggestions_result["success"]
        for s in suggestions_result["suggestions"]:
            pv = s["parameter_values"]
            assert pv["metal"] in valid_metals, f"Invalid metal: {pv['metal']}"
            assert pv["solvent"] in valid_solvents, f"Invalid solvent: {pv['solvent']}"

        # Submit results and get BO suggestions
        results_to_submit = []
        for s in suggestions_result["suggestions"]:
            pv = s["parameter_values"]
            results_to_submit.append(
                {
                    "suggestion_id": s["id"],
                    "parameter_values": pv,
                    "objective_values": {"yield": 50.0},
                }
            )
        await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(results_to_submit),
            submitted_by=owner_id,
            source="api",
        )

        # BO suggestions
        bo_result = await generate_suggestions(campaign_id)
        assert bo_result["success"]
        for s in bo_result["suggestions"]:
            pv = s["parameter_values"]
            assert pv["metal"] in valid_metals, f"Invalid BO metal: {pv['metal']}"
            assert pv["solvent"] in valid_solvents, f"Invalid BO solvent: {pv['solvent']}"

    @pytest.mark.asyncio
    async def test_categorical_campaign_with_large_space(self, setup_database) -> None:
        """4 params x 5 categories each = 625 combos. Verifies optimize_acqf_discrete handles it."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Large Categorical Space Test",
            "description": "4 categorical params, 5 categories each = 625 combos",
            "parameters": [
                {
                    "name": f"param_{i}",
                    "type": "categorical",
                    "categories": [f"cat_{j}" for j in range(5)],
                }
                for i in range(4)
            ],
            "objectives": [
                {"name": "score", "direction": "maximize"},
            ],
            "batch_size": 3,
            "initial_design_size": 4,
            "random_seed": 42,
        }

        from bo_mcp_server.domain import CampaignIntakeInput

        create_result = await create_campaign(
            CampaignIntakeInput.model_validate(intake_data), owner_id
        )
        assert create_result["success"]
        campaign_id = create_result["campaign_id"]

        # Initial design
        suggestions_result = await generate_suggestions(campaign_id)
        assert suggestions_result["success"]

        # Submit initial results
        results_to_submit = []
        for idx, s in enumerate(suggestions_result["suggestions"]):
            pv = s["parameter_values"]
            results_to_submit.append(
                {
                    "suggestion_id": s["id"],
                    "parameter_values": pv,
                    "objective_values": {"score": float(idx + 1)},
                }
            )
        await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(results_to_submit),
            submitted_by=owner_id,
            source="api",
        )

        # BO suggestions from the large space
        bo_result = await generate_suggestions(campaign_id)
        assert bo_result["success"]
        suggestions = bo_result["suggestions"]
        assert len(suggestions) == 3

        # All 3 suggestions should be unique
        params_list = [tuple(sorted(s["parameter_values"].items())) for s in suggestions]
        assert len(set(params_list)) == len(params_list), (
            f"Duplicate suggestions in large categorical space: {params_list}"
        )
