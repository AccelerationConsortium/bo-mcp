"""E2E integration test for categorical donor/acceptor BO lifecycle."""

from uuid import uuid4

import pytest

from bo_mcp_server.domain import (  # type: ignore[import-untyped]
    CampaignIntakeInput,
    InputParameter,
    Objective,
    ParameterType,
    ResultSubmissionInput,
)
from bo_mcp_server.tools.create_campaign import create_campaign  # type: ignore[import-untyped]
from bo_mcp_server.tools.generate_suggestions import (  # type: ignore[import-untyped]
    generate_suggestions,
)
from bo_mcp_server.tools.get_diagnostics import get_diagnostics  # type: ignore[import-untyped]
from bo_mcp_server.tools.submit_results import submit_results  # type: ignore[import-untyped]


def _to_result_inputs(results: list[dict]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


@pytest.mark.usefixtures("setup_database")
class TestFragmentLifecycleE2E:
    """End-to-end lifecycle test for donor/acceptor categorical campaign."""

    @pytest.mark.asyncio
    async def test_fragment_campaign_should_complete_six_cycles_without_force(self):
        """Desired behavior: normal run completes 6 cycles and returns a best pair."""

        owner_id = str(uuid4())
        n_cycles = 6
        batch_size = 3

        donor_categories = [
            "phenyl",
            "anisole",
            "aniline",
            "thiophene",
            "carbazole",
            "phenothiazine",
        ]
        acceptor_categories = [
            "phenyl",
            "benzonitrile",
            "pyridine",
            "pyrimidine",
            "benzothiadiazole",
            "triazine",
        ]
        pair_to_gap = {
            (donor, acceptor): round(5.9 - 0.17 * i - 0.22 * j + 0.03 * ((i + j) % 3), 6)
            for i, donor in enumerate(donor_categories)
            for j, acceptor in enumerate(acceptor_categories)
        }

        intake_data = CampaignIntakeInput(
            name="BO normal-run six-cycle completion test",
            description="Track when normal categorical run can complete all six cycles.",
            parameters=(
                InputParameter(
                    name="donor",
                    type=ParameterType.CATEGORICAL,
                    categories=tuple(donor_categories),
                ),
                InputParameter(
                    name="acceptor",
                    type=ParameterType.CATEGORICAL,
                    categories=tuple(acceptor_categories),
                ),
            ),
            objectives=(Objective(name="gap_eV", direction="minimize"),),
            batch_size=batch_size,
            max_iterations=n_cycles,
            initial_design_size=3,
            random_seed=1,
        )

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is True, f"Create failed: {create_result['errors']}"
        campaign_id = create_result["campaign_id"]

        observed: list[tuple[str, str, float]] = []
        for cycle in range(1, n_cycles + 1):
            gen = await generate_suggestions(campaign_id)
            assert gen["success"] is True, f"Generate failed at cycle {cycle}: {gen['errors']}"
            assert len(gen["suggestions"]) == batch_size

            results = []
            for suggestion in gen["suggestions"]:
                params = suggestion["parameter_values"]
                donor = params["donor"]
                acceptor = params["acceptor"]
                gap_e_v = pair_to_gap[(donor, acceptor)]
                observed.append((donor, acceptor, gap_e_v))
                results.append(
                    {
                        "suggestion_id": suggestion["id"],
                        "parameter_values": params,
                        "objective_values": {"gap_eV": gap_e_v},
                    }
                )

            submit = await submit_results(
                campaign_id=campaign_id,
                results=_to_result_inputs(results),
                submitted_by=owner_id,
                source="api",
            )
            assert submit["success"] is True, (
                f"Submit failed at cycle {cycle}: {submit['errors']} "
                "(this is the tracked bug condition)"
            )

        diagnostics = await get_diagnostics(campaign_id)
        assert diagnostics["success"] is True
        assert diagnostics["n_results"] == n_cycles * batch_size
        assert diagnostics["best_value"] is not None
        assert diagnostics["best_parameters"] is not None

        best_observed = min(observed, key=lambda x: x[2])
        assert abs(diagnostics["best_value"] - best_observed[2]) < 1e-9
        assert diagnostics["best_parameters"]["donor"] == best_observed[0]
        assert diagnostics["best_parameters"]["acceptor"] == best_observed[1]
