#!/usr/bin/env python3
"""Hard-coded categorical BO example for donor/acceptor fragment selection.

This mirrors the MCP toy workflow but uses a pre-defined objective sequence
instead of running expensive quantum chemistry.
"""

import asyncio
import hashlib

import dotenv

dotenv.load_dotenv()  # Load environment variables from .env if present

from bo_mcp_server.domain import (
    CampaignIntakeInput,
    InputParameter,
    Objective,
    ParameterType,
    ResultSubmissionInput,
    User,
)
from bo_mcp_server.storage import UserRepository, get_session, lifespan
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_diagnostics import get_diagnostics
from bo_mcp_server.tools.submit_results import submit_results

API_KEY = "dev-api-key-12345"
N_CYCLES = 6
EXPECTED_BATCH_SIZE = 2

# Replace these 12 values with your own local mock data.
MOCK_GAP_EV_SEQUENCE = [
    5.625384909529561,
    4.562828813866204,
    5.2100,
    4.9800,
    4.7700,
    4.6400,
    4.5200,
    4.4100,
    4.3600,
    4.2900,
    4.2400,
    4.1800,
]

CAMPAIGN_DATA = CampaignIntakeInput(
    name="BO: donor–acceptor fragment selection for minimal HOMO–LUMO gap",
    description=(
        "Categorical BO over donor (Set A) and acceptor (Set B) fragments; "
        "objective is to minimize HOMO–LUMO gap (eV) of the coupled molecule "
        "computed via fast DFT (B3LYP/def2-SVP)."
    ),
    parameters=[
        InputParameter(
            name="donor",
            type=ParameterType.CATEGORICAL,
            categories=[
                "phenyl",
                "anisole",
                "aniline",
                "thiophene",
                "carbazole",
                "phenothiazine",
            ],
        ),
        InputParameter(
            name="acceptor",
            type=ParameterType.CATEGORICAL,
            categories=[
                "phenyl",
                "benzonitrile",
                "pyridine",
                "pyrimidine",
                "benzothiadiazole",
                "triazine",
            ],
        ),
    ],
    objectives=[Objective(name="gap_eV", direction="minimize")],
    batch_size=EXPECTED_BATCH_SIZE,
    max_iterations=N_CYCLES,
    initial_design_size=2,
    random_seed=1,
)


def _mock_result_for(
    experiment_index: int, parameter_values: dict[str, str], suggestion_id: str
) -> ResultSubmissionInput:
    """Return one mocked observation payload from the fixed objective sequence."""
    if experiment_index >= len(MOCK_GAP_EV_SEQUENCE):
        raise IndexError(
            "Not enough mocked objective values. "
            f"Need at least {N_CYCLES * EXPECTED_BATCH_SIZE} entries."
        )

    gap_e_v = float(MOCK_GAP_EV_SEQUENCE[experiment_index])
    return ResultSubmissionInput(
        suggestion_id=suggestion_id,
        parameter_values=parameter_values,
        objective_values={"gap_eV": gap_e_v},
        metadata={
            "method": "MOCK: pre-defined gap_eV array (replace with local evaluator)",
            "gap_raw_hartree": gap_e_v / 27.211386245988,
            "coupled_smiles": f"mock::{parameter_values['donor']}::{parameter_values['acceptor']}",
            "sequence_index": experiment_index,
        },
    )


async def main() -> None:
    """Run 6 BO cycles (2 suggestions each) with hard-coded objective values."""
    print("=" * 72)
    print("BO-MCP Hard-Coded Example: Donor/Acceptor HOMO-LUMO Gap Minimization")
    print("=" * 72)

    required_results = N_CYCLES * EXPECTED_BATCH_SIZE
    if len(MOCK_GAP_EV_SEQUENCE) < required_results:
        raise ValueError(
            f"MOCK_GAP_EV_SEQUENCE has {len(MOCK_GAP_EV_SEQUENCE)} values, "
            f"but {required_results} are required for {N_CYCLES} cycles."
        )

    async with lifespan():
        api_key_hash = hashlib.sha256(API_KEY.encode()).hexdigest()
        async with get_session() as session:
            repo = UserRepository(session)
            user = await repo.get_by_email("test@example.com")
            if not user:
                user = User(
                    name="Test User",
                    email="test@example.com",
                    api_key_hash=api_key_hash,
                )
                user = await repo.save(user)
                print(f"Created test user: {user.id}")
            else:
                print(f"Using existing user: {user.id}")

        owner_id = str(user.id)

        print("\n" + "-" * 44)
        print("Step 1: Creating donor/acceptor BO campaign")
        print("-" * 44)
        create_result = await create_campaign(CAMPAIGN_DATA, owner_id)
        if not create_result["success"]:
            print(f"Failed to create campaign: {create_result['errors']}")
            return

        campaign_id = create_result["campaign_id"]
        print(f"Campaign created: {campaign_id}")
        print(f"Configured cycles: {N_CYCLES}, batch size: {EXPECTED_BATCH_SIZE}")

        experiment_index = 0
        best_gap = float("inf")
        best_combo: dict[str, str] | None = None

        for cycle in range(1, N_CYCLES + 1):
            print("\n" + "=" * 44)
            print(f"Cycle {cycle}/{N_CYCLES}")
            print("=" * 44)

            suggestions_result = await generate_suggestions(campaign_id)
            if not suggestions_result["success"]:
                print(f"Failed to generate suggestions: {suggestions_result['errors']}")
                break

            suggestions = suggestions_result["suggestions"]
            if len(suggestions) != EXPECTED_BATCH_SIZE:
                print(
                    "Unexpected suggestion count: "
                    f"expected {EXPECTED_BATCH_SIZE}, got {len(suggestions)}"
                )
                break

            print("Suggestions:")
            for i, suggestion in enumerate(suggestions, start=1):
                p = suggestion["parameter_values"]
                print(f"  {i}. donor={p['donor']}, acceptor={p['acceptor']}")

            results_to_submit = []
            for suggestion in suggestions:
                params = suggestion["parameter_values"]
                result_entry = _mock_result_for(experiment_index, params, suggestion["id"])
                experiment_index += 1
                gap_e_v = result_entry.objective_values["gap_eV"]

                if gap_e_v < best_gap:
                    best_gap = gap_e_v
                    best_combo = params

                print(
                    "  -> gap_eV="
                    f"{gap_e_v:.6f} (mock seq idx {result_entry.metadata['sequence_index']})"
                )
                results_to_submit.append(result_entry)

            submit_result = await submit_results(
                campaign_id=campaign_id,
                results=results_to_submit,
                submitted_by=owner_id,
                source="api",
            )
            if not submit_result["success"]:
                print(f"Failed to submit results: {submit_result['errors']}")
                break

            diagnostics = await get_diagnostics(campaign_id)
            if diagnostics["success"]:
                print(
                    f"Submitted {len(submit_result['result_ids'])} results "
                    f"(total={diagnostics['n_results']})"
                )
                print(f"Current best gap_eV: {best_gap:.6f}")
                if best_combo:
                    print(
                        "Current best combo: "
                        f"donor={best_combo['donor']}, acceptor={best_combo['acceptor']}"
                    )

        print("\n" + "=" * 72)
        print("Hard-coded optimization run complete")
        print(f"Total submitted experiments: {experiment_index}")
        if best_combo:
            print(
                "Best observed combo: "
                f"donor={best_combo['donor']}, acceptor={best_combo['acceptor']}, "
                f"gap_eV={best_gap:.6f}"
            )
        print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
