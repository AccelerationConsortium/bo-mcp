# ruff: noqa: I001,E402
# pylint: disable=wrong-import-order, wrong-import-position
#!/usr/bin/env python3
"""Hard-coded categorical BO example for donor/acceptor fragment selection.

This mirrors the MCP toy workflow but uses pre-computed gap values
for each donor/acceptor pair instead of running expensive quantum chemistry.
"""

import dotenv
import asyncio
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dotenv.load_dotenv()  # Load environment variables from .env if present

from demos.mcp_client_utils import get_or_create_demo_user

from bo_mcp_server.domain import (  # type: ignore[import-untyped]
    CampaignIntakeInput,
    InputParameter,
    Objective,
    ParameterType,
    ResultSubmissionInput,
)
from bo_mcp_server.storage import lifespan  # type: ignore[import-untyped]
from bo_mcp_server.tools.create_campaign import create_campaign  # type: ignore[import-untyped]
from bo_mcp_server.tools.generate_suggestions import (  # type: ignore[import-untyped]
    generate_suggestions,
)
from bo_mcp_server.tools.get_diagnostics import get_diagnostics  # type: ignore[import-untyped]
from bo_mcp_server.tools.submit_results import submit_results  # type: ignore[import-untyped]

N_CYCLES = 10
EXPECTED_BATCH_SIZE = 2
N_RUNS = 100


@dataclass(frozen=True)
class Pair:
    """
    Represents a pair of chemical groups for which we have a known gap energy value.
    """

    donor: str
    acceptor: str


GAP_EV: dict[Pair, float] = {
    Pair("phenyl", "phenyl"): 4.811,
    Pair("phenyl", "benzonitrile"): 4.099,
    Pair("phenyl", "pyridine"): 3.557,
    Pair("phenyl", "pyrimidine"): 3.229,
    Pair("phenyl", "benzothiadiazole"): 2.936,
    Pair("phenyl", "triazine"): 3.230,
    Pair("anisole", "phenyl"): 3.992,
    Pair("anisole", "benzonitrile"): 2.996,
    Pair("anisole", "pyridine"): 3.198,
    Pair("anisole", "pyrimidine"): 3.120,
    Pair("anisole", "benzothiadiazole"): 2.560,
    Pair("anisole", "triazine"): 3.020,
    Pair("aniline", "phenyl"): 3.514,
    Pair("aniline", "benzonitrile"): 2.685,
    Pair("aniline", "pyridine"): 2.971,
    Pair("aniline", "pyrimidine"): 2.851,
    Pair("aniline", "benzothiadiazole"): 2.274,
    Pair("aniline", "triazine"): 2.763,
    Pair("thiophene", "phenyl"): 4.223,
    Pair("thiophene", "benzonitrile"): 3.384,
    Pair("thiophene", "pyridine"): 3.140,
    Pair("thiophene", "pyrimidine"): 3.155,
    Pair("thiophene", "benzothiadiazole"): 2.500,
    Pair("thiophene", "triazine"): 3.141,
    Pair("carbazole", "phenyl"): 3.336,
    Pair("carbazole", "benzonitrile"): 2.713,
    Pair("carbazole", "pyridine"): 2.827,
    Pair("carbazole", "pyrimidine"): 2.651,
    Pair("carbazole", "benzothiadiazole"): 2.380,
    Pair("carbazole", "triazine"): 2.470,
    Pair("phenothiazine", "phenyl"): 2.500,
    Pair("phenothiazine", "benzonitrile"): 1.974,
    Pair("phenothiazine", "pyridine"): 2.171,
    Pair("phenothiazine", "pyrimidine"): 1.969,
    Pair("phenothiazine", "benzothiadiazole"): 1.641,
    Pair("phenothiazine", "triazine"): 1.757,
}
if len(GAP_EV) != 36:
    msg = f"GAP_EV must define all 36 donor×acceptor pairs, got {len(GAP_EV)}"
    raise RuntimeError(msg)
# Derive categories from GAP_EV keys to avoid duplicating source data.
DONOR_CATEGORIES = list(dict.fromkeys(pair.donor for pair in GAP_EV))
ACCEPTOR_CATEGORIES = list(dict.fromkeys(pair.acceptor for pair in GAP_EV))


CAMPAIGN_DATA = CampaignIntakeInput(
    name="BO: donor–acceptor fragment selection for minimal HOMO–LUMO gap",
    description=(
        "Categorical BO over donor (Set A) and acceptor (Set B) fragments; "
        "objective is to minimize HOMO–LUMO gap (eV) of the coupled molecule "
        "computed via fast DFT (B3LYP/def2-SVP)."
    ),
    parameters=(
        InputParameter(
            name="donor",
            type=ParameterType.CATEGORICAL,
            categories=tuple(DONOR_CATEGORIES),
        ),
        InputParameter(
            name="acceptor",
            type=ParameterType.CATEGORICAL,
            categories=tuple(ACCEPTOR_CATEGORIES),
        ),
    ),
    objectives=(Objective(name="gap_eV", direction="minimize"),),
    batch_size=EXPECTED_BATCH_SIZE,
    max_iterations=N_CYCLES,
    initial_design_size=2,
)


def _result_for(
    experiment_index: int, parameter_values: dict[str, str], suggestion_id: str
) -> ResultSubmissionInput:
    """Return one observation payload from GAP_EV donor/acceptor lookup."""
    pair = Pair(parameter_values["donor"], parameter_values["acceptor"])
    try:
        gap_e_v = float(GAP_EV[pair])
    except KeyError as exc:
        msg = f"Missing gap_eV for pair donor={pair.donor}, acceptor={pair.acceptor}"
        raise KeyError(msg) from exc

    return ResultSubmissionInput(
        suggestion_id=suggestion_id,
        parameter_values=parameter_values,
        objective_values={"gap_eV": gap_e_v},
        metadata={
            "method": "pre-computed GAP_EV lookup",
            "gap_raw_hartree": gap_e_v / 27.211386245988,
            "coupled_smiles": f"lookup::{parameter_values['donor']}::{parameter_values['acceptor']}",  # noqa: E501
            "sequence_index": experiment_index,
        },
    )


async def main() -> None:
    """Run multiple independent BO campaigns and plot averaged convergence."""
    print("=" * 72)
    print("BO-MCP Hard-Coded Example: Donor/Acceptor HOMO-LUMO Gap Minimization")
    print("=" * 72)
    print(f"Runs: {N_RUNS}, cycles per run: {N_CYCLES}, batch size: {EXPECTED_BATCH_SIZE}")

    expected_pairs = len(DONOR_CATEGORIES) * len(ACCEPTOR_CATEGORIES)
    if len(GAP_EV) != expected_pairs:
        msg = (
            f"GAP_EV has {len(GAP_EV)} entries, "
            f"but expected at least {expected_pairs} for full donor/acceptor coverage."
        )
        raise ValueError(msg)

    async with lifespan():
        owner_id = await get_or_create_demo_user(email="test@example.com", name="Test User")
        print(f"Using demo user: {owner_id}")

        all_runs_records: list[dict[str, float | int]] = []
        run_best_gaps: list[float] = []
        run_best_combos: list[dict[str, str] | None] = []

        for run_idx in range(1, N_RUNS + 1):
            print("\n" + "-" * 44)
            print(f"Run {run_idx}/{N_RUNS}: Creating donor/acceptor BO campaign")
            print("-" * 44)
            create_result = await create_campaign(CAMPAIGN_DATA, owner_id)
            if not create_result["success"]:
                print(f"Failed to create campaign: {create_result['errors']}")
                continue

            campaign_id = create_result["campaign_id"]
            print(f"Campaign created: {campaign_id}")
            print(f"Configured cycles: {N_CYCLES}, batch size: {EXPECTED_BATCH_SIZE}")

            experiment_index = 0
            best_gap = float("inf")
            best_combo: dict[str, str] | None = None

            for cycle in range(1, N_CYCLES + 1):
                print("\n" + "=" * 44)
                print(f"Run {run_idx}/{N_RUNS} | Cycle {cycle}/{N_CYCLES}")
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
                    result_entry = _result_for(experiment_index, params, suggestion["id"])
                    experiment_index += 1
                    gap_e_v = result_entry.objective_values["gap_eV"]

                    if gap_e_v < best_gap:
                        best_gap = gap_e_v
                        best_combo = params

                    print(
                        f"  -> gap_eV={gap_e_v:.6f} "
                        + f"(seq idx {result_entry.metadata['sequence_index']})"
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

                all_runs_records.append(
                    {
                        "run": run_idx,
                        "cycle": cycle,
                        "cumulative_best_gap_eV": best_gap,
                    }
                )

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

            run_best_gaps.append(best_gap)
            run_best_combos.append(best_combo)
            if best_combo:
                print(
                    f"Best observed combo in run {run_idx}: "
                    f"donor={best_combo['donor']}, acceptor={best_combo['acceptor']}, "
                    f"gap_eV={best_gap:.6f}"
                )
        print("\n" + "=" * 72)
        print("Hard-coded optimization runs complete")
        if run_best_gaps:
            global_best_idx = min(range(len(run_best_gaps)), key=lambda i: run_best_gaps[i])
            global_best_gap = run_best_gaps[global_best_idx]
            global_best_combo = run_best_combos[global_best_idx]
            if global_best_combo:
                print(
                    f"Best observed combo overall: "
                    f"donor={global_best_combo['donor']}, "
                    + f"acceptor={global_best_combo['acceptor']}, "
                    f"gap_eV={global_best_gap:.6f} (run {global_best_idx + 1})"
                )
            else:
                print(
                    f"Best observed gap overall: gap_eV={global_best_gap:.6f} "
                    f"(run {global_best_idx + 1})"
                )
        if not all_runs_records:
            print("No completed runs produced data for plotting.")
        else:
            curve_df = pd.DataFrame(all_runs_records)
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(8, 5))
            sns.lineplot(
                data=curve_df,
                x="cycle",
                y="cumulative_best_gap_eV",
                errorbar=("ci", 95),
                linewidth=2.2,
                color="#2E86AB",
                marker="o",
                label="Cumulative best (mean ± 95% CI across runs)",
            )
            plt.xlabel("batch index")
            plt.ylabel("HOMO-LUMO gap / eV")
            plt.xticks(list(range(1, N_CYCLES + 1)))
            plt.legend()
            plt.tight_layout()

            output_path = (
                Path(__file__).parent
                / "hardcoded_fragment_bo_example_cumulative_best_avg_100_runs.png"
            )
            plt.savefig(output_path, dpi=500)
            plt.close()
            print(f"Saved averaged cumulative-best plot to: {output_path}")

        print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
