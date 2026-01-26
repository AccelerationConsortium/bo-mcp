#!/usr/bin/env python3
"""Multi-Objective Bayesian Optimization via MCP Protocol.

This script demonstrates how to optimize multiple objectives simultaneously.
The MCP server automatically selects ModelListGP and qLogNEHVI for
multi-objective problems.

Configuration:
- 2 parameters (x1, x2)
- 2 conflicting objectives (f1, f2) to minimize
- The system finds the Pareto-optimal trade-off front

The system automatically:
- Creates ModelListGP (one GP per objective)
- Uses qLogNEHVI acquisition function
- Tracks hypervolume improvement
- Reports Pareto front size

Usage:
    uv run python scripts/demos/mcp_protocol_multi_objective.py
"""

import asyncio
import math
import random

from mcp_client_utils import (
    call_tool,
    connect_to_mcp_server,
    create_demo_campaign,
    get_or_create_demo_user,
    print_config_summary,
    print_diagnostics,
    print_header,
    print_iteration_header,
    print_method_selection,
    print_suggestion,
    submit_demo_results,
)


def branin_currin(x1: float, x2: float) -> tuple[float, float]:
    """Bi-objective Branin-Currin test function.

    Two conflicting objectives that produce a well-defined Pareto front.
    """
    # Branin function (scaled inputs)
    x1_b = x1 * 15 - 5
    x2_b = x2 * 15
    a, b, c, r, s, t = 1, 5.1 / (4 * math.pi**2), 5 / math.pi, 6, 10, 1 / (8 * math.pi)
    f1 = a * (x2_b - b * x1_b**2 + c * x1_b - r) ** 2
    f1 += s * (1 - t) * math.cos(x1_b) + s

    # Currin function
    factor1 = 1 - math.exp(-1 / (2 * max(x2, 1e-10)))
    numer = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
    denom = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20
    f2 = factor1 * numer / denom

    # Add noise
    f1 += random.gauss(0, 0.1)
    f2 += random.gauss(0, 0.1)

    return f1, f2


async def main() -> None:
    """Run multi-objective BO example via MCP protocol."""
    print_header("MULTI-OBJECTIVE BO via MCP Protocol")
    print()
    print("This demo optimizes two conflicting objectives simultaneously.")
    print("The system automatically selects appropriate multi-objective methods.")

    # Configuration for multi-objective optimization
    config = {
        "name": "Multi-Objective Demo",
        "parameters": [
            {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
            {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [
            {"name": "f1", "direction": "minimize"},
            {"name": "f2", "direction": "minimize"},
        ],
        "batch_size": 3,
    }

    print_config_summary(config)

    # Setup demo user (required for campaign ownership)
    print("\nSetting up demo user...")
    owner_id = await get_or_create_demo_user()
    print(f"Using user: {owner_id}")

    print("\nConnecting to MCP server...")

    async with connect_to_mcp_server() as session:
        print("Server connected successfully.")

        # Create campaign
        print("\nCreating campaign...")
        result = await create_demo_campaign(session, config, owner_id)

        if not result.get("success"):
            print(f"ERROR: {result.get('errors')}")
            return

        campaign_id = result["campaign_id"]
        print(f"Campaign created: {campaign_id}")

        # Run optimization loop
        n_iterations = 4
        all_results: list[tuple[dict, dict]] = []
        diagnostics = None

        for iteration in range(1, n_iterations + 1):
            # Generate suggestions
            suggestions_result = await call_tool(
                session, "generate_suggestions", {"campaign_id": campaign_id}
            )

            if not suggestions_result.get("success"):
                print(f"ERROR: {suggestions_result.get('errors')}")
                break

            # Display automatic method selection
            if "method_selection" in suggestions_result:
                print_method_selection(suggestions_result["method_selection"])

            is_initial = iteration == 1
            phase = "Initial Design" if is_initial else "BO-guided (qLogNEHVI)"
            print_iteration_header(iteration, phase)

            suggestions = suggestions_result["suggestions"]
            print(f"  Generated {len(suggestions)} suggestions")

            # Evaluate suggestions
            results_to_submit = []
            for i, s in enumerate(suggestions):
                params = s["parameter_values"]
                x1 = params.get("x1", 0.5)
                x2 = params.get("x2", 0.5)

                f1, f2 = branin_currin(x1, x2)

                results_to_submit.append(
                    {
                        "parameter_values": params,
                        "objective_values": {"f1": f1, "f2": f2},
                        "suggestion_id": s["id"],
                    }
                )

                all_results.append((params, {"f1": f1, "f2": f2}))
                print_suggestion(params, {"f1": f1, "f2": f2}, index=i + 1)

            # Submit results
            submit_result = await submit_demo_results(
                session, campaign_id, results_to_submit, owner_id
            )

            if not submit_result.get("success"):
                print(f"ERROR submitting: {submit_result.get('errors')}")
                break

            # Get diagnostics (shows hypervolume and Pareto front)
            diagnostics = await call_tool(session, "get_diagnostics", {"campaign_id": campaign_id})

            if diagnostics.get("success"):
                print_diagnostics(diagnostics, iteration)

        # Final results
        print_header("MULTI-OBJECTIVE OPTIMIZATION COMPLETE")
        print()
        print(f"  Total iterations: {n_iterations}")
        print(f"  Total evaluations: {len(all_results)}")

        if diagnostics and diagnostics.get("success"):
            hv = diagnostics.get("hypervolume")
            n_pareto = diagnostics.get("n_pareto_points")
            if hv is not None:
                print(f"  Final hypervolume: {hv:.4f}")
            if n_pareto is not None:
                print(f"  Pareto front size: {n_pareto}")

        print()
        print("Key Takeaway:")
        print("  With 2 objectives, the MCP server automatically:")
        print("  - Selected ModelListGP (one GP per objective)")
        print("  - Used qLogNEHVI acquisition (hypervolume-based)")
        print("  - Tracked Pareto front and hypervolume")
        print()
        print("Alternative: qLogNParEGO")
        print("  - Uses Chebyshev scalarization")
        print("  - Better for 3+ objectives")
        print("  - Set 'acquisition_method': 'qLogNParEGO' to use")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
