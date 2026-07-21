#!/usr/bin/env python3
"""High-Dimensional Bayesian Optimization (TuRBO) via MCP Protocol.

This script demonstrates optimization with many parameters. The MCP server
automatically detects high-dimensional problems and applies TuRBO
(Trust Region BO) for improved efficiency.

Configuration:
- 25 continuous parameters
- 1 objective to minimize
- System auto-selects TuRBO strategy

The system automatically:
- Detects high-dimensional problem (>20 params)
- Activates TuRBO strategy with local trust region
- Uses SingleTaskGP model
- Adapts trust region based on success/failure

Usage:
    uv run python scripts/demos/mcp_protocol_high_dim.py
"""

import asyncio
import math
import random

from mcp_client_utils import (
    call_tool,
    connect_to_mcp_server,
    create_demo_campaign,
    get_or_create_demo_user,
    print_diagnostics,
    print_header,
    print_iteration_header,
    print_method_selection,
    submit_demo_results,
)


def levy_high_dim(x: list[float]) -> float:
    """High-dimensional Levy function.

    A challenging multi-modal function commonly used for testing
    high-dimensional optimization. Global minimum at all x_i = 1.

    Args:
        x: List of parameter values in [0, 1], scaled internally to [-10, 10].
    """
    # Scale from [0, 1] to [-10, 10]
    x_scaled = [xi * 20 - 10 for xi in x]

    n = len(x_scaled)
    w = [1 + (xi - 1) / 4 for xi in x_scaled]

    term1 = math.sin(math.pi * w[0]) ** 2
    term3 = (w[-1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[-1]) ** 2)

    term2 = sum(
        (w[i] - 1) ** 2 * (1 + 10 * math.sin(math.pi * w[i] + 1) ** 2) for i in range(n - 1)
    )

    return term1 + term2 + term3 + random.gauss(0, 0.5)


async def main() -> None:
    """Run high-dimensional BO example via MCP protocol."""
    print_header("HIGH-DIMENSIONAL BO (TuRBO) via MCP Protocol")
    print()
    print("This demo optimizes a 25-dimensional function.")
    print("TuRBO automatically activates for high-dimensional problems.")

    # Configuration with many parameters
    n_params = 25
    config = {
        "name": "High-Dimensional Demo",
        "parameters": [
            {"name": f"x{i}", "type": "continuous", "bounds": [0.0, 1.0]} for i in range(n_params)
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "batch_size": 4,
    }

    print(f"\nConfiguration: {n_params} continuous parameters")
    print("  Each x_i in [0, 1]")
    print("  Objective: Levy function (optimum at x_i = 0.55)")

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
        best_value = float("inf")
        best_params: dict[str, float] = {}
        diagnostics = None

        for iteration in range(1, n_iterations + 1):
            # Generate suggestions
            suggestions_result = await call_tool(
                session, "generate_suggestions", {"campaign_id": campaign_id}
            )

            if not suggestions_result.get("success"):
                print(f"ERROR: {suggestions_result.get('errors')}")
                break

            # Display automatic method selection (should show TuRBO)
            if "method_selection" in suggestions_result:
                print_method_selection(suggestions_result["method_selection"])

            is_initial = iteration == 1
            phase = "Initial Design" if is_initial else "TuRBO-guided"
            print_iteration_header(iteration, phase)

            suggestions = suggestions_result["suggestions"]
            print(f"  Generated {len(suggestions)} suggestions")

            # Evaluate suggestions
            results_to_submit = []
            for i, s in enumerate(suggestions):
                params = s["parameter_values"]

                # Extract values in order
                x_vals = [params.get(f"x{j}", 0.5) for j in range(n_params)]
                y_val = levy_high_dim(x_vals)

                results_to_submit.append(
                    {
                        "parameter_values": params,
                        "objective_values": {"y": y_val},
                        "suggestion_id": s["suggestion_id"],
                    }
                )

                # Print summary (first 3 params + last param + objective)
                x0, x1, x2 = x_vals[0], x_vals[1], x_vals[2]
                xn = x_vals[-1]
                print(
                    f"  [{i + 1}] x0={x0:.3f}, x1={x1:.3f}, x2={x2:.3f}, "
                    f"..., x{n_params - 1}={xn:.3f} -> y={y_val:.2f}"
                )

                if y_val < best_value:
                    best_value = y_val
                    best_params = params.copy()

            # Submit results
            submit_result = await submit_demo_results(
                session, campaign_id, results_to_submit, owner_id
            )

            if not submit_result.get("success"):
                print(f"ERROR submitting: {submit_result.get('errors')}")
                break

            # Get diagnostics
            diagnostics = await call_tool(session, "get_diagnostics", {"campaign_id": campaign_id})

            if diagnostics.get("success"):
                print_diagnostics(diagnostics, iteration)

        # Final results
        print_header("HIGH-DIMENSIONAL OPTIMIZATION COMPLETE")
        print()
        print(f"  Total iterations: {n_iterations}")
        print(f"  Parameters: {n_params}")
        print(f"  Best value found: {best_value:.4f}")
        print()
        print("  Best parameters (first 5):")
        for i in range(min(5, n_params)):
            val = best_params.get(f"x{i}", 0)
            opt = 0.55  # Optimal at (1-(-10))/(20) = 0.55 in [0,1] scale
            print(f"    x{i}: {val:.4f} (optimal: ~{opt:.2f})")

        print()
        print("Key Takeaway:")
        print(f"  With {n_params} parameters, the MCP server:")
        print("  - Detected high-dimensional problem")
        print("  - Activated TuRBO trust region strategy")
        print("  - Maintains local search region that adapts")
        print("  - Shrinks region on failures, expands on successes")
        print()
        print("When TuRBO activates:")
        print("  - >20 continuous parameters")
        print("  - Single objective")
        print("  - Can also force with 'use_turbo': True")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
