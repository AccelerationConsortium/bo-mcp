#!/usr/bin/env python3
"""Simplest possible Bayesian Optimization via MCP Protocol.

This script demonstrates how minimal user input leads to a fully functional
Bayesian Optimization workflow. The MCP server automatically selects the
best model and acquisition function based on your problem specification.

Minimal Configuration Required:
- 1 parameter (x)
- 1 objective (y, minimize)
- That's it!

The system automatically:
- Creates a SingleTaskGP model
- Uses qLogNEI acquisition function
- Applies appropriate input transforms
- Handles all the complex BO machinery

Usage:
    uv run python scripts/demos/mcp_protocol_simple.py
"""

import asyncio
import random

from mcp_client_utils import (
    call_tool,
    connect_to_mcp_server,
    create_demo_campaign,
    get_or_create_demo_user,
    print_config_summary,
    print_diagnostics,
    print_final_results,
    print_header,
    print_iteration_header,
    print_method_selection,
    print_suggestion,
    submit_demo_results,
)


def quadratic_objective(x: float) -> float:
    """Simple quadratic function: f(x) = (x - 3)^2.

    Optimal at x = 3 with f(3) = 0.
    """
    return (x - 3) ** 2 + random.gauss(0, 0.1)


async def main() -> None:
    """Run simplest BO example via MCP protocol."""
    print_header("BAYESIAN OPTIMIZATION via MCP Protocol")
    print()
    print("This demo shows the simplest possible BO setup:")
    print("  - 1 continuous parameter")
    print("  - 1 objective to minimize")
    print("  - Everything else is automatic!")

    # MINIMAL CONFIGURATION - This is all you need!
    config = {
        "name": "Simple BO Demo",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 10.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }

    print_config_summary(config)

    # Setup demo user (required for campaign ownership)
    print("\nSetting up demo user...")
    owner_id = await get_or_create_demo_user()
    print(f"Using user: {owner_id}")

    print("\nConnecting to MCP server...")

    async with connect_to_mcp_server() as session:
        print("Server connected successfully.")

        # Step 1: Create campaign
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
            # Step 2: Generate suggestions
            suggestions_result = await call_tool(
                session, "generate_suggestions", {"campaign_id": campaign_id}
            )

            if not suggestions_result.get("success"):
                print(f"ERROR: {suggestions_result.get('errors')}")
                break

            # Display automatic method selection (transparency!)
            if "method_selection" in suggestions_result:
                print_method_selection(suggestions_result["method_selection"])

            # Determine phase
            is_initial = iteration == 1
            phase = "Initial Design" if is_initial else "BO-guided"
            print_iteration_header(iteration, phase)

            suggestions = suggestions_result["suggestions"]
            print(f"  Generated {len(suggestions)} suggestions")

            # Step 3: Evaluate suggestions (user's objective function)
            results_to_submit = []
            for i, s in enumerate(suggestions):
                params = s["parameter_values"]
                x_val = params.get("x", 5.0)

                # Evaluate objective
                y_val = quadratic_objective(x_val)

                results_to_submit.append(
                    {
                        "parameter_values": params,
                        "objective_values": {"y": y_val},
                        "suggestion_id": s["suggestion_id"],
                    }
                )

                print_suggestion(params, {"y": y_val}, index=i + 1)

                # Track best
                if y_val < best_value:
                    best_value = y_val
                    best_params = params.copy()

            # Step 4: Submit results
            submit_result = await submit_demo_results(
                session, campaign_id, results_to_submit, owner_id
            )

            if not submit_result.get("success"):
                print(f"ERROR submitting: {submit_result.get('errors')}")
                break

            # Step 5: Get diagnostics
            diagnostics = await call_tool(session, "get_diagnostics", {"campaign_id": campaign_id})

            if diagnostics.get("success"):
                print_diagnostics(diagnostics, iteration)

        # Final results
        print_final_results(best_params, best_value, n_iterations, diagnostics)

        print()
        print("Key Takeaway:")
        print("  With just 3 lines of config, the MCP server:")
        print("  - Selected SingleTaskGP model")
        print("  - Used qLogNEI acquisition (handles noise)")
        print("  - Applied Normalize + Standardize transforms")
        print("  - Ran a complete BO workflow!")
        print()
        print("True optimum: x=3.0, y=0.0")
        print(f"Found: x={best_params.get('x', 0):.3f}, y={best_value:.4f}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
