#!/usr/bin/env python3
"""Mixed Parameter Types via MCP Protocol.

This script demonstrates optimization with different parameter types:
- Continuous (temperature)
- Discrete (pressure)
- Categorical (catalyst)

The MCP server automatically handles encoding and transforms for
mixed parameter types.

The system automatically:
- Applies one-hot encoding for categorical parameters
- Uses appropriate bounds handling for discrete parameters
- Combines all types into a unified GP model

Usage:
    uv run python scripts/demos/mcp_protocol_mixed_params.py
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
    print_header,
    print_iteration_header,
    print_method_selection,
    submit_demo_results,
)

# Catalyst effects lookup
CATALYST_EFFECTS = {
    "A": 0.0,  # Baseline
    "B": -0.5,  # Better
    "C": 0.3,  # Worse
}


def chemical_yield(temperature: float, pressure: int, catalyst: str) -> float:
    """Simulate a chemical reaction yield.

    Args:
        temperature: Reaction temperature (100-200 C)
        pressure: Pressure level (1-10 atm)
        catalyst: Catalyst type (A, B, or C)

    Returns:
        Yield percentage (higher is better)
    """
    # Optimal around temp=150, pressure=5
    temp_effect = -0.01 * (temperature - 150) ** 2
    pressure_effect = -0.1 * (pressure - 5) ** 2
    catalyst_effect = CATALYST_EFFECTS.get(catalyst, 0)

    # Interaction effects
    interaction = 0.005 * temperature * pressure / 10

    yield_pct = 80 + temp_effect + pressure_effect + catalyst_effect + interaction
    yield_pct += random.gauss(0, 2)  # Noise

    return yield_pct


async def main() -> None:
    """Run mixed parameter types example via MCP protocol."""
    print_header("MIXED PARAMETER TYPES via MCP Protocol")
    print()
    print("This demo optimizes with different parameter types:")
    print("  - Continuous: temperature (100-200)")
    print("  - Discrete: pressure (1-10)")
    print("  - Categorical: catalyst (A, B, C)")

    # Configuration with mixed parameter types
    config = {
        "name": "Mixed Parameters Demo",
        "parameters": [
            {
                "name": "temperature",
                "type": "continuous",
                "bounds": [100.0, 200.0],
            },
            {
                "name": "pressure",
                "type": "discrete",
                "bounds": [1, 10],
            },
            {
                "name": "catalyst",
                "type": "categorical",
                "categories": ["A", "B", "C"],
            },
        ],
        "objectives": [{"name": "yield", "direction": "maximize"}],
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
        best_value = float("-inf")  # Maximizing yield
        best_params: dict = {}
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
            phase = "Initial Design" if is_initial else "BO-guided"
            print_iteration_header(iteration, phase)

            suggestions = suggestions_result["suggestions"]
            print(f"  Generated {len(suggestions)} suggestions")

            # Evaluate suggestions
            results_to_submit = []
            for i, s in enumerate(suggestions):
                params = s["parameter_values"]
                temp = params.get("temperature", 150)
                press = int(params.get("pressure", 5))
                cat = params.get("catalyst", "A")

                yield_val = chemical_yield(temp, press, cat)

                results_to_submit.append(
                    {
                        "parameter_values": params,
                        "objective_values": {"yield": yield_val},
                        "suggestion_id": s["id"],
                    }
                )

                print(
                    f"  [{i + 1}] temp={temp:.1f}, press={press}, "
                    f"catalyst={cat} -> yield={yield_val:.2f}%"
                )

                if yield_val > best_value:
                    best_value = yield_val
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
        print_header("MIXED PARAMETER OPTIMIZATION COMPLETE")
        print()
        print(f"  Total iterations: {n_iterations}")
        print(f"  Best yield: {best_value:.2f}%")
        print()
        print("  Best parameters:")
        print(f"    Temperature: {best_params.get('temperature', 0):.1f} C (optimal: ~150)")
        print(f"    Pressure: {int(best_params.get('pressure', 0))} atm (optimal: ~5)")
        print(f"    Catalyst: {best_params.get('catalyst', '?')} (optimal: B)")

        print()
        print("Key Takeaway:")
        print("  Mixed parameter types are handled automatically:")
        print("  - Continuous: normalized to [0, 1]")
        print("  - Discrete: rounded to nearest valid value")
        print("  - Categorical: one-hot encoded internally")
        print()
        print("Transforms applied:")
        print("  - Normalize (continuous/discrete)")
        print("  - OneHotEncode (categorical)")
        print("  - Standardize (outputs)")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
