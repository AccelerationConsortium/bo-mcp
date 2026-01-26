#!/usr/bin/env python3
"""Constrained Optimization (Mixture Formulation) via MCP Protocol.

This script demonstrates optimization with parameter constraints.
A common use case is mixture formulations where components must sum to 1.

Configuration:
- 3 parameters (a, b, c) representing mixture components
- Sum constraint: a + b + c = 1.0
- Objective: maximize quality

The system automatically:
- Validates constraints on input
- Ensures all suggestions satisfy constraints
- Uses constrained acquisition optimization

Usage:
    uv run python scripts/demos/mcp_protocol_constrained.py
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


def mixture_quality(a: float, b: float, c: float) -> float:
    """Simulate mixture quality based on component fractions.

    Args:
        a, b, c: Component fractions (should sum to 1)

    Returns:
        Quality score (higher is better)
    """
    # Optimal mixture around a=0.3, b=0.5, c=0.2
    # Quality peaks when components are in right proportions
    quality = (
        10 * a * b  # Synergy between a and b
        + 5 * b * c  # Synergy between b and c
        - 3 * a * c  # Negative interaction a and c
        + 2 * (1 - (a - 0.3) ** 2)  # Peak at a=0.3
        + 3 * (1 - (b - 0.5) ** 2)  # Peak at b=0.5
        + 1 * (1 - (c - 0.2) ** 2)  # Peak at c=0.2
    )

    quality += random.gauss(0, 0.1)  # Noise
    return quality


async def main() -> None:
    """Run constrained optimization example via MCP protocol."""
    print_header("CONSTRAINED OPTIMIZATION via MCP Protocol")
    print()
    print("This demo optimizes a mixture formulation where:")
    print("  - Three components (a, b, c)")
    print("  - Must sum to exactly 1.0")
    print("  - Maximize quality score")

    # Configuration with sum-to-one constraint
    config = {
        "name": "Mixture Formulation Demo",
        "parameters": [
            {"name": "a", "type": "continuous", "bounds": [0.0, 1.0]},
            {"name": "b", "type": "continuous", "bounds": [0.0, 1.0]},
            {"name": "c", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "quality", "direction": "maximize"}],
        "constraints": [
            {
                "type": "sum_equals",
                "parameters": ["a", "b", "c"],
                "value": 1.0,
            }
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
        best_value = float("-inf")  # Maximizing quality
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

            # Display automatic method selection
            if "method_selection" in suggestions_result:
                print_method_selection(suggestions_result["method_selection"])

            is_initial = iteration == 1
            phase = "Initial Design" if is_initial else "Constrained BO"
            print_iteration_header(iteration, phase)

            suggestions = suggestions_result["suggestions"]
            print(f"  Generated {len(suggestions)} suggestions")

            # Evaluate suggestions
            results_to_submit = []
            for i, s in enumerate(suggestions):
                params = s["parameter_values"]
                a_val = params.get("a", 0.33)
                b_val = params.get("b", 0.33)
                c_val = params.get("c", 0.34)

                # Verify constraint is satisfied
                total = a_val + b_val + c_val
                constraint_satisfied = abs(total - 1.0) < 0.01

                quality = mixture_quality(a_val, b_val, c_val)

                results_to_submit.append(
                    {
                        "parameter_values": params,
                        "objective_values": {"quality": quality},
                        "suggestion_id": s["id"],
                    }
                )

                constraint_str = "OK" if constraint_satisfied else f"VIOLATED (sum={total:.3f})"
                print(
                    f"  [{i + 1}] a={a_val:.3f}, b={b_val:.3f}, c={c_val:.3f} "
                    f"(sum={total:.3f}) -> quality={quality:.3f} [{constraint_str}]"
                )

                if quality > best_value:
                    best_value = quality
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
        print_header("CONSTRAINED OPTIMIZATION COMPLETE")
        print()
        print(f"  Total iterations: {n_iterations}")
        print(f"  Best quality: {best_value:.3f}")
        print()

        a_best = best_params.get("a", 0)
        b_best = best_params.get("b", 0)
        c_best = best_params.get("c", 0)
        total_best = a_best + b_best + c_best

        print("  Best mixture formulation:")
        print(f"    Component A: {a_best:.3f} (optimal: ~0.30)")
        print(f"    Component B: {b_best:.3f} (optimal: ~0.50)")
        print(f"    Component C: {c_best:.3f} (optimal: ~0.20)")
        print(f"    Sum: {total_best:.3f} (constraint: 1.00)")

        print()
        print("Key Takeaway:")
        print("  Parameter constraints are handled automatically:")
        print("  - sum_equals: components must sum to value")
        print("  - All suggestions satisfy the constraint")
        print("  - Acquisition optimization respects bounds")
        print()
        print("Other constraint types available:")
        print("  - sum_leq: a + b + c <= value")
        print("  - linear_leq: coefficients * params <= value")
        print("  - custom: user-defined constraint functions")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
