#!/usr/bin/env python3
"""Demonstration of Multi-Fidelity Bayesian Optimization via MCP Server.

Multi-fidelity BO allows using cheap/inaccurate evaluations to guide
optimization while reserving expensive/accurate evaluations for promising regions.

This script demonstrates:
1. Setting up a campaign with a fidelity parameter
2. Running multi-fidelity optimization via MCP tools
3. How fidelity affects cost and accuracy trade-offs

Usage:
    uv run python scripts/multifidelity_mcp_demo.py
"""

import asyncio
import math
import random

from demos.mcp_client_utils import get_or_create_demo_user

from bo_mcp_server.storage import init_database
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_diagnostics import get_diagnostics
from bo_mcp_server.tools.submit_results import submit_results


def multi_fidelity_objective(x1: float, x2: float, fidelity: float) -> tuple[float, float]:
    """Simulate a multi-fidelity objective function.

    The function value approaches the true objective as fidelity increases.
    Cost increases with fidelity.

    Args:
        x1: First parameter
        x2: Second parameter
        fidelity: Fidelity level (0 to 1), where 1 is highest accuracy

    Returns:
        Tuple of (objective_value, evaluation_cost)
    """
    # True objective: Branin-like function
    true_value = (
        (x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5 / math.pi * x1 - 6) ** 2
        + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1)
        + 10
    )

    # Low-fidelity approximation has bias and noise
    bias = (1 - fidelity) * 0.5 * (x1 + x2)  # Bias decreases with fidelity
    noise_scale = (1 - fidelity) * 2.0  # Noise decreases with fidelity
    noise = random.gauss(0, noise_scale)

    observed_value = true_value + bias + noise

    # Cost model: exponential in fidelity
    cost = 1.0 + 99.0 * fidelity**2  # Cost from 1 (low) to 100 (high)

    return observed_value, cost


async def main():
    """Run multi-fidelity BO demonstration via MCP."""
    print("=" * 60)
    print("MULTI-FIDELITY BO DEMONSTRATION (MCP Standalone)")
    print("=" * 60)
    print()
    print("Multi-fidelity optimization uses cheap approximations to")
    print("guide the search, reserving expensive evaluations for")
    print("promising candidates.")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    owner_id = await get_or_create_demo_user(
        email="multifidelity@example.com",
        name="Multi-Fidelity Demo User",
        api_key="multifidelity-demo-key",
    )
    print(f"    Using demo user: {owner_id}")

    # Define optimization problem with fidelity parameter
    print("\n[3] Defining multi-fidelity optimization problem...")

    campaign_config = {
        "name": "Multi-Fidelity Branin Optimization",
        "description": "Demonstrating multi-fidelity BO with cheap/expensive evaluations",
        "parameters": [
            {
                "name": "x1",
                "type": "continuous",
                "bounds": [-5.0, 10.0],
                "description": "First design variable",
            },
            {
                "name": "x2",
                "type": "continuous",
                "bounds": [0.0, 15.0],
                "description": "Second design variable",
            },
        ],
        "objectives": [
            {"name": "f", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 2,
        # v2.0: Multi-fidelity configuration
        "fidelity_parameter": {
            "name": "fidelity",
            "bounds": [0.1, 1.0],  # Fidelity from 10% to 100%
            "target": 1.0,  # We want optimal at full fidelity
            "cost_weight": 1.0,
            "fixed_cost": 1.0,
        },
        "acquisition_method": "qMFKG",
    }

    print("    Parameters: x1 ∈ [-5, 10], x2 ∈ [0, 15]")
    print("    Fidelity: 10% to 100% (higher = more accurate, more expensive)")
    print("    Objective: Minimize f (Branin-like function)")
    print()

    # Create campaign
    print("[4] Creating campaign...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")

    # Run optimization iterations
    n_iterations = 4
    total_cost = 0.0
    best_value = float("inf")
    best_at_target_fidelity = float("inf")

    print("\n" + "=" * 60)
    print("OPTIMIZATION LOOP")
    print("=" * 60)

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Generate suggestions (with fidelity recommendations)
        print("\n[A] Generating suggestions with fidelity recommendations...")
        suggestions_result = await generate_suggestions(campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Run experiments at different fidelities
        print("\n[B] Running experiments...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            x1 = params.get("x1", 0.0)
            x2 = params.get("x2", 0.0)

            # For demo: vary fidelity based on iteration
            # Early iterations use low fidelity, later use high
            fidelity = min(0.3 + 0.2 * iteration, 1.0)

            obj_value, cost = multi_fidelity_objective(x1, x2, fidelity)
            total_cost += cost

            results_to_submit.append(
                {
                    "parameter_values": {**params, "fidelity": fidelity},
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["id"],
                    "metadata": {"cost": cost, "fidelity": fidelity},
                }
            )

            print(f"    x1={x1:.2f}, x2={x2:.2f}, fidelity={fidelity:.1f}")
            print(f"      f={obj_value:.4f}, cost={cost:.1f}")

            if fidelity >= 0.95 and obj_value < best_at_target_fidelity:
                best_at_target_fidelity = obj_value
            if obj_value < best_value:
                best_value = obj_value

        # Submit results
        print("\n[C] Submitting results...")
        submit_result = await submit_results(
            campaign_id=campaign_id,
            results=results_to_submit,
            submitted_by=owner_id,
            source="api",
        )

        if not submit_result["success"]:
            print(f"    ERROR: {submit_result['errors']}")
            break

        print(f"    Submitted {len(submit_result['result_ids'])} results")

        # Get diagnostics
        print("\n[D] Checking progress...")
        diagnostics = await get_diagnostics(campaign_id)

        if diagnostics["success"]:
            print(f"    Total evaluations: {diagnostics['n_results']}")
            print(f"    Best f value (any fidelity): {best_value:.4f}")
            print(f"    Best f value (target fidelity): {best_at_target_fidelity:.4f}")
            print(f"    Total cost spent: {total_cost:.1f}")

    # Final summary
    print("\n" + "=" * 60)
    print("MULTI-FIDELITY OPTIMIZATION COMPLETE")
    print("=" * 60)

    print("\nResults Summary:")
    print(f"  Total cost spent: {total_cost:.1f}")
    print(f"  Best value found (any fidelity): {best_value:.4f}")
    print(f"  Best value at target fidelity: {best_at_target_fidelity:.4f}")
    print()
    print("Key Insight: Multi-fidelity BO spent most budget on low-fidelity")
    print("evaluations to explore, then used high-fidelity for exploitation.")
    print()
    print(f"Campaign ID: {campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
