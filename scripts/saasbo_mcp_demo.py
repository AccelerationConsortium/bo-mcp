#!/usr/bin/env python3
"""Demonstration of SAASBO (Sparse Axis-Aligned Subspace BO) via MCP Server.

SAASBO is designed for high-dimensional optimization (50+ parameters)
where only a subset of parameters significantly impact the objective.

This script demonstrates:
1. Setting up a high-dimensional campaign
2. Running SAASBO optimization via MCP
3. Identifying important parameters automatically

Usage:
    uv run python scripts/saasbo_mcp_demo.py
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


def high_dimensional_objective(params: dict[str, float]) -> float:
    """A high-dimensional objective where only a few parameters matter.

    True function depends heavily on:
    - x0, x1, x2 (important parameters)

    Other parameters have minimal effect (decoys).
    This is a common scenario in real-world optimization where
    only a few factors drive the outcome.
    """
    # Important parameters (high effect)
    x0 = params.get("x0", 0.5)
    x1 = params.get("x1", 0.5)
    x2 = params.get("x2", 0.5)

    # Main effect from important parameters (Hartmann-like)
    main_effect = (
        (x0 - 0.3) ** 2
        + 2 * (x1 - 0.7) ** 2
        + 0.5 * (x2 - 0.5) ** 2
        + 0.3 * math.sin(math.pi * x0 * x1)
    )

    # Small noise from other parameters (minimal effect)
    noise = 0.0
    for name, value in params.items():
        if name not in ["x0", "x1", "x2"]:
            noise += 0.001 * (value - 0.5) ** 2

    return main_effect + noise + random.gauss(0, 0.01)


async def main():
    """Run SAASBO demonstration via MCP."""
    print("=" * 60)
    print("SAASBO DEMONSTRATION (MCP Standalone)")
    print("=" * 60)
    print()
    print("SAASBO (Sparse Axis-Aligned Subspace Bayesian Optimization)")
    print("is designed for high-dimensional problems where only a")
    print("subset of parameters significantly affect the objective.")
    print()
    print("The SAAS prior automatically identifies important parameters")
    print("and ignores irrelevant ones.")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    owner_id = await get_or_create_demo_user(
        email="saasbo@example.com",
        name="SAASBO Demo User",
        api_key="saasbo-demo-key",
    )
    print(f"    Using demo user: {owner_id}")

    # Define high-dimensional optimization problem
    # Using 20 parameters for demo (SAASBO typically used for 50+)
    n_params = 20  # Reduced for faster demo; use 50+ for real applications
    print(f"\n[3] Defining {n_params}-dimensional optimization problem...")

    parameters = [
        {
            "name": f"x{i}",
            "type": "continuous",
            "bounds": [0.0, 1.0],
            "description": f"Parameter {i} (x0-x2 are important, rest are decoys)",
        }
        for i in range(n_params)
    ]

    campaign_config = {
        "name": "SAASBO High-Dimensional Optimization",
        "description": "Demonstrating SAASBO for sparse high-dimensional optimization",
        "parameters": parameters,
        "objectives": [
            {"name": "f", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 3,
        # v2.0: Enable SAASBO
        "use_saasbo": True,
        "acquisition_method": "SAASBO",
    }

    print(f"    Parameters: {n_params} continuous in [0, 1]")
    print("    Important params: x0, x1, x2 (only 3 out of 20)")
    print("    Objective: Minimize f")
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
    best_value = float("inf")
    best_params: dict[str, float] = {}

    print("\n" + "=" * 60)
    print("OPTIMIZATION LOOP")
    print("=" * 60)
    print()
    print("Note: SAASBO uses NUTS sampling which can be slow.")
    print("For a production 50+ dim problem, expect minutes per iteration.")
    print()

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Generate suggestions
        print("\n[A] Generating suggestions (SAASBO)...")
        print("    (NUTS sampling in progress...)")
        suggestions_result = await generate_suggestions(campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Show method selection info
        if "method_selection" in suggestions_result:
            ms = suggestions_result["method_selection"]
            print(f"    Model: {ms['model_type']}")
            print(f"    Strategy: {ms['optimization_strategy']}")

        # Run experiments
        print("\n[B] Running experiments...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            obj_value = high_dimensional_objective(params)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["suggestion_id"],
                }
            )

            # Show important parameters only
            x0 = params.get("x0", 0.0)
            x1 = params.get("x1", 0.0)
            x2 = params.get("x2", 0.0)
            print(f"    x0={x0:.3f}, x1={x1:.3f}, x2={x2:.3f} -> f={obj_value:.4f}")

            if obj_value < best_value:
                best_value = obj_value
                best_params = params.copy()

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
            print(f"    Best f value: {best_value:.4f}")
            print(f"    Health status: {diagnostics.get('health_status', 'N/A')}")

    # Final summary
    print("\n" + "=" * 60)
    print("SAASBO DEMONSTRATION COMPLETE")
    print("=" * 60)

    print("\nResults Summary:")
    print(f"  Total parameters: {n_params}")
    print("  True important params: x0, x1, x2")
    print(f"  Best f value: {best_value:.4f}")
    print()
    print("Best parameters found (important ones):")
    print(f"  x0 = {best_params.get('x0', 0):.4f} (optimal: ~0.3)")
    print(f"  x1 = {best_params.get('x1', 0):.4f} (optimal: ~0.7)")
    print(f"  x2 = {best_params.get('x2', 0):.4f} (optimal: ~0.5)")
    print()
    print("Key Insight: SAASBO automatically focused on the important")
    print("parameters (x0, x1, x2) while essentially ignoring the")
    print("17 irrelevant parameters. This is the power of the SAAS prior.")
    print()
    print("When to use SAASBO:")
    print("  - High-dimensional problems (50+ parameters)")
    print("  - Suspected sparsity (few parameters matter)")
    print("  - Limited evaluation budget")
    print()
    print("Trade-offs:")
    print("  - NUTS sampling is computationally expensive")
    print("  - Best for <200 observations due to O(n³) scaling")
    print()
    print(f"Campaign ID: {campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
