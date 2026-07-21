#!/usr/bin/env python3
"""Demonstration of TuRBO (Trust Region Bayesian Optimization) via MCP Server.

This script demonstrates how to use the standalone MCP server with TuRBO
for high-dimensional optimization (20+ parameters).

Features demonstrated:
- TuRBO trust region management
- High-dimensional optimization (25 parameters)
- Trust region expansion/contraction dynamics
- When to use TuRBO vs standard BO

Usage:
    uv run python scripts/demos/mcp_turbo_demo.py
"""

import asyncio
import math
import random

from mcp_client_utils import get_or_create_demo_user

from bo_mcp_server.storage import init_database
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_diagnostics import get_diagnostics
from bo_mcp_server.tools.submit_results import submit_results


def levy_function(params: dict[str, float]) -> float:
    """Levy function - a challenging high-dimensional test function.

    Has many local minima, making it ideal for testing TuRBO.
    Global minimum at x_i = 1 for all i, f(x*) = 0.
    """
    values = list(params.values())
    n = len(values)
    w = [(1 + (xi - 1) / 4) for xi in values]

    term1 = math.sin(math.pi * w[0]) ** 2
    term2 = sum(
        (w[i] - 1) ** 2 * (1 + 10 * math.sin(math.pi * w[i] + 1) ** 2) for i in range(n - 1)
    )
    term3 = (w[-1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[-1]) ** 2)

    return term1 + term2 + term3 + random.gauss(0, 0.1)


def sparse_high_dim_objective(params: dict[str, float]) -> float:
    """High-dimensional objective where only a few parameters matter.

    Parameters x0, x1, x2 are important (70% of effect).
    Other parameters contribute noise (~30% of effect).

    This is realistic for many real-world problems.
    """
    # Important parameters
    x0 = params.get("x0", 0.5)
    x1 = params.get("x1", 0.5)
    x2 = params.get("x2", 0.5)

    # Main effect (optimum at x0=0.3, x1=0.7, x2=0.5)
    main_effect = (x0 - 0.3) ** 2 + 2 * (x1 - 0.7) ** 2 + 0.5 * (x2 - 0.5) ** 2

    # Small noise from other parameters
    noise = sum(0.01 * (params.get(f"x{i}", 0.5) - 0.5) ** 2 for i in range(3, len(params)))

    return main_effect + noise + random.gauss(0, 0.01)


async def main() -> None:
    """Run TuRBO demonstration via MCP."""
    print("=" * 60)
    print("TuRBO (TRUST REGION BO) DEMO (MCP Standalone)")
    print("=" * 60)
    print()
    print("TuRBO is designed for high-dimensional optimization (20+ params)")
    print("where standard BO struggles due to the curse of dimensionality.")
    print()
    print("Key features:")
    print("  - Local trust region around best point")
    print("  - Expands after consecutive improvements")
    print("  - Contracts after failures")
    print("  - Restarts when stuck")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    owner_id = await get_or_create_demo_user(
        email="turbo@example.com",
        name="TuRBO Demo User",
        api_key="turbo-demo-key",
    )
    print(f"    Using demo user: {owner_id}")

    # Define high-dimensional optimization problem
    n_params = 25
    print(f"\n[3] Defining {n_params}-dimensional optimization problem...")

    parameters = [
        {
            "name": f"x{i}",
            "type": "continuous",
            "bounds": [-5.0, 10.0],
            "description": f"Parameter {i}",
        }
        for i in range(n_params)
    ]

    campaign_config = {
        "name": "High-Dimensional TuRBO Optimization",
        "description": f"Optimizing {n_params}D Levy function using TuRBO",
        "parameters": parameters,
        "objectives": [
            {"name": "f", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 4,
        "use_turbo": True,
    }

    print(f"    Parameters: {n_params} continuous in [-5, 10]")
    print("    Objective: minimize f (Levy function)")
    print("    TuRBO: enabled")
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
    n_iterations = 5
    best_value = float("inf")
    best_params: dict[str, float] = {}

    print("\n" + "=" * 60)
    print("TURBO OPTIMIZATION LOOP")
    print("=" * 60)
    print()
    print("Watch the trust region dynamics as optimization progresses.")
    print("TuRBO focuses search in a local region around the best point.")
    print()

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Generate suggestions
        print("\n[A] Generating suggestions (with TuRBO)...")
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
            print(f"    Strategy: {ms.get('optimization_strategy', 'N/A')}")
            if "turbo_info" in ms:
                ti = ms["turbo_info"]
                print(f"    Trust region length: {ti.get('length', 'N/A')}")

        # Run experiments
        print("\n[B] Running experiments (evaluating Levy function)...")
        results_to_submit = []
        iteration_best = float("inf")

        for s in suggestions:
            params = s["parameter_values"]
            obj_value = levy_function(params)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["suggestion_id"],
                }
            )

            # Show first 3 params and result
            p0 = params.get("x0", 0)
            p1 = params.get("x1", 0)
            p2 = params.get("x2", 0)
            print(f"    x0={p0:.2f}, x1={p1:.2f}, x2={p2:.2f}, ... -> f={obj_value:.4f}")

            if obj_value < iteration_best:
                iteration_best = obj_value
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
            print(f"    This iteration best: {iteration_best:.4f}")

            # TuRBO dynamics summary
            if iteration_best < best_value * 1.001:  # Improvement
                print("    TuRBO: Likely expanding trust region (improvement)")
            else:
                print("    TuRBO: May contract trust region (no improvement)")

    # Final summary
    print("\n" + "=" * 60)
    print("TURBO OPTIMIZATION COMPLETE")
    print("=" * 60)

    print("\nResults Summary:")
    print(f"  Dimension: {n_params}")
    print(f"  Total iterations: {n_iterations}")
    print(f"  Best f value found: {best_value:.4f}")
    print("  Global optimum: f* = 0 at x_i = 1 for all i")
    print()
    print("  Best parameters (first 5 shown):")
    for i in range(min(5, n_params)):
        val = best_params.get(f"x{i}", 0)
        print(f"    x{i} = {val:.4f} (optimal: 1.0)")
    print()
    print("When to use TuRBO:")
    print("  - 20+ parameters (high-dimensional)")
    print("  - Single objective only")
    print("  - Local optimization is sufficient")
    print()
    print("When NOT to use TuRBO:")
    print("  - Low dimensions (<20 params)")
    print("  - Multi-objective optimization")
    print("  - Need global exploration")
    print()
    print(f"Campaign ID: {campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
