#!/usr/bin/env python3
"""Demonstration of Cost-Aware Bayesian Optimization (EIpu) via MCP Server.

This script demonstrates how to use the standalone MCP server with
cost-aware optimization (Expected Improvement per Unit cost).

Features demonstrated:
- EIpu acquisition function
- Cost modeling and tracking
- Balancing improvement with evaluation cost
- When to use cost-aware optimization

Usage:
    uv run python scripts/demos/mcp_cost_aware_demo.py
"""

import asyncio
import hashlib
import math
import random

from bo_mcp_server.domain import User
from bo_mcp_server.storage import UserRepository, get_session, init_database
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_diagnostics import get_diagnostics
from bo_mcp_server.tools.submit_results import submit_results


def expensive_simulation(
    resolution: float,
    iterations: int,
    precision: float,
) -> tuple[float, float]:
    """Simulate an expensive computational experiment.

    Args:
        resolution: Grid resolution (0-1), higher = more accurate but slower
        iterations: Number of solver iterations (10-1000)
        precision: Numerical precision (0-1), higher = more accurate but slower

    Returns:
        Tuple of (objective_value, cost) where:
        - objective_value: Simulation result (to be minimized)
        - cost: Computational cost (wall-clock hours)
    """
    # True objective (Branin-like in resolution x precision space)
    res_scaled = resolution * 10  # Scale to [0, 10]
    prec_scaled = precision * 15  # Scale to [0, 15]

    objective = (
        (prec_scaled - 5.1 / (4 * math.pi**2) * res_scaled**2 + 5 / math.pi * res_scaled - 6) ** 2
        + 10 * (1 - 1 / (8 * math.pi)) * math.cos(res_scaled)
        + 10
    ) / 50  # Normalize

    # Add noise based on precision (lower precision = more noise)
    noise_scale = 0.5 * (1 - precision)
    objective += random.gauss(0, noise_scale)

    # Cost model: exponential in resolution and iterations
    # High resolution and many iterations are expensive
    base_cost = 0.1  # Minimum cost (hours)
    resolution_cost = 2.0 * math.exp(3 * resolution)  # Exponential in resolution
    iteration_cost = 0.001 * iterations  # Linear in iterations
    precision_cost = 1.0 * precision**2  # Quadratic in precision

    cost = base_cost + resolution_cost + iteration_cost + precision_cost
    return objective, cost


async def main() -> None:
    """Run cost-aware BO demonstration via MCP."""
    print("=" * 60)
    print("COST-AWARE BO (EIpu) DEMO (MCP Standalone)")
    print("=" * 60)
    print()
    print("Cost-aware BO optimizes Expected Improvement per Unit cost:")
    print("  EIpu(x) = EI(x) / E[cost(x)]")
    print()
    print("This balances objective improvement with evaluation cost,")
    print("preferring cheap experiments that still provide useful info.")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    api_key = "cost-aware-demo-key"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email("costaware@example.com")
        if not user:
            user = User(
                name="Cost-Aware Demo User",
                email="costaware@example.com",
                api_key_hash=api_key_hash,
            )
            user = await repo.save(user)
            print(f"    Created user: {user.id}")
        else:
            print(f"    Using existing user: {user.id}")

    owner_id = str(user.id)

    # Define optimization problem with cost tracking
    print("\n[3] Defining cost-aware optimization problem...")

    campaign_config = {
        "name": "Cost-Aware Simulation Optimization",
        "description": "Optimize simulation parameters considering computational cost",
        "parameters": [
            {
                "name": "resolution",
                "type": "continuous",
                "bounds": [0.1, 0.9],
                "description": "Grid resolution (higher = more accurate, more costly)",
            },
            {
                "name": "iterations",
                "type": "discrete",
                "bounds": [10, 500],
                "description": "Solver iterations (more = better, costs linear)",
            },
            {
                "name": "precision",
                "type": "continuous",
                "bounds": [0.1, 0.9],
                "description": "Numerical precision (higher = less noise, more costly)",
            },
        ],
        "objectives": [
            {"name": "result", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 3,
        "use_cost_aware": True,
        "acquisition_method": "EIpu",
    }

    print("    Parameters: resolution, iterations, precision")
    print("    Objective: minimize simulation result")
    print("    Cost model: exponential in resolution, linear in iterations")
    print("    Acquisition: EIpu (Expected Improvement per Unit cost)")
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
    best_params: dict[str, float | int] = {}
    total_cost = 0.0
    cost_budget = 50.0  # Budget in compute hours

    print("\n" + "=" * 60)
    print("COST-AWARE OPTIMIZATION LOOP")
    print("=" * 60)
    print(f"\nBudget: {cost_budget:.1f} compute hours")
    print("EIpu will prefer cheap experiments that provide useful information.")
    print()

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Check budget
        if total_cost >= cost_budget:
            print(f"\n    Budget exhausted ({total_cost:.1f} / {cost_budget:.1f} hours)")
            break

        # Generate suggestions
        print("\n[A] Generating suggestions (EIpu acquisition)...")
        suggestions_result = await generate_suggestions(campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Show method selection info
        if "method_selection" in suggestions_result:
            ms = suggestions_result["method_selection"]
            print(f"    Acquisition: {ms.get('acquisition_function', 'N/A')}")

        # Run experiments
        print("\n[B] Running experiments...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            res = params.get("resolution", 0.5)
            iters = int(params.get("iterations", 100))
            prec = params.get("precision", 0.5)

            obj_value, cost = expensive_simulation(res, iters, prec)
            total_cost += cost

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"result": obj_value},
                    "suggestion_id": s["id"],
                    "metadata": {"cost": cost},
                }
            )

            print(
                f"    res={res:.2f}, iter={iters:3d}, prec={prec:.2f} "
                f"-> result={obj_value:.4f}, cost={cost:.2f}h"
            )

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
            remaining = cost_budget - total_cost
            print(f"    Total evaluations: {diagnostics['n_results']}")
            print(f"    Best result: {best_value:.4f}")
            print(f"    Cost: {total_cost:.1f}h / {cost_budget:.1f}h ({remaining:.1f}h left)")

    # Final summary
    print("\n" + "=" * 60)
    print("COST-AWARE OPTIMIZATION COMPLETE")
    print("=" * 60)

    print("\nResults Summary:")
    print(f"  Total cost spent: {total_cost:.1f} compute hours")
    print(f"  Budget used: {100 * total_cost / cost_budget:.0f}%")
    print(f"  Best result found: {best_value:.4f}")
    print()
    print("  Best parameters:")
    print(f"    Resolution: {best_params.get('resolution', 0):.3f}")
    print(f"    Iterations: {int(best_params.get('iterations', 0))}")
    print(f"    Precision: {best_params.get('precision', 0):.3f}")
    print()
    print("Key Insight: Cost-aware BO tends to suggest cheaper experiments")
    print("(lower resolution, fewer iterations) while still seeking improvement.")
    print("This is efficient when evaluation costs vary significantly.")
    print()
    print("When to use EIpu:")
    print("  - Evaluation costs vary significantly across the design space")
    print("  - You have a limited compute/experiment budget")
    print("  - Cheap experiments still provide useful information")
    print()
    print("When NOT to use EIpu:")
    print("  - All evaluations have similar cost")
    print("  - Finding the absolute optimum is more important than efficiency")
    print("  - Cost is unknown until after evaluation")
    print()
    print(f"Campaign ID: {campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
