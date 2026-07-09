#!/usr/bin/env python3
"""Demonstration of Multi-Objective Bayesian Optimization via MCP Server.

This script demonstrates how to use the standalone MCP server for multi-objective
optimization (Pareto front discovery) without REST API or frontend.

Features demonstrated:
- Campaign creation with multiple objectives (2-4 objectives)
- Pareto-aware suggestion generation using qLogNEHVI
- Hypervolume tracking and Pareto front visualization
- Trade-off exploration

Usage:
    uv run python scripts/demos/mcp_multi_objective_demo.py
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


def branin_currin(x1: float, x2: float) -> tuple[float, float]:
    """Branin-Currin bi-objective test function.

    A standard benchmark for multi-objective optimization.
    Both objectives are to be minimized.
    """
    # Branin function
    x1_b = x1 * 15 - 5  # Scale to [-5, 10]
    x2_b = x2 * 15  # Scale to [0, 15]
    a, b, c, r, s, t = 1, 5.1 / (4 * math.pi**2), 5 / math.pi, 6, 10, 1 / (8 * math.pi)
    f1 = a * (x2_b - b * x1_b**2 + c * x1_b - r) ** 2 + s * (1 - t) * math.cos(x1_b) + s

    # Currin function
    factor1 = 1 - math.exp(-1 / (2 * max(x2, 1e-10)))
    numer = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
    denom = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20
    f2 = factor1 * numer / denom

    return f1 + random.gauss(0, 0.1), f2 + random.gauss(0, 0.1)


def tri_objective_problem(x1: float, x2: float, x3: float) -> tuple[float, float, float]:
    """Three-objective test problem for demonstrating 3+ objective support.

    Conflicting objectives that form a 3D Pareto front.
    """
    # Objective 1: Minimize (peaks far from origin)
    f1 = (x1 - 1) ** 2 + (x2 - 1) ** 2 + x3**2

    # Objective 2: Minimize (peaks near center)
    f2 = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 + (x3 - 0.5) ** 2

    # Objective 3: Minimize (peaks at origin)
    f3 = x1**2 + x2**2 + (x3 - 1) ** 2

    return (
        f1 + random.gauss(0, 0.01),
        f2 + random.gauss(0, 0.01),
        f3 + random.gauss(0, 0.01),
    )


async def run_bi_objective_demo(owner_id: str) -> None:
    """Run bi-objective optimization demonstration."""
    print("\n" + "=" * 60)
    print("PART 1: BI-OBJECTIVE OPTIMIZATION (Branin-Currin)")
    print("=" * 60)
    print()
    print("Optimizing two conflicting objectives simultaneously.")
    print("The result is a Pareto front of non-dominated solutions.")
    print()

    campaign_config = {
        "name": "Bi-Objective Optimization Demo",
        "description": "Branin-Currin bi-objective benchmark",
        "parameters": [
            {
                "name": "x1",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "description": "First design variable (normalized)",
            },
            {
                "name": "x2",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "description": "Second design variable (normalized)",
            },
        ],
        "objectives": [
            {"name": "f1_branin", "direction": "minimize", "unit": "value"},
            {"name": "f2_currin", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 3,
    }

    print("[A] Creating bi-objective campaign...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")

    # Run optimization
    n_iterations = 4

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 40}")
        print(f"Iteration {iteration}")
        print("─" * 40)

        suggestions_result = await generate_suggestions(campaign_id)
        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            x1, x2 = params.get("x1", 0.5), params.get("x2", 0.5)
            f1, f2 = branin_currin(x1, x2)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"f1_branin": f1, "f2_currin": f2},
                    "suggestion_id": s["id"],
                }
            )
            print(f"    x1={x1:.3f}, x2={x2:.3f} -> f1={f1:.2f}, f2={f2:.2f}")

        await submit_results(
            campaign_id=campaign_id,
            results=results_to_submit,
            submitted_by=owner_id,
            source="api",
        )

        diagnostics = await get_diagnostics(campaign_id)
        if diagnostics["success"]:
            print(f"    Hypervolume: {diagnostics.get('hypervolume', 0):.4f}")
            print(f"    Pareto points: {diagnostics.get('n_pareto_points', 0)}")

    # Final diagnostics
    print("\nFinal Bi-Objective Results:")
    final_diag = await get_diagnostics(campaign_id)
    if final_diag["success"]:
        print(f"  Total evaluations: {final_diag['n_results']}")
        print(f"  Pareto front size: {final_diag.get('n_pareto_points', 0)}")
        print(f"  Final hypervolume: {final_diag.get('hypervolume', 0):.4f}")

        if final_diag.get("pareto_front"):
            print("\n  Pareto Front (non-dominated solutions):")
            for i, point in enumerate(final_diag["pareto_front"][:5], 1):
                f1 = point.get("f1_branin", 0)
                f2 = point.get("f2_currin", 0)
                print(f"    {i}. f1={f1:.2f}, f2={f2:.2f}")


async def run_tri_objective_demo(owner_id: str) -> None:
    """Run three-objective optimization demonstration."""
    print("\n" + "=" * 60)
    print("PART 2: THREE-OBJECTIVE OPTIMIZATION")
    print("=" * 60)
    print()
    print("Demonstrating support for 3+ objectives.")
    print("The Pareto front becomes a Pareto surface in 3D.")
    print()

    campaign_config = {
        "name": "Tri-Objective Optimization Demo",
        "description": "Three-objective benchmark problem",
        "parameters": [
            {
                "name": "x1",
                "type": "continuous",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x2",
                "type": "continuous",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x3",
                "type": "continuous",
                "bounds": [0.0, 1.0],
            },
        ],
        "objectives": [
            {"name": "obj1", "direction": "minimize", "unit": "value"},
            {"name": "obj2", "direction": "minimize", "unit": "value"},
            {"name": "obj3", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 4,
    }

    print("[A] Creating tri-objective campaign...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")

    # Run optimization
    n_iterations = 3

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 40}")
        print(f"Iteration {iteration}")
        print("─" * 40)

        suggestions_result = await generate_suggestions(campaign_id)
        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            x1 = params.get("x1", 0.5)
            x2 = params.get("x2", 0.5)
            x3 = params.get("x3", 0.5)
            f1, f2, f3 = tri_objective_problem(x1, x2, x3)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"obj1": f1, "obj2": f2, "obj3": f3},
                    "suggestion_id": s["id"],
                }
            )
            print(f"    x=({x1:.2f},{x2:.2f},{x3:.2f}) -> ({f1:.3f},{f2:.3f},{f3:.3f})")

        await submit_results(
            campaign_id=campaign_id,
            results=results_to_submit,
            submitted_by=owner_id,
            source="api",
        )

        diagnostics = await get_diagnostics(campaign_id)
        if diagnostics["success"]:
            print(f"    Hypervolume: {diagnostics.get('hypervolume', 0):.4f}")
            print(f"    Pareto points: {diagnostics.get('n_pareto_points', 0)}")

    # Final results
    print("\nFinal Tri-Objective Results:")
    final_diag = await get_diagnostics(campaign_id)
    if final_diag["success"]:
        print(f"  Total evaluations: {final_diag['n_results']}")
        print(f"  Pareto surface size: {final_diag.get('n_pareto_points', 0)}")
        print(f"  Final hypervolume: {final_diag.get('hypervolume', 0):.4f}")


async def main() -> None:
    """Run multi-objective BO demonstration via MCP."""
    print("=" * 60)
    print("MULTI-OBJECTIVE BO DEMONSTRATION (MCP Standalone)")
    print("=" * 60)
    print()
    print("Multi-objective optimization finds Pareto-optimal solutions")
    print("that represent the best trade-offs between conflicting objectives.")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    owner_id = await get_or_create_demo_user(
        email="multi-obj@example.com",
        name="Multi-Objective Demo User",
        api_key="multi-objective-demo-key",
    )
    print(f"    Using demo user: {owner_id}")

    # Run demonstrations
    await run_bi_objective_demo(owner_id)
    await run_tri_objective_demo(owner_id)

    # Summary
    print("\n" + "=" * 60)
    print("MULTI-OBJECTIVE DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("Key Insights:")
    print("  1. qLogNEHVI acquisition explores the entire Pareto front")
    print("  2. Hypervolume measures overall optimization progress")
    print("  3. Pareto points represent best trade-off solutions")
    print("  4. No single 'best' solution - users choose based on preferences")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
