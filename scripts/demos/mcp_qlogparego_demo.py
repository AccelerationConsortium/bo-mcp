#!/usr/bin/env python3
"""Demonstration of qLogNParEGO Acquisition Function via MCP Server.

This script demonstrates how to use the standalone MCP server with
qLogNParEGO for multi-objective optimization using Chebyshev scalarization.

Features demonstrated:
- qLogNParEGO vs qLogNEHVI acquisition
- Chebyshev scalarization with random weights
- Alternative multi-objective exploration
- When to use qLogNParEGO

Usage:
    uv run python scripts/demos/mcp_qlogparego_demo.py
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


def dtlz2_objectives(x: list[float], n_objectives: int = 2) -> list[float]:
    """DTLZ2 test function - standard multi-objective benchmark.

    Creates a spherical Pareto front.

    Args:
        x: Decision variables (first n_objectives-1 for angles, rest for radius)
        n_objectives: Number of objectives

    Returns:
        List of objective values (all to be minimized)
    """
    # Calculate g (controls distance from Pareto front)
    g = sum((xi - 0.5) ** 2 for xi in x[n_objectives - 1 :])

    # Calculate objectives
    objectives = []
    for i in range(n_objectives):
        f = 1 + g
        for j in range(n_objectives - 1 - i):
            f *= math.cos(x[j] * math.pi / 2)
        if i > 0:
            f *= math.sin(x[n_objectives - 1 - i] * math.pi / 2)
        objectives.append(f + random.gauss(0, 0.01))

    return objectives


def branin_currin(x1: float, x2: float) -> tuple[float, float]:
    """Branin-Currin bi-objective benchmark."""
    # Branin function
    x1_b = x1 * 15 - 5
    x2_b = x2 * 15
    a, b, c, r, s, t = 1, 5.1 / (4 * math.pi**2), 5 / math.pi, 6, 10, 1 / (8 * math.pi)
    f1 = a * (x2_b - b * x1_b**2 + c * x1_b - r) ** 2 + s * (1 - t) * math.cos(x1_b) + s

    # Currin function
    factor1 = 1 - math.exp(-1 / (2 * max(x2, 1e-10)))
    numer = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
    denom = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20
    f2 = factor1 * numer / denom

    return f1 + random.gauss(0, 0.1), f2 + random.gauss(0, 0.1)


async def run_parego_demo(owner_id: str) -> tuple[str, dict[str, float]]:
    """Run qLogNParEGO demonstration."""
    print("\n" + "=" * 60)
    print("PART 1: qLogNParEGO OPTIMIZATION")
    print("=" * 60)
    print()
    print("qLogNParEGO uses Chebyshev scalarization with random weights")
    print("to explore different parts of the Pareto front.")
    print()

    campaign_config = {
        "name": "qLogNParEGO Demo",
        "description": "Multi-objective optimization with Chebyshev scalarization",
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
        ],
        "objectives": [
            {"name": "f1", "direction": "minimize", "unit": "value"},
            {"name": "f2", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 3,
        "acquisition_method": "qLogNParEGO",
    }

    print("[A] Creating campaign with qLogNParEGO...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return "", {}

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")
    print("    Acquisition: qLogNParEGO")

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

        # Show acquisition info
        if "method_selection" in suggestions_result:
            ms = suggestions_result["method_selection"]
            print(f"    Acquisition: {ms.get('acquisition_function', 'N/A')}")

        results_to_submit = []
        for s in suggestions:
            params = s["parameter_values"]
            x1, x2 = params.get("x1", 0.5), params.get("x2", 0.5)
            f1, f2 = branin_currin(x1, x2)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"f1": f1, "f2": f2},
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

    # Get final diagnostics
    diagnostics = await get_diagnostics(campaign_id)
    results = {
        "hypervolume": diagnostics.get("hypervolume", 0),
        "n_pareto": diagnostics.get("n_pareto_points", 0),
        "n_results": diagnostics.get("n_results", 0),
    }

    print(f"\n    Final: HV={results['hypervolume']:.4f}, Pareto={results['n_pareto']}")

    return campaign_id, results


async def run_nehvi_demo(owner_id: str) -> tuple[str, dict[str, float]]:
    """Run qLogNEHVI demonstration for comparison."""
    print("\n" + "=" * 60)
    print("PART 2: qLogNEHVI OPTIMIZATION (Comparison)")
    print("=" * 60)
    print()
    print("qLogNEHVI directly optimizes hypervolume improvement.")
    print()

    campaign_config = {
        "name": "qLogNEHVI Demo",
        "description": "Multi-objective optimization with hypervolume",
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
        ],
        "objectives": [
            {"name": "f1", "direction": "minimize", "unit": "value"},
            {"name": "f2", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 3,
        "acquisition_method": "qLogNEHVI",
    }

    print("[A] Creating campaign with qLogNEHVI...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return "", {}

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")
    print("    Acquisition: qLogNEHVI")

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
                    "objective_values": {"f1": f1, "f2": f2},
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

    # Get final diagnostics
    diagnostics = await get_diagnostics(campaign_id)
    results = {
        "hypervolume": diagnostics.get("hypervolume", 0),
        "n_pareto": diagnostics.get("n_pareto_points", 0),
        "n_results": diagnostics.get("n_results", 0),
    }

    print(f"\n    Final: HV={results['hypervolume']:.4f}, Pareto={results['n_pareto']}")

    return campaign_id, results


async def main() -> None:
    """Run qLogNParEGO demonstration via MCP."""
    print("=" * 60)
    print("qLogNParEGO DEMO (MCP Standalone)")
    print("=" * 60)
    print()
    print("qLogNParEGO is an alternative multi-objective acquisition that")
    print("uses Chebyshev scalarization instead of hypervolume improvement.")
    print()
    print("How it works:")
    print("  1. Generate random weight vector w = (w1, w2, ...)")
    print("  2. Scalarize objectives: max_i{ w_i * (f_i - utopia_i) }")
    print("  3. Optimize this scalarized objective")
    print("  4. Different weights explore different Pareto regions")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    owner_id = await get_or_create_demo_user(
        email="parego@example.com",
        name="qLogNParEGO Demo User",
        api_key="qlogparego-demo-key",
    )
    print(f"    Using demo user: {owner_id}")

    # Run both demonstrations
    parego_id, parego_results = await run_parego_demo(owner_id)
    nehvi_id, nehvi_results = await run_nehvi_demo(owner_id)

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: qLogNParEGO vs qLogNEHVI")
    print("=" * 60)
    print()
    print("                    ParEGO     NEHVI")
    print("  " + "-" * 35)
    parego_hv = parego_results.get("hypervolume", 0)
    nehvi_hv = nehvi_results.get("hypervolume", 0)
    parego_pareto = parego_results.get("n_pareto", 0)
    nehvi_pareto = nehvi_results.get("n_pareto", 0)
    parego_n = parego_results.get("n_results", 0)
    nehvi_n = nehvi_results.get("n_results", 0)
    print(f"  Hypervolume:      {parego_hv:8.4f}  {nehvi_hv:8.4f}")
    print(f"  Pareto points:    {parego_pareto:8d}  {nehvi_pareto:8d}")
    print(f"  Total results:    {parego_n:8d}  {nehvi_n:8d}")

    # Final summary
    print("\n" + "=" * 60)
    print("qLogNParEGO DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("Key Differences:")
    print()
    print("  qLogNParEGO:")
    print("    - Uses random Chebyshev scalarizations")
    print("    - Faster than qLogNEHVI for many objectives")
    print("    - Better exploration of Pareto front extremes")
    print("    - Works well with 3+ objectives")
    print()
    print("  qLogNEHVI:")
    print("    - Directly optimizes hypervolume")
    print("    - More focused on dominated hypervolume")
    print("    - Can be slow for many objectives (>4)")
    print("    - Better for well-defined reference points")
    print()
    print("When to use qLogNParEGO:")
    print("  - 3+ objectives (faster than NEHVI)")
    print("  - Want diverse Pareto exploration")
    print("  - Reference point is hard to define")
    print()
    print("When to use qLogNEHVI:")
    print("  - 2-3 objectives")
    print("  - Hypervolume is the primary metric")
    print("  - Have a good reference point")
    print()
    print(f"qLogNParEGO Campaign ID: {parego_id}")
    print(f"qLogNEHVI Campaign ID: {nehvi_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
