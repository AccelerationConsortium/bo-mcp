#!/usr/bin/env python3
"""Demonstration of Outcome Constraints via MCP Server.

This script demonstrates how to use the standalone MCP server with
outcome constraints - constraints that are learned from data rather
than known in advance.

Features demonstrated:
- Outcome constraint specification (threshold on objectives)
- Feasibility modeling via GP
- Constrained optimization with learned feasibility

Usage:
    uv run python scripts/demos/mcp_outcome_constraints_demo.py
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


def chemical_process(
    temperature: float,
    pressure: float,
    catalyst_loading: float,
) -> tuple[float, float]:
    """Simulate a chemical process with yield and purity outputs.

    Args:
        temperature: Process temperature (normalized 0-1)
        pressure: Process pressure (normalized 0-1)
        catalyst_loading: Catalyst amount (normalized 0-1)

    Returns:
        Tuple of (yield, purity) where:
        - yield: Reaction yield (0-100%), to be maximized
        - purity: Product purity (0-100%), must be >= 90% to be usable
    """
    # Yield peaks at moderate conditions
    yield_base = 80 + 20 * math.exp(-2 * ((temperature - 0.6) ** 2 + (pressure - 0.7) ** 2))

    # Catalyst helps but has diminishing returns
    yield_catalyst = 15 * math.sqrt(catalyst_loading)

    # High temp/pressure reduce yield
    yield_penalty = -10 * max(0, temperature - 0.8) - 10 * max(0, pressure - 0.8)

    yield_val = yield_base + yield_catalyst + yield_penalty + random.gauss(0, 2)
    yield_val = max(0, min(100, yield_val))

    # Purity degrades at extreme conditions and high catalyst
    purity_base = 95 - 30 * abs(temperature - 0.5) - 25 * abs(pressure - 0.5)
    purity_catalyst = -20 * max(0, catalyst_loading - 0.5)  # Too much catalyst hurts purity

    purity_val = purity_base + purity_catalyst + random.gauss(0, 1)
    purity_val = max(0, min(100, purity_val))

    return yield_val, purity_val


async def main() -> None:
    """Run outcome constraints demonstration via MCP."""
    print("=" * 60)
    print("OUTCOME CONSTRAINTS DEMO (MCP Standalone)")
    print("=" * 60)
    print()
    print("Outcome constraints define feasibility thresholds on objectives")
    print("that are LEARNED from observed data, not known in advance.")
    print()
    print("Example: Maximize yield subject to purity >= 90%")
    print("  - Purity constraint is modeled by a GP")
    print("  - P(purity >= 90% | x) is predicted for each suggestion")
    print("  - Acquisition is weighted by feasibility probability")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    owner_id = await get_or_create_demo_user(
        email="outcome@example.com",
        name="Outcome Constraints Demo User",
        api_key="outcome-constraints-demo-key",
    )
    print(f"    Using demo user: {owner_id}")

    # Define optimization problem with outcome constraint
    print("\n[3] Defining optimization problem with outcome constraint...")

    campaign_config = {
        "name": "Chemical Process Optimization",
        "description": "Maximize yield subject to purity >= 90%",
        "parameters": [
            {
                "name": "temperature",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "description": "Process temperature (normalized)",
            },
            {
                "name": "pressure",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "description": "Process pressure (normalized)",
            },
            {
                "name": "catalyst_loading",
                "type": "continuous",
                "bounds": [0.1, 0.8],
                "description": "Catalyst loading (normalized)",
            },
        ],
        "objectives": [
            {"name": "yield", "direction": "maximize", "unit": "%"},
            {"name": "purity", "direction": "maximize", "unit": "%"},
        ],
        "outcome_constraints": [
            {
                "objective_name": "purity",
                "threshold": 90.0,
                "greater_than": True,  # purity >= 90%
            }
        ],
        "batch_size": 3,
    }

    print("    Parameters: temperature, pressure, catalyst_loading")
    print("    Objective: maximize yield")
    print("    Constraint: purity >= 90% (learned from data)")
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
    best_feasible_yield = 0.0
    best_params: dict[str, float] = {}
    n_feasible = 0
    n_infeasible = 0

    print("\n" + "=" * 60)
    print("CONSTRAINED OPTIMIZATION LOOP")
    print("=" * 60)
    print()
    print("Legend: [OK] = purity >= 90% (feasible), [X] = purity < 90% (infeasible)")
    print()

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Generate suggestions
        print("\n[A] Generating suggestions (with feasibility weighting)...")
        suggestions_result = await generate_suggestions(campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Run experiments
        print("\n[B] Running experiments...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            temp = params.get("temperature", 0.5)
            pres = params.get("pressure", 0.5)
            cat = params.get("catalyst_loading", 0.3)

            yield_val, purity_val = chemical_process(temp, pres, cat)
            is_feasible = purity_val >= 90.0

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"yield": yield_val, "purity": purity_val},
                    "suggestion_id": s["suggestion_id"],
                }
            )

            status = "[OK]" if is_feasible else "[X] "
            print(
                f"    T={temp:.2f}, P={pres:.2f}, cat={cat:.2f} -> "
                f"yield={yield_val:.1f}%, purity={purity_val:.1f}% {status}"
            )

            if is_feasible:
                n_feasible += 1
                if yield_val > best_feasible_yield:
                    best_feasible_yield = yield_val
                    best_params = params.copy()
            else:
                n_infeasible += 1

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
            total = n_feasible + n_infeasible
            feas_rate = 100 * n_feasible / total if total > 0 else 0
            print(f"    Total evaluations: {diagnostics['n_results']}")
            print(f"    Feasibility rate: {feas_rate:.0f}% ({n_feasible}/{total})")
            print(f"    Best feasible yield: {best_feasible_yield:.1f}%")

    # Final summary
    print("\n" + "=" * 60)
    print("OUTCOME CONSTRAINTS OPTIMIZATION COMPLETE")
    print("=" * 60)

    total = n_feasible + n_infeasible
    feas_rate = 100 * n_feasible / total if total > 0 else 0

    print("\nResults Summary:")
    print(f"  Total evaluations: {total}")
    print(f"  Feasible (purity >= 90%): {n_feasible} ({feas_rate:.0f}%)")
    print(f"  Infeasible (purity < 90%): {n_infeasible}")
    print()
    print(f"  Best feasible yield: {best_feasible_yield:.1f}%")
    if best_params:
        print("  Best feasible parameters:")
        print(f"    Temperature: {best_params.get('temperature', 0):.3f}")
        print(f"    Pressure: {best_params.get('pressure', 0):.3f}")
        print(f"    Catalyst loading: {best_params.get('catalyst_loading', 0):.3f}")
    print()
    print("Key Insight: Outcome constraints let you optimize with")
    print("constraints that are unknown until measured. The GP learns")
    print("the feasible region from data and guides search toward")
    print("both high-yield AND feasible points.")
    print()
    print(f"Campaign ID: {campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
