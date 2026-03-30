#!/usr/bin/env python3
"""
MCP Standalone Example: Using BO-MCP without REST API or Frontend

This script demonstrates how to use the BO-MCP tools directly in your own
Python workflows, without needing the FastAPI server or React frontend.

Usage:
    uv run python scripts/mcp_standalone_example.py
"""

import asyncio
import hashlib
import random

from bo_mcp_server.domain import User
from bo_mcp_server.storage import UserRepository, get_session, init_database
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_diagnostics import get_diagnostics
from bo_mcp_server.tools.submit_results import submit_results


def your_experiment(params: dict) -> dict:
    """
    Replace this with your actual experiment/simulation.

    This example simulates a simple function with two objectives.
    In real usage, you would:
    - Run a physical experiment
    - Call a simulation
    - Query an external system
    - etc.
    """
    x1 = params.get("x1", 5.0)
    x2 = params.get("x2", 5.0)

    # Objective 1: Maximize (peaks near x1=7, x2=3)
    obj1 = 100 - (x1 - 7) ** 2 - (x2 - 3) ** 2 + random.gauss(0, 1)

    # Objective 2: Minimize (lower is better near origin)
    obj2 = x1**2 + x2**2 + random.gauss(0, 0.5)

    return {"objective1": round(obj1, 2), "objective2": round(obj2, 2)}


async def main():
    """Run a complete optimization workflow using MCP tools directly."""

    print("=" * 60)
    print("BO-MCP Standalone Example")
    print("Using MCP tools directly without REST API or frontend")
    print("=" * 60)

    # =========================================================================
    # Step 1: Initialize the database
    # =========================================================================
    print("\n[1] Initializing database...")
    await init_database()

    # =========================================================================
    # Step 2: Create or get a user (required for ownership tracking)
    # =========================================================================
    print("[2] Setting up user...")
    api_key = "standalone-example-key"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email("standalone@example.com")
        if not user:
            user = User(
                name="Standalone User",
                email="standalone@example.com",
                api_key_hash=api_key_hash,
            )
            user = await repo.save(user)
            print(f"    Created user: {user.id}")
        else:
            print(f"    Using existing user: {user.id}")

    owner_id = str(user.id)

    # =========================================================================
    # Step 3: Define your optimization problem
    # =========================================================================
    print("[3] Defining optimization problem...")

    campaign_config = {
        "name": "Standalone Optimization Example",
        "description": "Demonstrating MCP standalone usage",
        "parameters": [
            {
                "name": "x1",
                "type": "continuous",
                "bounds": [0.0, 10.0],
                "description": "First design variable",
            },
            {
                "name": "x2",
                "type": "continuous",
                "bounds": [0.0, 10.0],
                "description": "Second design variable",
            },
        ],
        "objectives": [
            {"name": "objective1", "direction": "maximize", "unit": "score"},
            {"name": "objective2", "direction": "minimize", "unit": "cost"},
        ],
        "batch_size": 3,
    }

    # =========================================================================
    # Step 4: Create the campaign (includes validation)
    # =========================================================================
    print("[4] Creating campaign...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    if result["warnings"]:
        print(f"    Warnings: {result['warnings']}")

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")

    # =========================================================================
    # Step 6: Run optimization iterations
    # =========================================================================
    n_iterations = 3

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration}")
        print("=" * 60)

        # Generate suggestions
        print("\n[A] Generating suggestions...")
        suggestions_result = await generate_suggestions(campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions:")

        for i, s in enumerate(suggestions, 1):
            params = s["parameter_values"]
            print(f"      {i}. x1={params['x1']:.2f}, x2={params['x2']:.2f}")

        # Run experiments
        print("\n[B] Running experiments...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            objectives = your_experiment(params)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": objectives,
                    "suggestion_id": s["id"],
                }
            )

            print(
                f"      Result: obj1={objectives['objective1']:.1f}, "
                f"obj2={objectives['objective2']:.1f}"
            )

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
            print(f"    Total results: {diagnostics['n_results']}")
            print(f"    Pareto points: {diagnostics['n_pareto_points']}")
            print(f"    Hypervolume:   {diagnostics['hypervolume']:.4f}")
            print(f"    Health status: {diagnostics.get('health_status', 'N/A')}")

    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    final_diagnostics = await get_diagnostics(campaign_id)
    if final_diagnostics["success"] and final_diagnostics.get("pareto_front"):
        print("\nFinal Pareto Front (best trade-offs found):")
        for i, point in enumerate(final_diagnostics["pareto_front"], 1):
            print(
                f"  {i}. objective1={point['objective1']:.1f}, objective2={point['objective2']:.1f}"
            )

    print(f"\nCampaign ID: {campaign_id}")
    print("You can view this campaign in the web UI at:")
    print(f"  http://localhost:3001/campaign/{campaign_id}")


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    asyncio.run(main())
