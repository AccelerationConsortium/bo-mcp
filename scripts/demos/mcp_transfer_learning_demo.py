#!/usr/bin/env python3
"""Demonstration of Transfer Learning (RGPE) via MCP Server.

This script demonstrates how to use the standalone MCP server with
transfer learning to leverage knowledge from prior optimization campaigns.

Features demonstrated:
- Creating prior campaigns with historical data
- RGPE (Rank-weighted GP Ensemble) configuration
- Transfer learning from multiple prior campaigns
- Accelerated optimization on related tasks

Usage:
    uv run python scripts/demos/mcp_transfer_learning_demo.py
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


def prior_task_objective(x1: float, x2: float) -> float:
    """Objective function for the prior task.

    A Branin-like function representing a previously optimized system.
    """
    result = (
        (x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5 / math.pi * x1 - 6) ** 2
        + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1)
        + 10
    )
    return result + random.gauss(0, 0.5)


def new_task_objective(x1: float, x2: float) -> float:
    """Objective function for the new task.

    Similar to prior but with modifications - represents a related
    but not identical problem (e.g., new product variant).
    """
    # Similar to prior but with offset and scale
    prior_val = prior_task_objective(x1, x2)
    # Small perturbation representing task difference
    perturbation = 0.5 * math.sin(x1 * 0.5) + 0.3 * math.cos(x2 * 0.3)
    return prior_val * 0.95 + perturbation + random.gauss(0, 0.3)


async def create_prior_campaign(owner_id: str, name: str, shift: float = 0.0) -> str | None:
    """Create and populate a prior campaign with historical data."""
    print(f"\n    Creating prior campaign: {name}...")

    prior_config = {
        "name": name,
        "description": f"Historical optimization (shift={shift})",
        "parameters": [
            {
                "name": "x1",
                "type": "continuous",
                "bounds": [-5.0, 10.0],
            },
            {
                "name": "x2",
                "type": "continuous",
                "bounds": [0.0, 15.0],
            },
        ],
        "objectives": [
            {"name": "f", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 3,
    }

    result = await create_campaign(prior_config, owner_id)
    if not result["success"]:
        print(f"      ERROR: {result['errors']}")
        return None

    campaign_id = result["campaign_id"]

    # Populate with historical data
    historical_points = [
        (-3.0, 2.0),
        (0.0, 7.5),
        (3.0, 3.0),
        (6.0, 10.0),
        (9.0, 5.0),
        (3.14 + shift, 2.27),  # Near optimum
        (9.42 + shift, 2.47),  # Near optimum
    ]

    for x1, x2 in historical_points:
        # Clip to bounds
        x1 = max(-5.0, min(10.0, x1))
        x2 = max(0.0, min(15.0, x2))

        suggestions_result = await generate_suggestions(campaign_id, batch_size=1)
        if not suggestions_result["success"]:
            continue

        s = suggestions_result["suggestions"][0]
        obj_value = prior_task_objective(x1, x2)

        await submit_results(
            campaign_id=campaign_id,
            results=[
                {
                    "parameter_values": {"x1": x1, "x2": x2},
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["suggestion_id"],
                }
            ],
            submitted_by=owner_id,
            source="api",
        )

    print(f"      Added {len(historical_points)} observations")
    return campaign_id


async def main() -> None:
    """Run transfer learning demonstration via MCP."""
    print("=" * 60)
    print("TRANSFER LEARNING (RGPE) DEMO (MCP Standalone)")
    print("=" * 60)
    print()
    print("RGPE (Rank-weighted GP Ensemble) transfers knowledge from")
    print("prior optimization campaigns to accelerate new optimization.")
    print()
    print("How it works:")
    print("  1. Fit separate GP models to prior campaign data")
    print("  2. Fit a GP to current campaign data")
    print("  3. Weight models by their ranking performance")
    print("  4. Use ensemble for suggestions")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    owner_id = await get_or_create_demo_user(
        email="transfer@example.com",
        name="Transfer Learning Demo User",
        api_key="transfer-learning-demo-key",
    )
    print(f"    Using demo user: {owner_id}")

    # Phase 1: Create prior campaigns
    print("\n" + "=" * 60)
    print("PHASE 1: CREATE PRIOR CAMPAIGNS")
    print("=" * 60)
    print()
    print("Creating historical campaigns to transfer knowledge from...")

    prior_ids = []

    # Create first prior campaign
    prior1_id = await create_prior_campaign(owner_id, "Prior Task A", shift=0.0)
    if prior1_id:
        prior_ids.append(prior1_id)

    # Create second prior campaign (slightly different)
    prior2_id = await create_prior_campaign(owner_id, "Prior Task B", shift=0.5)
    if prior2_id:
        prior_ids.append(prior2_id)

    if not prior_ids:
        print("Failed to create prior campaigns")
        return

    print(f"\n    Created {len(prior_ids)} prior campaigns")
    for pid in prior_ids:
        print(f"      - {pid}")

    # Phase 2: Create new campaign with transfer learning
    print("\n" + "=" * 60)
    print("PHASE 2: CREATE NEW CAMPAIGN WITH TRANSFER")
    print("=" * 60)
    print()
    print("Creating new campaign that leverages prior knowledge...")

    new_config = {
        "name": "New Task with Transfer Learning",
        "description": "Optimization leveraging prior campaign knowledge",
        "parameters": [
            {
                "name": "x1",
                "type": "continuous",
                "bounds": [-5.0, 10.0],
            },
            {
                "name": "x2",
                "type": "continuous",
                "bounds": [0.0, 15.0],
            },
        ],
        "objectives": [
            {"name": "f", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 2,
        # Transfer learning configuration
        "transfer_learning": {
            "prior_campaign_ids": prior_ids,
            "num_ranking_samples": 256,
        },
    }

    result = await create_campaign(new_config, owner_id)
    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    new_campaign_id = result["campaign_id"]
    print(f"    New campaign created: {new_campaign_id}")
    print(f"    Transferring from {len(prior_ids)} prior campaigns")

    # Phase 3: Run optimization on new task
    print("\n" + "=" * 60)
    print("PHASE 3: OPTIMIZE NEW TASK WITH TRANSFER")
    print("=" * 60)
    print()
    print("Running optimization with RGPE ensemble...")

    n_iterations = 4
    best_value = float("inf")
    best_params: dict[str, float] = {}

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Generate suggestions (using RGPE with prior knowledge)
        print("\n[A] Generating suggestions (with transfer learning)...")
        suggestions_result = await generate_suggestions(new_campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Show method info
        if "method_selection" in suggestions_result:
            ms = suggestions_result["method_selection"]
            print(f"    Model: {ms.get('model_type', 'N/A')}")
            if "transfer" in str(ms).lower():
                print("    Transfer learning: ACTIVE")

        # Run experiments on new task
        print("\n[B] Running experiments on new task...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            x1 = params.get("x1", 0.0)
            x2 = params.get("x2", 0.0)

            obj_value = new_task_objective(x1, x2)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["suggestion_id"],
                }
            )

            print(f"    x1={x1:6.2f}, x2={x2:6.2f} -> f={obj_value:.4f}")

            if obj_value < best_value:
                best_value = obj_value
                best_params = params.copy()

        # Submit results
        print("\n[C] Submitting results...")
        submit_result = await submit_results(
            campaign_id=new_campaign_id,
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
        diagnostics = await get_diagnostics(new_campaign_id)

        if diagnostics["success"]:
            print(f"    Total evaluations: {diagnostics['n_results']}")
            print(f"    Best f value: {best_value:.4f}")

    # Final summary
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING DEMONSTRATION COMPLETE")
    print("=" * 60)

    print("\nResults Summary:")
    print(f"  Prior campaigns used: {len(prior_ids)}")
    print(f"  New campaign iterations: {n_iterations}")
    print(f"  Best f value found: {best_value:.4f}")
    print()
    print("  Best parameters:")
    print(f"    x1 = {best_params.get('x1', 0):.4f}")
    print(f"    x2 = {best_params.get('x2', 0):.4f}")
    print()
    print("Key Insight: Transfer learning used prior optimization knowledge")
    print("to make better suggestions in early iterations. The RGPE ensemble")
    print("automatically weighs prior models by their relevance to the new task.")
    print()
    print("When to use Transfer Learning:")
    print("  1. New task is similar to prior tasks (same parameters)")
    print("  2. Prior tasks have sufficient data (10+ observations)")
    print("  3. New task has limited evaluation budget")
    print("  4. Tasks share similar optimal regions")
    print()
    print("When NOT to use Transfer Learning:")
    print("  - New task is fundamentally different")
    print("  - Prior data is low quality or sparse")
    print("  - Tasks have completely different optima")
    print()
    print(f"New campaign ID: {new_campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
