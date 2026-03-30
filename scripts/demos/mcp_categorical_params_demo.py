#!/usr/bin/env python3
"""Demonstration of Categorical and Discrete Parameters via MCP Server.

This script demonstrates how to use the standalone MCP server with different
parameter types: continuous, discrete (integer), and categorical.

Features demonstrated:
- Continuous parameters (real-valued)
- Discrete parameters (integer-valued)
- Categorical parameters (choice from options)
- Mixed parameter spaces

Usage:
    uv run python scripts/demos/mcp_categorical_params_demo.py
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


def catalyst_experiment(
    temperature: float,
    pressure: int,
    catalyst: str,
    concentration: float,
) -> float:
    """Simulate a chemical experiment with mixed parameter types.

    Args:
        temperature: Reaction temperature in Celsius (continuous)
        pressure: Pressure in bar (discrete integer)
        catalyst: Catalyst type (categorical)
        concentration: Reactant concentration (continuous)

    Returns:
        Yield percentage (to be maximized)
    """
    # Base yield depends on catalyst type
    catalyst_effects = {
        "Platinum": 0.85,
        "Palladium": 0.80,
        "Rhodium": 0.75,
        "Nickel": 0.65,
    }
    base_yield = catalyst_effects.get(catalyst, 0.5)

    # Temperature effect (optimal around 150C)
    temp_factor = 1 - 0.001 * (temperature - 150) ** 2

    # Pressure effect (linear increase with diminishing returns)
    pressure_factor = 1 - 0.5 * (1 / (1 + 0.1 * pressure))

    # Concentration effect (optimal around 0.5)
    conc_factor = 1 - 2 * (concentration - 0.5) ** 2

    # Interaction effects
    if catalyst == "Platinum" and temperature > 160:
        base_yield += 0.05  # Platinum works better at high temp
    if catalyst == "Nickel" and pressure > 8:
        base_yield += 0.03  # Nickel benefits from high pressure

    yield_val = base_yield * temp_factor * pressure_factor * conc_factor * 100
    yield_val = max(0, min(100, yield_val + random.gauss(0, 2)))

    return yield_val


async def main() -> None:
    """Run categorical parameters demonstration via MCP."""
    print("=" * 60)
    print("CATEGORICAL & MIXED PARAMETERS DEMO (MCP Standalone)")
    print("=" * 60)
    print()
    print("Demonstrating optimization with mixed parameter types:")
    print("  - Continuous: temperature, concentration")
    print("  - Discrete (integer): pressure")
    print("  - Categorical: catalyst type")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    api_key = "categorical-demo-key"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email("categorical@example.com")
        if not user:
            user = User(
                name="Categorical Demo User",
                email="categorical@example.com",
                api_key_hash=api_key_hash,
            )
            user = await repo.save(user)
            print(f"    Created user: {user.id}")
        else:
            print(f"    Using existing user: {user.id}")

    owner_id = str(user.id)

    # Define mixed-parameter optimization problem
    print("\n[3] Defining mixed-parameter optimization problem...")

    campaign_config = {
        "name": "Catalyst Optimization with Mixed Parameters",
        "description": "Optimize yield with mixed parameter types",
        "parameters": [
            {
                "name": "temperature",
                "type": "continuous",
                "bounds": [100.0, 200.0],
                "description": "Reaction temperature in Celsius",
            },
            {
                "name": "pressure",
                "type": "discrete",
                "bounds": [1, 15],
                "description": "Pressure in bar (integer)",
            },
            {
                "name": "catalyst",
                "type": "categorical",
                "choices": ["Platinum", "Palladium", "Rhodium", "Nickel"],
                "description": "Catalyst material",
            },
            {
                "name": "concentration",
                "type": "continuous",
                "bounds": [0.1, 0.9],
                "description": "Reactant concentration (mol/L)",
            },
        ],
        "objectives": [
            {"name": "yield", "direction": "maximize", "unit": "%"},
        ],
        "batch_size": 3,
    }

    print("    Parameters:")
    print("      temperature: continuous [100, 200] C")
    print("      pressure: discrete [1, 15] bar")
    print("      catalyst: categorical {Platinum, Palladium, Rhodium, Nickel}")
    print("      concentration: continuous [0.1, 0.9] mol/L")
    print("    Objective: maximize yield (%)")
    print()

    # Create campaign
    print("[4] Creating campaign...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    if result["warnings"]:
        print(f"    Warnings: {result['warnings']}")

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")

    # Run optimization iterations
    n_iterations = 5
    best_yield = 0.0
    best_params: dict[str, float | int | str] = {}
    catalyst_counts: dict[str, int] = {
        "Platinum": 0,
        "Palladium": 0,
        "Rhodium": 0,
        "Nickel": 0,
    }

    print("\n" + "=" * 60)
    print("OPTIMIZATION LOOP")
    print("=" * 60)

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Generate suggestions
        print("\n[A] Generating suggestions...")
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
            temp = params.get("temperature", 150.0)
            pres = int(params.get("pressure", 5))
            cat = params.get("catalyst", "Platinum")
            conc = params.get("concentration", 0.5)

            yield_val = catalyst_experiment(temp, pres, cat, conc)
            catalyst_counts[cat] = catalyst_counts.get(cat, 0) + 1

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"yield": yield_val},
                    "suggestion_id": s["id"],
                }
            )

            print(
                f"    T={temp:.0f}C, P={pres}bar, {cat:<10}, c={conc:.2f} -> yield={yield_val:.1f}%"
            )

            if yield_val > best_yield:
                best_yield = yield_val
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
            print(f"    Best yield: {best_yield:.1f}%")
            print(f"    Health status: {diagnostics.get('health_status', 'N/A')}")

    # Final summary
    print("\n" + "=" * 60)
    print("MIXED PARAMETERS OPTIMIZATION COMPLETE")
    print("=" * 60)

    print("\nResults Summary:")
    print(f"  Total iterations: {n_iterations}")
    print(f"  Best yield found: {best_yield:.1f}%")
    print()
    print("  Best parameters:")
    print(f"    Temperature: {best_params.get('temperature', 0):.1f} C")
    print(f"    Pressure: {int(best_params.get('pressure', 0))} bar")
    print(f"    Catalyst: {best_params.get('catalyst', 'N/A')}")
    print(f"    Concentration: {best_params.get('concentration', 0):.2f} mol/L")
    print()
    print("  Catalyst exploration (how often each was tried):")
    for cat, count in sorted(catalyst_counts.items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"    {cat:<10}: {count:2d} {bar}")
    print()
    print("Key Insight: BO automatically handles mixed parameter types,")
    print("using one-hot encoding for categoricals internally.")
    print()
    print(f"Campaign ID: {campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
