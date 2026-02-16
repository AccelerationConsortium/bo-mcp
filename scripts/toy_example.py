#!/usr/bin/env python3
"""Toy example: Run a complete Bayesian Optimization workflow."""

import asyncio
import hashlib
import random

import dotenv

dotenv.load_dotenv()

from bo_mcp_server.domain import User
from bo_mcp_server.storage import UserRepository, get_session, lifespan
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_diagnostics import get_diagnostics
from bo_mcp_server.tools.submit_results import submit_results

API_KEY = "dev-api-key-12345"


def simulate_experiment(params: dict) -> dict:
    """Simulate a chemical reaction experiment.

    This is a toy function that simulates yield and cost based on parameters.
    In real use, you would run actual experiments.
    """
    temp = params.get("temperature", 100)
    pressure = params.get("pressure", 5)
    catalyst = params.get("catalyst", "Pt")

    # Simulate yield (maximize) - higher near temp=100, pressure=5
    base_yield = 100 - 0.01 * (temp - 100) ** 2 - 0.5 * (pressure - 5) ** 2
    catalyst_bonus = {"Pt": 5, "Pd": 8, "Rh": 3}.get(catalyst, 0)
    noise = random.gauss(0, 2)
    yield_val = max(0, min(100, base_yield + catalyst_bonus + noise))

    # Simulate cost (minimize) - higher temp/pressure = higher cost
    base_cost = 50 + 0.5 * temp + 10 * pressure
    catalyst_cost = {"Pt": 50, "Pd": 30, "Rh": 70}.get(catalyst, 40)
    cost_noise = random.gauss(0, 5)
    cost_val = max(0, base_cost + catalyst_cost + cost_noise)

    return {"yield": round(yield_val, 2), "cost": round(cost_val, 2)}


async def main():
    """Run the toy example."""
    print("=" * 60)
    print("BO-MCP Toy Example: Chemical Yield Optimization")
    print("=" * 60)

    async with lifespan():
        # Create or get test user
        api_key_hash = hashlib.sha256(API_KEY.encode()).hexdigest()
        async with get_session() as session:
            repo = UserRepository(session)
            user = await repo.get_by_email("test@example.com")
            if not user:
                user = User(
                    name="Test User",
                    email="test@example.com",
                    api_key_hash=api_key_hash,
                )
                user = await repo.save(user)
                print(f"Created test user: {user.id}")
            else:
                print(f"Using existing user: {user.id}")

        owner_id = str(user.id)

        # Step 1: Create campaign
        print("\n" + "-" * 40)
        print("Step 1: Creating optimization campaign")
        print("-" * 40)

        campaign_data = {
            "name": "Chemical Yield Optimization",
            "description": "Optimize yield while minimizing cost",
            "parameters": [
                {
                    "name": "temperature",
                    "type": "continuous",
                    "bounds": [50.0, 150.0],
                    "description": "Reaction temperature (°C)",
                },
                {
                    "name": "pressure",
                    "type": "continuous",
                    "bounds": [1.0, 10.0],
                    "description": "Pressure (bar)",
                },
                {
                    "name": "catalyst",
                    "type": "categorical",
                    "categories": ["Pt", "Pd", "Rh"],
                    "description": "Catalyst type",
                },
            ],
            "objectives": [
                {"name": "yield", "direction": "maximize", "unit": "%"},
                {"name": "cost", "direction": "minimize", "unit": "USD"},
            ],
            "batch_size": 3,
        }

        result = await create_campaign(campaign_data, owner_id)
        if not result["success"]:
            print(f"Failed to create campaign: {result['errors']}")
            return

        campaign_id = result["campaign_id"]
        print(f"Campaign created: {campaign_id}")

        # Run optimization iterations
        n_iterations = 5
        for iteration in range(1, n_iterations + 1):
            print("\n" + "=" * 40)
            print(f"Iteration {iteration}")
            print("=" * 40)

            # Step 2: Generate suggestions
            print("\nGenerating suggestions...")
            suggestions_result = await generate_suggestions(campaign_id)

            if not suggestions_result["success"]:
                print(f"Failed to generate suggestions: {suggestions_result['errors']}")
                break

            suggestions = suggestions_result["suggestions"]
            print(f"Got {len(suggestions)} suggestions:")
            for i, s in enumerate(suggestions, 1):
                params = s["parameter_values"]
                print(
                    f"  {i}. temp={params['temperature']:.1f}°C, "
                    f"pressure={params['pressure']:.1f}bar, "
                    f"catalyst={params['catalyst']}"
                )

            # Step 3: Run experiments (simulated)
            print("\nRunning experiments (simulated)...")
            results_to_submit = []
            for s in suggestions:
                params = s["parameter_values"]
                objectives = simulate_experiment(params)
                results_to_submit.append(
                    {
                        "parameter_values": params,
                        "objective_values": objectives,
                        "suggestion_id": s["id"],
                    }
                )
                print(f"  Result: yield={objectives['yield']:.1f}%, cost=${objectives['cost']:.0f}")

            # Step 4: Submit results
            print("\nSubmitting results...")
            submit_result = await submit_results(
                campaign_id=campaign_id,
                results=results_to_submit,
                submitted_by=owner_id,
                source="api",
            )

            if not submit_result["success"]:
                print(f"Failed to submit results: {submit_result['errors']}")
                break

            print(f"Submitted {len(submit_result['result_ids'])} results")

            # Step 5: Get diagnostics
            diagnostics = await get_diagnostics(campaign_id)

            if diagnostics["success"]:
                print("\nDiagnostics:")
                print(f"  Total results: {diagnostics['n_results']}")
                print(f"  Pareto points: {diagnostics['n_pareto_points']}")
                print(f"  Hypervolume: {diagnostics['hypervolume']:.4f}")

                if diagnostics.get("pareto_front"):
                    print("\n  Pareto Front (best trade-offs):")
                    for p in diagnostics["pareto_front"][:5]:  # Show top 5
                        print(f"    yield={p['yield']:.1f}%, cost=${p['cost']:.0f}")

        print("\n" + "=" * 60)
        print("Optimization complete!")
        print("=" * 60)


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    asyncio.run(main())
