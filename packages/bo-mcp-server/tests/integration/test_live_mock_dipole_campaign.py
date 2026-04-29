"""Live API integration test for a mock categorical dipole campaign.

This test exercises the running docker-compose stack end-to-end while keeping
the "evaluation" step entirely local to the test file. It uses a deterministic
lookup table derived from measured dipole values, so the BO loop can be tested
without invoking PySCF or any external chemistry service.

Run with:
    uv run pytest -m docker
    packages/bo-mcp-server/tests/integration/test_live_mock_dipole_campaign.py
"""

from statistics import mean
from uuid import uuid4

import httpx
import pytest

pytestmark = pytest.mark.docker

BASE_URL = "http://localhost:8000"
API_KEY = "dev-api-key-12345"
HEADERS = {"X-API-Key": API_KEY}

OBJECTIVE_NAME = "dipole_moment_magnitude_debye"
MOLECULES = [
    "acetonitrile",
    "nitromethane",
    "acetone",
    "formaldehyde",
    "chloromethane",
    "methanol",
    "dimethyl ether",
    "trans-1,2-dichloroethene",
    "methane",
]
SOLVENTS = [
    "N-heptane",
    "Cyclohexane",
    "Toluene",
    "Chloroform",
    "Acetonitrile",
    "Methanol",
    "Water",
    "Dimethylsulfoxide",
]

# Exact measured values supplied by the user.
EXACT_DIPOLE_VALUES: dict[tuple[str, str], float] = {
    ("acetonitrile", "Chloroform"): 4.747596766109,
    ("nitromethane", "Water"): 4.021849186946,
    ("acetone", "Chloroform"): 3.637071446469,
    ("formaldehyde", "Toluene"): 2.485174816115,
    ("chloromethane", "Methanol"): 2.238627604854,
    ("chloromethane", "Water"): 2.238627604841,
    ("methanol", "Cyclohexane"): 1.871460184879,
    ("dimethyl ether", "Chloroform"): 1.468954578947,
    ("trans-1,2-dichloroethene", "N-heptane"): 0.001619133642,
    ("trans-1,2-dichloroethene", "Acetonitrile"): 0.001619133546,
    ("trans-1,2-dichloroethene", "Cyclohexane"): 0.001619133490,
    ("trans-1,2-dichloroethene", "Water"): 0.001619133297,
    ("trans-1,2-dichloroethene", "Toluene"): 0.001619133055,
    ("methane", "Chloroform"): 0.000029938639,
    ("methane", "Cyclohexane"): 0.000029938639,
    ("methane", "Acetonitrile"): 0.000029938639,
    ("methane", "Toluene"): 0.000029938639,
    ("methane", "N-heptane"): 0.000029938639,
    ("methane", "Dimethylsulfoxide"): 0.000029938639,
    ("methane", "Methanol"): 0.000029938639,
}

# Fallback per-molecule values for combinations not explicitly listed above.
# These keep the mock evaluator total over the full categorical product space.
MOLECULE_BASELINES: dict[str, float] = {
    "acetonitrile": 4.747596766109,
    "nitromethane": 4.021849186946,
    "acetone": 3.637071446469,
    "formaldehyde": 2.485174816115,
    "chloromethane": mean([2.238627604854, 2.238627604841]),
    "methanol": 1.871460184879,
    "dimethyl ether": 1.468954578947,
    "trans-1,2-dichloroethene": mean(
        [
            0.001619133642,
            0.001619133546,
            0.001619133490,
            0.001619133297,
            0.001619133055,
        ]
    ),
    "methane": 0.000029938639,
}


def _dipole_value(molecule: str, solvent: str) -> float:
    return EXACT_DIPOLE_VALUES.get((molecule, solvent), MOLECULE_BASELINES[molecule])


def _combo_key(suggestion: dict) -> tuple[str, str]:
    params = suggestion["parameter_values"]
    return params["molecule"], params["implicit_solvent"]


class TestLiveMockDipoleCampaign:
    """Integration test for categorical BO behavior with a mock dipole evaluator."""

    def test_duplicate_completed_points_are_not_resuggested(self) -> None:
        try:
            response = httpx.get(f"{BASE_URL}/health", timeout=30.0)
        except httpx.HTTPError as exc:
            pytest.skip(
                "docker-compose API is not reachable at "
                f"{BASE_URL}. Start the stack from the repo root before "
                f"running this test. Original error: {exc}"
            )
        assert response.status_code == 200, response.text
        assert response.json()["healthy"] is True

        intake = {
            "name": f"Mock Dipole Campaign {uuid4()}",
            "description": (
                "Live categorical BO campaign using a deterministic dipole lookup table "
                "instead of PySCF to probe duplicate-suggestion behavior."
            ),
            "parameters": [
                {
                    "name": "molecule",
                    "type": "categorical",
                    "categories": MOLECULES,
                    "description": "Mock molecule identifier",
                },
                {
                    "name": "implicit_solvent",
                    "type": "categorical",
                    "categories": SOLVENTS,
                    "description": "Mock implicit solvent name",
                },
            ],
            "objectives": [
                {
                    "name": OBJECTIVE_NAME,
                    "direction": "maximize",
                    "unit": "Debye",
                }
            ],
            "batch_size": 2,
            "max_iterations": 10,
            "random_seed": 42,
        }

        create = httpx.post(
            f"{BASE_URL}/api/campaigns",
            json={"intake": intake},
            headers=HEADERS,
            timeout=60.0,
        )
        assert create.status_code == 200, create.text
        create_data = create.json()
        assert create_data["success"] is True, create_data
        campaign_id = create_data["campaign_id"]

        seen_combinations: dict[tuple[str, str], int] = {}

        for iteration in range(1, 11):
            generate = httpx.post(
                f"{BASE_URL}/api/suggestions/{campaign_id}/generate",
                params={"batch_size": 2},
                headers=HEADERS,
                timeout=60.0,
            )
            assert generate.status_code == 200, generate.text
            generate_data = generate.json()
            assert generate_data["success"] is True, generate_data

            suggestions = generate_data["suggestions"]
            assert len(suggestions) == 2, generate_data

            batch_combinations = [_combo_key(s) for s in suggestions]
            assert len(batch_combinations) == len(set(batch_combinations)), (
                f"Duplicate suggestions returned within iteration {iteration}: {batch_combinations}"
            )

            repeated_from_completed = [
                combo for combo in batch_combinations if combo in seen_combinations
            ]
            assert not repeated_from_completed, (
                "Campaign re-suggested already completed categorical points before the "
                "search space was exhausted. "
                f"iteration={iteration} repeated={repeated_from_completed} "
                f"seen={sorted(seen_combinations)}"
            )

            results_payload = {
                "results": [
                    {
                        "suggestion_id": suggestion["id"],
                        "parameter_values": suggestion["parameter_values"],
                        "objective_values": {
                            OBJECTIVE_NAME: _dipole_value(*_combo_key(suggestion))
                        },
                    }
                    for suggestion in suggestions
                ],
                "source": "api",
            }

            submit = httpx.post(
                f"{BASE_URL}/api/results/{campaign_id}",
                json=results_payload,
                headers=HEADERS,
                timeout=60.0,
            )
            assert submit.status_code == 200, submit.text
            submit_data = submit.json()
            assert submit_data["success"] is True, (
                f"Result submission failed on iteration {iteration}: {submit_data}"
            )

            for combo in batch_combinations:
                seen_combinations[combo] = iteration
