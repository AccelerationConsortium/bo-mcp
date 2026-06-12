"""Integration tests for all API endpoints against running Docker containers.

These tests require the API server to be running locally.
Run with: pytest -m docker
"""

import httpx
import pytest

pytestmark = pytest.mark.docker  # Mark all tests in this module

BASE_URL = "http://localhost:8000"
API_KEY = "dev-api-key-12345"
HEADERS = {"X-API-Key": API_KEY}


class TestAPIEndpoints:
    """Test all API endpoints in sequence to verify full workflow."""

    campaign_id: str = ""
    spec_id: str = ""
    suggestion_id: str = ""
    result_id: str = ""
    compare_campaign_id: str = ""
    transfer_source_campaign_id: str = ""
    transfer_target_campaign_id: str = ""

    def test_01_health_check(self):
        """Test health check endpoint."""
        response = httpx.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is True
        assert data["service"] == "api"
        assert data["database"] == "connected"
        assert isinstance(data["version"], str)
        assert data["uptime_seconds"] >= 0
        print("✓ Health check passed")

    def test_02_create_campaign(self):
        """Test campaign creation endpoint."""
        intake_data = {
            "intake": {
                "name": "API Test Campaign",
                "description": "Test campaign for API verification",
                "parameters": [
                    {
                        "name": "temperature",
                        "type": "continuous",
                        "bounds": [0.0, 100.0],
                    },
                    {
                        "name": "pressure",
                        "type": "continuous",
                        "bounds": [1.0, 10.0],
                    },
                ],
                "objectives": [
                    {"name": "yield", "direction": "maximize"},
                    {"name": "cost", "direction": "minimize"},
                ],
            }
        }
        response = httpx.post(
            f"{BASE_URL}/api/campaigns",
            json=intake_data,
            headers=HEADERS,
        )
        assert response.status_code == 201, f"Create campaign failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Campaign creation failed: {data.get('errors')}"
        assert data["campaign_id"] is not None
        assert "warnings" in data
        TestAPIEndpoints.campaign_id = data["campaign_id"]
        TestAPIEndpoints.spec_id = data["spec_id"]
        print(f"✓ Create campaign passed (id={TestAPIEndpoints.campaign_id[:8]}...)")

    def test_03_get_campaign_spec(self):
        """Test get campaign spec endpoint."""
        assert TestAPIEndpoints.spec_id, "No spec_id from previous test"
        response = httpx.get(
            f"{BASE_URL}/api/campaigns/spec/{TestAPIEndpoints.spec_id}",
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Get campaign spec failed: {response.text}"
        data = response.json()
        assert data["name"] == "API Test Campaign"
        assert len(data["parameters"]) == 2
        assert len(data["objectives"]) == 2
        print("✓ Get campaign spec passed")

    def test_04_get_campaign(self):
        """Test get campaign endpoint."""
        assert TestAPIEndpoints.campaign_id, "No campaign_id from previous test"
        response = httpx.get(
            f"{BASE_URL}/api/campaigns/{TestAPIEndpoints.campaign_id}",
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Get campaign failed: {response.text}"
        data = response.json()
        assert data["id"] == TestAPIEndpoints.campaign_id
        assert data["status"] == "created"
        print("✓ Get campaign passed")

    def test_05_list_campaigns(self):
        """Test list campaigns endpoint."""
        response = httpx.get(
            f"{BASE_URL}/api/campaigns",
            headers=HEADERS,
        )
        assert response.status_code == 200, f"List campaigns failed: {response.text}"
        data = response.json()
        assert "campaigns" in data
        assert data["total"] >= 1
        print(f"✓ List campaigns passed (total={data['total']})")

    def test_06_generate_suggestions(self):
        """Test generate suggestions endpoint."""
        assert TestAPIEndpoints.campaign_id, "No campaign_id from previous test"
        response = httpx.post(
            f"{BASE_URL}/api/suggestions/{TestAPIEndpoints.campaign_id}/generate",
            headers=HEADERS,
        )
        assert response.status_code == 201, f"Generate suggestions failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Generation failed: {data.get('errors')}"
        assert len(data["suggestions"]) > 0
        TestAPIEndpoints.suggestion_id = data["suggestions"][0]["id"]
        print(f"✓ Generate suggestions passed (count={len(data['suggestions'])})")

    def test_07_get_suggestions(self):
        """Test get suggestions endpoint."""
        assert TestAPIEndpoints.campaign_id, "No campaign_id from previous test"
        response = httpx.get(
            f"{BASE_URL}/api/suggestions/{TestAPIEndpoints.campaign_id}",
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Get suggestions failed: {response.text}"
        data = response.json()
        # Returns a list directly, not a dict
        assert isinstance(data, list), f"Expected list, got {type(data)}"
        assert len(data) > 0
        print(f"✓ Get suggestions passed (count={len(data)})")

    def test_08_submit_results(self):
        """Test submit results endpoint."""
        assert TestAPIEndpoints.campaign_id, "No campaign_id from previous test"
        assert TestAPIEndpoints.suggestion_id, "No suggestion_id from previous test"

        # Use 'api' as source (valid pattern: ^(gui|file_upload|api)$)
        results_data = {
            "results": [
                {
                    "suggestion_id": TestAPIEndpoints.suggestion_id,
                    "parameter_values": {"temperature": 50.0, "pressure": 5.0},
                    "objective_values": {"yield": 0.75, "cost": 100.0},
                }
            ],
            "source": "api",  # Must be: gui, file_upload, or api
        }
        response = httpx.post(
            f"{BASE_URL}/api/results/{TestAPIEndpoints.campaign_id}",
            json=results_data,
            headers=HEADERS,
        )
        assert response.status_code == 201, f"Submit results failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Result submission failed: {data.get('errors')}"
        assert len(data["result_ids"]) > 0
        TestAPIEndpoints.result_id = data["result_ids"][0]
        print(f"✓ Submit results passed (count={len(data['result_ids'])})")

    def test_09_get_results(self):
        """Test get results endpoint."""
        assert TestAPIEndpoints.campaign_id, "No campaign_id from previous test"
        response = httpx.get(
            f"{BASE_URL}/api/results/{TestAPIEndpoints.campaign_id}",
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Get results failed: {response.text}"
        data = response.json()
        # Returns a list directly, not a dict
        assert isinstance(data, list), f"Expected list, got {type(data)}"
        assert len(data) >= 1
        print(f"✓ Get results passed (count={len(data)})")

    def test_10_get_diagnostics(self):
        """Test get diagnostics endpoint."""
        assert TestAPIEndpoints.campaign_id, "No campaign_id from previous test"
        response = httpx.get(
            f"{BASE_URL}/api/diagnostics/{TestAPIEndpoints.campaign_id}",
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Get diagnostics failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Get diagnostics failed: {data.get('errors')}"
        assert "campaign_status" in data
        assert "n_results" in data
        print("✓ Get diagnostics passed")

    def test_11_get_suggestion_explanation(self):
        """Test get suggestion explanation endpoint."""
        assert TestAPIEndpoints.suggestion_id, "No suggestion_id from previous test"
        response = httpx.get(
            f"{BASE_URL}/api/suggestions/{TestAPIEndpoints.suggestion_id}/explanation",
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Get suggestion explanation failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Explanation failed: {data.get('errors')}"
        assert data["explanation"] is not None
        assert data["provenance"] is not None
        print("✓ Get suggestion explanation passed")

    def test_12_manage_campaign_lifecycle(self):
        """Test lifecycle management endpoint."""
        assert TestAPIEndpoints.campaign_id, "No campaign_id from previous test"
        response = httpx.post(
            f"{BASE_URL}/api/campaigns/{TestAPIEndpoints.campaign_id}/lifecycle",
            json={"action": "pause"},
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Lifecycle management failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Lifecycle action failed: {data.get('errors')}"
        assert data["status"] == "paused"
        print("✓ Manage campaign lifecycle passed")

    def test_13_batch_status(self):
        """Test batch campaign status endpoint."""
        assert TestAPIEndpoints.campaign_id, "No campaign_id from previous test"
        response = httpx.post(
            f"{BASE_URL}/api/campaigns/status/batch",
            json={
                "campaign_ids": [TestAPIEndpoints.campaign_id, "not-a-uuid"],
                "verbosity": "minimal",
            },
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Batch status failed: {response.text}"
        data = response.json()
        assert data["success"] is True
        assert TestAPIEndpoints.campaign_id in data["campaigns"]
        assert "not-a-uuid" in data["failed_ids"]
        print("✓ Batch status passed")

    def test_14_compare_campaigns(self):
        """Test compare campaigns endpoint."""
        compare_intake = {
            "intake": {
                "name": "API Compare Campaign",
                "description": "Comparison candidate for API verification",
                "parameters": [
                    {
                        "name": "temperature",
                        "type": "continuous",
                        "bounds": [0.0, 100.0],
                    },
                    {
                        "name": "pressure",
                        "type": "continuous",
                        "bounds": [1.0, 10.0],
                    },
                ],
                "objectives": [
                    {"name": "yield", "direction": "maximize"},
                    {"name": "cost", "direction": "minimize"},
                ],
            }
        }
        create_response = httpx.post(
            f"{BASE_URL}/api/campaigns",
            json=compare_intake,
            headers=HEADERS,
        )
        assert create_response.status_code == 201, (
            f"Create compare campaign failed: {create_response.text}"
        )
        TestAPIEndpoints.compare_campaign_id = create_response.json()["campaign_id"]

        generate_response = httpx.post(
            f"{BASE_URL}/api/suggestions/{TestAPIEndpoints.compare_campaign_id}/generate",
            headers=HEADERS,
        )
        assert generate_response.status_code == 201, (
            f"Generate compare suggestions failed: {generate_response.text}"
        )

        result_payload = {
            "results": [
                {
                    "parameter_values": {"temperature": 40.0, "pressure": 4.0},
                    "objective_values": {"yield": 0.82, "cost": 90.0},
                }
            ],
            "source": "api",
        }
        submit_response = httpx.post(
            f"{BASE_URL}/api/results/{TestAPIEndpoints.compare_campaign_id}",
            json=result_payload,
            headers=HEADERS,
        )
        assert submit_response.status_code == 201, (
            f"Submit compare results failed: {submit_response.text}"
        )

        response = httpx.post(
            f"{BASE_URL}/api/campaigns/compare",
            json={
                "campaign_ids": [
                    TestAPIEndpoints.campaign_id,
                    TestAPIEndpoints.compare_campaign_id,
                ],
                "verbosity": "standard",
            },
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Compare campaigns failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Compare failed: {data.get('errors')}"
        assert len(data["campaigns"]) == 2
        assert data["comparison"] is not None
        print("✓ Compare campaigns passed")

    def test_15_discover_transfer_candidates(self):
        """Test transfer candidate discovery endpoint."""
        source_intake = {
            "intake": {
                "name": "API Transfer Source",
                "description": "Source campaign for transfer discovery",
                "parameters": [
                    {
                        "name": "temperature",
                        "type": "continuous",
                        "bounds": [20.0, 100.0],
                    },
                    {
                        "name": "pressure",
                        "type": "continuous",
                        "bounds": [1.0, 10.0],
                    },
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            }
        }
        target_intake = {
            "intake": {
                "name": "API Transfer Target",
                "description": "Target campaign for transfer discovery",
                "parameters": [
                    {
                        "name": "temperature",
                        "type": "continuous",
                        "bounds": [30.0, 90.0],
                    },
                    {
                        "name": "pressure",
                        "type": "continuous",
                        "bounds": [2.0, 8.0],
                    },
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            }
        }
        source_response = httpx.post(
            f"{BASE_URL}/api/campaigns",
            json=source_intake,
            headers=HEADERS,
        )
        target_response = httpx.post(
            f"{BASE_URL}/api/campaigns",
            json=target_intake,
            headers=HEADERS,
        )
        assert source_response.status_code == 201, (
            f"Create transfer source failed: {source_response.text}"
        )
        assert target_response.status_code == 201, (
            f"Create transfer target failed: {target_response.text}"
        )

        TestAPIEndpoints.transfer_source_campaign_id = source_response.json()["campaign_id"]
        TestAPIEndpoints.transfer_target_campaign_id = target_response.json()["campaign_id"]

        generate_response = httpx.post(
            f"{BASE_URL}/api/suggestions/{TestAPIEndpoints.transfer_source_campaign_id}/generate",
            headers=HEADERS,
        )
        assert generate_response.status_code == 201, (
            f"Generate transfer suggestions failed: {generate_response.text}"
        )

        result_payload = {
            "results": [
                {
                    "parameter_values": {"temperature": 50.0, "pressure": 5.0},
                    "objective_values": {"yield": 0.8},
                },
                {
                    "parameter_values": {"temperature": 60.0, "pressure": 6.0},
                    "objective_values": {"yield": 0.85},
                },
                {
                    "parameter_values": {"temperature": 70.0, "pressure": 7.0},
                    "objective_values": {"yield": 0.9},
                },
            ],
            "source": "api",
        }
        submit_response = httpx.post(
            f"{BASE_URL}/api/results/{TestAPIEndpoints.transfer_source_campaign_id}",
            json=result_payload,
            headers=HEADERS,
        )
        assert submit_response.status_code == 201, (
            f"Submit transfer results failed: {submit_response.text}"
        )

        response = httpx.post(
            f"{BASE_URL}/api/campaigns/{TestAPIEndpoints.transfer_target_campaign_id}/transfer-candidates",
            json={
                "similarity_threshold": 0.3,
                "max_candidates": 5,
                "verbosity": "standard",
            },
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Discover transfer candidates failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Transfer candidate discovery failed: {data.get('errors')}"
        assert data["target_campaign"]["name"] == "API Transfer Target"
        assert len(data["candidates"]) >= 1
        print("✓ Discover transfer candidates passed")


if __name__ == "__main__":
    # Run tests manually
    test = TestAPIEndpoints()
    tests = [
        test.test_01_health_check,
        test.test_02_create_campaign,
        test.test_03_get_campaign_spec,
        test.test_04_get_campaign,
        test.test_05_list_campaigns,
        test.test_06_generate_suggestions,
        test.test_07_get_suggestions,
        test.test_08_submit_results,
        test.test_09_get_results,
        test.test_10_get_diagnostics,
        test.test_11_get_suggestion_explanation,
        test.test_12_manage_campaign_lifecycle,
        test.test_13_batch_status,
        test.test_14_compare_campaigns,
        test.test_15_discover_transfer_candidates,
    ]

    passed = 0
    failed = 0
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
