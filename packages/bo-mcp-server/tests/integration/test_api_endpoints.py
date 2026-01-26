"""Integration tests for all API endpoints against running Docker containers."""

import httpx

BASE_URL = "http://localhost:8000"
API_KEY = "dev-api-key-12345"
HEADERS = {"X-API-Key": API_KEY}


class TestAPIEndpoints:
    """Test all API endpoints in sequence to verify full workflow."""

    campaign_id: str = ""
    spec_id: str = ""
    suggestion_id: str = ""
    result_id: str = ""

    def test_01_health_check(self):
        """Test health check endpoint."""
        response = httpx.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✓ Health check passed")

    def test_02_validate_intake(self):
        """Test intake validation endpoint."""
        intake_data = {
            "name": "Test Campaign",
            "description": "API test campaign",
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
        response = httpx.post(
            f"{BASE_URL}/api/campaigns/validate",
            json=intake_data,
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Validate intake failed: {response.text}"
        data = response.json()
        assert data["valid"] is True, f"Validation errors: {data.get('errors')}"
        print("✓ Validate intake passed")

    def test_03_create_campaign(self):
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
        assert response.status_code == 200, f"Create campaign failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Campaign creation failed: {data.get('errors')}"
        assert data["campaign_id"] is not None
        TestAPIEndpoints.campaign_id = data["campaign_id"]
        TestAPIEndpoints.spec_id = data["spec_id"]
        print(f"✓ Create campaign passed (id={TestAPIEndpoints.campaign_id[:8]}...)")

    def test_04_get_campaign_spec(self):
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

    def test_05_get_campaign(self):
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

    def test_06_list_campaigns(self):
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

    def test_07_generate_suggestions(self):
        """Test generate suggestions endpoint."""
        assert TestAPIEndpoints.campaign_id, "No campaign_id from previous test"
        response = httpx.post(
            f"{BASE_URL}/api/suggestions/{TestAPIEndpoints.campaign_id}/generate",
            headers=HEADERS,
        )
        assert response.status_code == 200, f"Generate suggestions failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Generation failed: {data.get('errors')}"
        assert len(data["suggestions"]) > 0
        TestAPIEndpoints.suggestion_id = data["suggestions"][0]["id"]
        print(f"✓ Generate suggestions passed (count={len(data['suggestions'])})")

    def test_08_get_suggestions(self):
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

    def test_09_submit_results(self):
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
        assert response.status_code == 200, f"Submit results failed: {response.text}"
        data = response.json()
        assert data["success"] is True, f"Result submission failed: {data.get('errors')}"
        assert len(data["result_ids"]) > 0
        TestAPIEndpoints.result_id = data["result_ids"][0]
        print(f"✓ Submit results passed (count={len(data['result_ids'])})")

    def test_10_get_results(self):
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

    def test_11_get_diagnostics(self):
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


if __name__ == "__main__":
    # Run tests manually
    test = TestAPIEndpoints()
    tests = [
        test.test_01_health_check,
        test.test_02_validate_intake,
        test.test_03_create_campaign,
        test.test_04_get_campaign_spec,
        test.test_05_get_campaign,
        test.test_06_list_campaigns,
        test.test_07_generate_suggestions,
        test.test_08_get_suggestions,
        test.test_09_submit_results,
        test.test_10_get_results,
        test.test_11_get_diagnostics,
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
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
