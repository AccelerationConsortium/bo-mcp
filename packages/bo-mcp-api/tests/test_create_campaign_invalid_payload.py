"""Malformed advanced REST payloads must return HTTP 422.

The REST ``IntakeData`` schema keeps advanced spec fields
(``turbo_config``, ``saasbo_config``, ``fidelity_parameter``,
``transfer_learning``, ``outcome_constraints``) as untyped dicts so the
REST schema stays decoupled from the bo-engine/bo-mcp-server domain
models. Strict shape validation happens in the route when the payload
is re-validated through ``CampaignIntakeInput``. Pre-fix, a malformed
inner field (e.g. a non-numeric ``turbo_config.initial_length``) raised
a raw Pydantic ``ValidationError`` from inside the route handler, which
FastAPI surfaces as a 500. Clients now see a proper 422 with a
FastAPI-style ``detail`` envelope identifying the offending field.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_malformed_turbo_config_returns_422(api_client, auth_headers, persisted_user):
    """Non-numeric ``turbo_config.initial_length`` → 422 not 500."""
    payload = {
        "intake": {
            "name": "Malformed turbo_config",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "turbo_config": {"initial_length": "not-a-number"},
        }
    }
    response = await api_client.post("/api/campaigns", json=payload, headers=auth_headers)

    assert response.status_code == 422, response.text
    body = response.json()
    assert "detail" in body
    # The offending field path is preserved in the error envelope so a
    # client can point the user at exactly which input was wrong.
    paths = [tuple(err["loc"]) for err in body["detail"]]
    assert any("turbo_config" in path and "initial_length" in path for path in paths), paths


@pytest.mark.asyncio
async def test_malformed_saasbo_config_returns_422(api_client, auth_headers, persisted_user):
    """Wrong type for a nested SAASBO field → 422 not 500."""
    payload = {
        "intake": {
            "name": "Malformed saasbo_config",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "saasbo_config": {"warmup_steps": "not-an-int"},
        }
    }
    response = await api_client.post("/api/campaigns", json=payload, headers=auth_headers)
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_malformed_outcome_constraint_returns_422(api_client, auth_headers, persisted_user):
    """Missing required key inside an outcome constraint → 422 not 500."""
    payload = {
        "intake": {
            "name": "Malformed outcome_constraint",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "outcome_constraints": [{"threshold": 0.5}],  # missing objective_name
        }
    }
    response = await api_client.post("/api/campaigns", json=payload, headers=auth_headers)
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_malformed_fidelity_parameter_returns_422(api_client, auth_headers, persisted_user):
    """Invalid bounds inside fidelity_parameter → 422 not 500."""
    payload = {
        "intake": {
            "name": "Malformed fidelity_parameter",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "fidelity_parameter": {
                "name": "fidelity",
                "bounds": [1.0, 0.1],  # inverted bounds — validator rejects
                "target": 1.0,
            },
        }
    }
    response = await api_client.post("/api/campaigns", json=payload, headers=auth_headers)
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_unknown_field_returns_422(api_client, auth_headers, persisted_user):
    """An unknown top-level intake key fails at the REST schema (``extra=forbid``)."""
    payload = {
        "intake": {
            "name": "Unknown field",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "totally_not_a_field": True,
        }
    }
    response = await api_client.post("/api/campaigns", json=payload, headers=auth_headers)
    assert response.status_code == 422, response.text
