"""REST parity for ``force`` on result submission.

The MCP ``bo_submit_results`` tool exposes ``force=True`` to bypass the
exact-duplicate-coordinate check -- the documented path for submitting
an intentional replicate. The REST ``ResultBatchCreate`` schema
previously dropped the field, so a REST client could not submit a
replicate the optimizer had asked for; rejecting the suggestion instead
only retires that suggestion record without excluding the coordinates
from future generation, causing a repeated generate-and-reject loop.

These tests assert that REST submission rejects an exact duplicate by
default and accepts it once ``force=True`` is set, matching the MCP
tool's semantics.
"""

import pytest

from bo_mcp_server.tools.create_campaign import create_campaign


async def _create_campaign(owner_id: str) -> str:
    intake = {
        "name": "REST Force Parity",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    return created["campaign_id"]


def _result_row(x: float, y: float) -> dict:
    return {"parameter_values": {"x": x}, "objective_values": {"y": y}}


@pytest.mark.asyncio
async def test_exact_duplicate_rejected_without_force(
    api_client, auth_headers, persisted_user
) -> None:
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)
    body = {"results": [_result_row(0.4, 1.0)], "source": "api"}

    first = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=auth_headers)
    second = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=auth_headers)

    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    assert second.json()["success"] is False


@pytest.mark.asyncio
async def test_exact_duplicate_accepted_with_force(
    api_client, auth_headers, persisted_user
) -> None:
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)
    body = {"results": [_result_row(0.4, 1.0)], "source": "api"}
    forced_body = {"results": [_result_row(0.4, 1.0)], "source": "api", "force": True}

    first = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=auth_headers)
    second = await api_client.post(
        f"/api/results/{campaign_id}", json=forced_body, headers=auth_headers
    )

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert second.json()["success"] is True
    assert first.json()["result_ids"] != second.json()["result_ids"]
