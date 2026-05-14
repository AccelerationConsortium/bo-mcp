"""Resource handlers return the structured error envelope (TODO 1.9).

Background: MCP resources previously returned bare ``"Error: ..."``
strings while MCP tools returned ``{"success": false, "error": {...}}``.
Agents had to maintain two parsers; mixed-format errors are a known
failure mode for LLM agents. This suite pins the unified shape so
``campaign://{id}``, ``suggestions://{id}``, ``suggestion://{id}``, and
``events://{id}`` all emit the same JSON envelope on failure.

The success path still returns Markdown text. Detection on the client
side is "does the response start with ``{``" — which is documented in
``render_resource_error``'s docstring and in the per-resource return
type comments.
"""

from __future__ import annotations

import json

import pytest

from bo_mcp_server.resources.campaign_resource import get_campaign
from bo_mcp_server.resources.events_resource import get_campaign_events
from bo_mcp_server.resources.suggestion_resource import get_suggestion, get_suggestions

pytestmark = pytest.mark.usefixtures("setup_database")


@pytest.mark.asyncio
async def test_get_campaign_invalid_uuid_returns_envelope() -> None:
    """A non-UUID input returns the standard ``INVALID_CAMPAIGN_ID`` envelope."""
    raw = await get_campaign("not-a-uuid")
    payload = json.loads(raw)

    assert payload["success"] is False
    assert payload["error"]["code"] == "E001"
    assert "campaign_id" in payload["error"]["details"]
    # ``errors`` array retained for backward-compatibility with the
    # original tool envelope shape.
    assert payload["errors"] == [payload["error"]["message"]]


@pytest.mark.asyncio
async def test_get_campaign_unknown_id_returns_envelope() -> None:
    """An unknown campaign id returns ``CAMPAIGN_NOT_FOUND`` (E002), not text."""
    raw = await get_campaign("00000000-0000-0000-0000-000000000000")
    payload = json.loads(raw)

    assert payload["success"] is False
    assert payload["error"]["code"] == "E002"
    assert payload["error"]["recovery_action"]  # non-empty


@pytest.mark.asyncio
async def test_get_suggestions_invalid_uuid_returns_envelope() -> None:
    raw = await get_suggestions("not-a-uuid")
    payload = json.loads(raw)
    assert payload["error"]["code"] == "E001"


@pytest.mark.asyncio
async def test_get_suggestion_invalid_uuid_returns_envelope() -> None:
    raw = await get_suggestion("not-a-uuid")
    payload = json.loads(raw)
    assert payload["error"]["code"] == "E005"


@pytest.mark.asyncio
async def test_events_invalid_uuid_returns_envelope() -> None:
    raw = await get_campaign_events("not-a-uuid")
    payload = json.loads(raw)
    assert payload["error"]["code"] == "E001"
