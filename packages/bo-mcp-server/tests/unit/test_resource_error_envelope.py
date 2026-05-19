"""Resource handlers raise typed errors instead of returning text envelopes (TODO 8.43).

Background: MCP resources previously returned bare ``"Error: ..."``
strings while MCP tools returned ``{"success": false, "error": {...}}``.
Agents had to maintain two parsers; mixed-format errors are a known
failure mode for LLM agents. An earlier pass (TODO 1.9) unified the
shape by returning the JSON envelope as the resource body — but
clients that route on the MCP protocol's success/error flag still saw
the read as successful and missed the failure.

8.43 fixes that gap: resource handlers now raise
:class:`ResourceOperationError`, which FastMCP re-raises as a
``ResourceError`` (a protocol-level failure). The structured envelope
still rides inside ``str(exc)`` so clients that read the exception
message can keep using the same recovery logic.

Reference: MCP resource error semantics
https://modelcontextprotocol.io/specification/2025-06-18/server/resources#errors
distinguishes successful reads from failure responses; raising lets
FastMCP map our resource failures onto the latter.
"""

from __future__ import annotations

import json

import pytest

from bo_mcp_server.errors import ResourceOperationError
from bo_mcp_server.resources.campaign_resource import get_campaign
from bo_mcp_server.resources.events_resource import get_campaign_events
from bo_mcp_server.resources.suggestion_resource import get_suggestion, get_suggestions

pytestmark = pytest.mark.usefixtures("setup_database")


def _envelope_from_error(exc: ResourceOperationError) -> dict:
    """Round-trip the embedded envelope back through JSON for assertions."""
    return json.loads(str(exc))


@pytest.mark.asyncio
async def test_get_campaign_invalid_uuid_raises_envelope() -> None:
    """A non-UUID input raises ``INVALID_CAMPAIGN_ID`` (E001)."""
    with pytest.raises(ResourceOperationError) as exc_info:
        await get_campaign("not-a-uuid")
    payload = _envelope_from_error(exc_info.value)

    assert payload["success"] is False
    assert payload["error"]["code"] == "E001"
    assert "campaign_id" in payload["error"]["details"]
    # ``errors`` array retained for backward-compatibility with the
    # original tool envelope shape.
    assert payload["errors"] == [payload["error"]["message"]]


@pytest.mark.asyncio
async def test_get_campaign_unknown_id_raises_envelope() -> None:
    """An unknown campaign id raises ``CAMPAIGN_NOT_FOUND`` (E002), not text."""
    with pytest.raises(ResourceOperationError) as exc_info:
        await get_campaign("00000000-0000-0000-0000-000000000000")
    payload = _envelope_from_error(exc_info.value)

    assert payload["success"] is False
    assert payload["error"]["code"] == "E002"
    assert payload["error"]["recovery_action"]  # non-empty


@pytest.mark.asyncio
async def test_get_suggestions_invalid_uuid_raises_envelope() -> None:
    with pytest.raises(ResourceOperationError) as exc_info:
        await get_suggestions("not-a-uuid")
    payload = _envelope_from_error(exc_info.value)
    assert payload["error"]["code"] == "E001"


@pytest.mark.asyncio
async def test_get_suggestion_invalid_uuid_raises_envelope() -> None:
    with pytest.raises(ResourceOperationError) as exc_info:
        await get_suggestion("not-a-uuid")
    payload = _envelope_from_error(exc_info.value)
    assert payload["error"]["code"] == "E005"


@pytest.mark.asyncio
async def test_events_invalid_uuid_raises_envelope() -> None:
    with pytest.raises(ResourceOperationError) as exc_info:
        await get_campaign_events("not-a-uuid")
    payload = _envelope_from_error(exc_info.value)
    assert payload["error"]["code"] == "E001"


@pytest.mark.asyncio
async def test_mcp_read_resource_surfaces_clean_envelope_for_invalid_uuid() -> None:
    """Real wire path: ``mcp.read_resource`` surfaces a clean ``McpError``.

    Pre-fix the read went through ``ResourceTemplate.create_resource`` →
    ``ResourceManager.get_resource`` and both layers wrapped the
    handler's exception in ``ValueError("Error creating resource from
    template: …")`` — so the JSON-RPC client saw the envelope behind
    two prefix wraps. The 8.43 follow-up patches the FastMCP read
    path to detect ``ResourceOperationError`` on the cause chain and
    raise :class:`McpError` with the clean envelope in
    ``error.message`` / ``error.data`` plus a semantically-correct
    JSON-RPC code (``INVALID_PARAMS`` for caller-supplied validation
    failures).
    """
    from mcp import types
    from mcp.shared.exceptions import McpError

    from bo_mcp_server.server import create_mcp_server

    mcp = create_mcp_server()
    with pytest.raises(McpError) as exc_info:
        await mcp.read_resource("campaign://not-a-uuid")
    err = exc_info.value.error
    assert err.code == types.INVALID_PARAMS
    # The message must NOT carry the FastMCP template wrap prefix.
    assert "Error creating resource from template" not in err.message
    # The structured envelope rides on ``data`` so clients route on it
    # without re-parsing JSON.
    assert err.data is not None
    assert err.data["error"]["code"] == "E001"
    assert err.data["error"]["details"]["campaign_id"] == "not-a-uuid"


@pytest.mark.asyncio
async def test_mcp_read_resource_surfaces_clean_envelope_for_unknown_id() -> None:
    """The same protection covers the well-formed-but-unknown-id branch.

    Sibling of the invalid-UUID test: the FastMCP wrap chain double-
    prefixes for *any* exception raised inside the handler. The fix
    must cover well-formed UUIDs whose campaign was never created
    (``CAMPAIGN_NOT_FOUND`` / E002) as well.
    """
    from mcp import types
    from mcp.shared.exceptions import McpError

    from bo_mcp_server.server import create_mcp_server

    mcp = create_mcp_server()
    with pytest.raises(McpError) as exc_info:
        await mcp.read_resource("campaign://00000000-0000-0000-0000-000000000000")
    err = exc_info.value.error
    assert err.code == types.INVALID_PARAMS
    assert "Error creating resource from template" not in err.message
    assert err.data is not None
    assert err.data["error"]["code"] == "E002"
