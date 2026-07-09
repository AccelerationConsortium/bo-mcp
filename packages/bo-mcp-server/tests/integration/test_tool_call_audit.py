"""Tool-boundary audit trail.

Every tool dispatch through the wrapped ``ToolManager.call_tool``
(the single chokepoint the FastMCP transport uses) must persist a
``TOOL_CALL`` event so the ``events://{campaign_id}`` resource can
answer "what happened to this campaign?" for campaigns manipulated
via create / generate / lifecycle tools — not just the one operation
that writes ``Event`` rows directly.

The failure model is pinned end-to-end as well: with
``AUDIT_FAILURES_FATAL`` unset the tool response is returned
untouched even when persistence breaks (availability wins), while
flipping the flag converts the failure into a ``DATABASE_ERROR``
envelope so compliance deployments never silently lose audit rows.

Reference: MCP tools are expected to be observable server-side; the
MCP spec leaves auditing to implementations, so the contract pinned
here is this server's own ``bo_mcp_server.audit`` failure model.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from sqlalchemy.exc import SQLAlchemyError

from bo_mcp_server import audit as audit_module
from bo_mcp_server.domain.event import EventType
from bo_mcp_server.storage import EventRepository, get_session
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from tests.factories import seed_owner


async def _call(tool: str, arguments: dict[str, Any]) -> Any:
    """Dispatch a tool call exactly the way production serving does.

    ``FastMCP.call_tool`` requests ``convert_result=True``, so the raw
    response dict is reduced to ``(content_blocks, structured_dict)``
    before it reaches the transport — the audit hook must therefore run
    on the raw result *before* that conversion, which is exactly what
    these tests pin by not bypassing the conversion step. Successful
    structured responses are unwrapped back to the dict; synthesized
    boundary envelopes (e.g. the fatal-audit ``DATABASE_ERROR``) come
    back as raw dicts and pass through unchanged.
    """
    from bo_mcp_server.server import create_mcp_server

    mcp = create_mcp_server()
    result = await mcp.call_tool(tool, arguments)
    if isinstance(result, tuple):
        _content_blocks, structured = result
        return structured
    return result


async def _create_running_campaign(owner_id: str, name: str) -> str:
    """Create a campaign and advance it to RUNNING via one generation."""
    created = await create_campaign(
        {
            "name": name,
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "batch_size": 1,
        },
        owner_id,
    )
    assert created["success"] is True, created
    campaign_id = created["campaign_id"]
    generated = await generate_suggestions(campaign_id)
    assert generated["success"] is True, generated
    return campaign_id


async def _events_for(campaign_id: str):
    async with get_session() as session:
        return await EventRepository(session).list_by_campaign(UUID(campaign_id))


@pytest.mark.usefixtures("setup_database")
class TestToolCallAudit:
    @pytest.mark.asyncio
    async def test_create_campaign_event_attributed_to_minted_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The creation event lands under the campaign the call minted.

        ``bo_create_campaign`` takes no ``campaign_id`` argument — the
        id exists only in the result — so attribution must fall back
        to the response. Without that, ``events://{new_campaign_id}``
        would omit the campaign's first and most important mutation.
        """
        monkeypatch.setenv("DEV_AUTH", "1")
        monkeypatch.setenv("API_ENV", "development")

        result = await _call(
            "bo_create_campaign",
            {
                "intake_data": {
                    "name": "Audit Trail Create",
                    "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
                    "objectives": [{"name": "y", "direction": "minimize"}],
                },
            },
        )
        assert result["success"] is True, result
        campaign_id = result["campaign_id"]

        events = await _events_for(campaign_id)
        create_events = [e for e in events if e.tool_name == "bo_create_campaign"]
        assert len(create_events) == 1
        event = create_events[0]
        assert event.event_type == EventType.TOOL_CALL
        assert event.output_summary["success"] is True

    @pytest.mark.asyncio
    async def test_boundary_dispatch_writes_tool_call_event(self) -> None:
        """A lifecycle tool driven through ``call_tool`` leaves an audit row."""
        campaign_id = await _create_running_campaign(await seed_owner(), "Audit Trail Pause")

        result = await _call("bo_pause_campaign", {"campaign_id": campaign_id})
        assert result["success"] is True, result

        events = await _events_for(campaign_id)
        pause_events = [e for e in events if e.tool_name == "bo_pause_campaign"]
        assert len(pause_events) == 1
        event = pause_events[0]
        assert event.event_type == EventType.TOOL_CALL
        assert event.input_summary["campaign_id"] == campaign_id
        assert event.output_summary["success"] is True

    @pytest.mark.asyncio
    async def test_failure_envelope_is_audited_with_error_code(self) -> None:
        """Operation-level rejections are recorded with their error code.

        Pausing a freshly-created (non-RUNNING) campaign yields the
        ``INVALID_STATE_TRANSITION`` envelope; the audit row must
        capture ``success=False`` plus the code so the trail explains
        failures, not only successes.
        """
        owner_id = await seed_owner()
        created = await create_campaign(
            {
                "name": "Audit Trail Invalid Pause",
                "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
            owner_id,
        )
        campaign_id = created["campaign_id"]

        result = await _call("bo_pause_campaign", {"campaign_id": campaign_id})
        assert result["success"] is False, result

        events = await _events_for(campaign_id)
        pause_events = [e for e in events if e.tool_name == "bo_pause_campaign"]
        assert len(pause_events) == 1
        assert pause_events[0].output_summary["success"] is False
        assert pause_events[0].output_summary["error_code"] == result["error"]["code"]

    @pytest.mark.asyncio
    async def test_audit_failure_fatal_mode_returns_database_error_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``AUDIT_FAILURES_FATAL=true`` converts a broken audit sink into an envelope.

        The boundary — not the tool body — owns the conversion: the
        raised :class:`AuditPersistenceError` becomes a
        ``DATABASE_ERROR`` envelope returned in place of the tool
        result, so a compliance deployment cannot mutate state
        while silently skipping the audit row.
        """
        campaign_id = await _create_running_campaign(await seed_owner(), "Audit Fatal Mode")

        class _BrokenRepo:
            def __init__(self, _session: Any) -> None: ...

            async def save(self, _event: Any) -> None:
                msg = "audit sink down"
                raise SQLAlchemyError(msg)

        monkeypatch.setenv("AUDIT_FAILURES_FATAL", "true")
        monkeypatch.setattr(audit_module, "EventRepository", _BrokenRepo)

        result = await _call("bo_pause_campaign", {"campaign_id": campaign_id})
        assert result["success"] is False, result
        assert result["error"]["code"] == "E103"  # DATABASE_ERROR
        assert result["error"]["details"]["audit_failure"] is True

    @pytest.mark.asyncio
    async def test_audit_failure_nonfatal_mode_returns_tool_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default mode: a broken audit sink never fails the tool call."""
        campaign_id = await _create_running_campaign(await seed_owner(), "Audit Nonfatal Mode")

        class _BrokenRepo:
            def __init__(self, _session: Any) -> None: ...

            async def save(self, _event: Any) -> None:
                msg = "audit sink down"
                raise SQLAlchemyError(msg)

        monkeypatch.setenv("AUDIT_FAILURES_FATAL", "false")
        monkeypatch.setattr(audit_module, "EventRepository", _BrokenRepo)

        result = await _call("bo_pause_campaign", {"campaign_id": campaign_id})
        assert result["success"] is True, result
