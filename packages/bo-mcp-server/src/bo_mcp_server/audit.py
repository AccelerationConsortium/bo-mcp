"""Audit logging for MCP tool calls.

Records every tool invocation with compact input/output summaries.
Decoupled from any specific LLM — logs what was called, not why.
"""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from bo_mcp_server.domain.event import Event, EventType
from bo_mcp_server.storage import EventRepository, get_session

logger = logging.getLogger(__name__)


async def log_tool_call(
    tool_name: str,
    input_summary: dict[str, Any],
    output_summary: dict[str, Any],
    campaign_id: str | None = None,
    actor_id: str | None = None,
) -> None:
    """Log an MCP tool invocation as an audit event.

    Audit logging must never break the tool call: persistence failures
    (``SQLAlchemyError``) and invalid input payloads (``ValueError``,
    ``RuntimeError``) are caught and logged with ``exc_info=True`` so the
    traceback survives in operator logs while the parent tool call keeps
    running. Truly unexpected exception types are allowed to surface so
    programming bugs are not silently buried.

    Trace-id enrichment lives at the storage layer
    (:meth:`EventRepository.save`) so every event-emitting path — this
    helper, plus operations like ``update_suggestion_status`` that
    write ``Event`` rows directly — picks it up uniformly.

    Args:
        tool_name: Name of the MCP tool (e.g., "bo_create_campaign")
        input_summary: Compact summary of input arguments (not the full payload)
        output_summary: Compact summary of output (success/failure, key metrics)
        campaign_id: Associated campaign ID, if applicable
        actor_id: Identity of the caller, if known
    """
    try:
        event = Event(
            campaign_id=UUID(campaign_id) if campaign_id else None,
            event_type=EventType.TOOL_CALL,
            tool_name=tool_name,
            input_summary=input_summary,
            output_summary=output_summary,
            actor_id=actor_id,
        )
        async with get_session() as session:
            repo = EventRepository(session)
            await repo.save(event)
    except (SQLAlchemyError, ValueError, RuntimeError):
        logger.warning("Failed to log audit event for %s", tool_name, exc_info=True)
