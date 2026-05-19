"""Audit logging for MCP tool calls.

Records every tool invocation with compact input/output summaries.
Decoupled from any specific LLM — logs what was called, not why.

Failure model
-------------

Audit persistence runs in its own short-lived transaction (separate
from the tool's own writes) so a transient DB hiccup on the audit row
does not roll back already-committed business state. Two failure modes
must stay distinguishable in operator dashboards:

* The default (``AUDIT_FAILURES_FATAL=false``): the parent tool keeps
  running, but every failure bumps :data:`metrics.AUDIT_FAILURES` so a
  dashboard alarm fires within seconds. This preserves availability —
  the tool would otherwise fail on cosmetic audit-row issues — at the
  cost of an audit gap.
* Compliance deployments flip ``AUDIT_FAILURES_FATAL=true``: the
  audit error is re-raised so the surrounding tool surface converts it
  into an error envelope. An audit gap is no longer silently
  acceptable. The counter still increments so SOC dashboards see a
  uniform signal regardless of the mode.

Only ``SQLAlchemyError`` / ``ValueError`` / ``RuntimeError`` are caught
explicitly: storage / payload-shape failures are the documented
exceptions the audit row is allowed to swallow. Programming bugs
(``AttributeError`` from a misconfigured event, etc.) still propagate
unchanged so they are not silently buried under the audit mode flag.
"""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from bo_mcp_server.domain.event import Event, EventType
from bo_mcp_server.metrics import record_audit_failure
from bo_mcp_server.settings import get_audit_failures_fatal
from bo_mcp_server.storage import EventRepository, get_session

logger = logging.getLogger(__name__)


class AuditPersistenceError(RuntimeError):
    """Raised when audit-event persistence fails under fatal-mode.

    Carries the original exception via :attr:`__cause__` so callers
    that surface the failure as a structured error envelope can include
    the underlying exception class for triage without leaking the raw
    message (which may include connection strings or query fragments).
    """


async def log_tool_call(
    tool_name: str,
    input_summary: dict[str, Any],
    output_summary: dict[str, Any],
    campaign_id: str | None = None,
    actor_id: str | None = None,
) -> None:
    """Log an MCP tool invocation as an audit event.

    See the module docstring for the failure-model contract: failures
    always bump :data:`metrics.AUDIT_FAILURES`; whether they also
    propagate is controlled by ``AUDIT_FAILURES_FATAL``.

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

    Raises:
        AuditPersistenceError: only when ``AUDIT_FAILURES_FATAL=1`` and
            the persistence path raised a known recoverable failure.
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
    except (SQLAlchemyError, ValueError, RuntimeError) as exc:
        record_audit_failure(tool_name)
        logger.warning(
            "Failed to log audit event for %s (%s); fatal=%s",
            tool_name,
            type(exc).__name__,
            get_audit_failures_fatal(),
            exc_info=True,
        )
        if get_audit_failures_fatal():
            msg = f"Audit event for {tool_name} failed to persist: {type(exc).__name__}"
            raise AuditPersistenceError(msg) from exc
