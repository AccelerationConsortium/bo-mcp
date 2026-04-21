"""Audit event entity for MCP tool call logging."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from bo_mcp_server.domain.utils import utcnow


class EventType(StrEnum):
    """Type of audit event."""

    TOOL_CALL = "tool_call"
    LIFECYCLE = "lifecycle"
    ERROR = "error"


class Event(BaseModel):
    """Audit event recording an MCP tool invocation."""

    id: UUID = Field(default_factory=uuid4)
    campaign_id: UUID | None = None  # None for non-campaign-specific events
    event_type: EventType = EventType.TOOL_CALL
    tool_name: str
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    actor_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
