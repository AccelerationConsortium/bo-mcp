"""Result entity."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from bo_mcp_server.domain.utils import utcnow


class ResultSource(StrEnum):
    """Source of the result data."""

    GUI = "gui"  # Manual entry via web interface
    FILE_UPLOAD = "file_upload"  # CSV/Excel upload
    API = "api"  # Programmatic submission


class Result(BaseModel):
    """Result entity representing an experimental observation."""

    id: UUID = Field(default_factory=uuid4)
    campaign_id: UUID
    suggestion_id: UUID | None = None  # None for historical/manual data
    parameter_values: dict[str, Any]  # Parameter name -> value
    objective_values: dict[str, float]  # Objective name -> observed value
    source: ResultSource
    submitted_by: UUID  # User who submitted
    metadata: dict[str, Any] = Field(default_factory=dict)  # Extra info
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def is_from_suggestion(self) -> bool:
        """Check if result came from a suggestion."""
        return self.suggestion_id is not None

    def format_objectives(self) -> str:
        """Format objectives for display."""
        parts = []
        for name, value in self.objective_values.items():
            parts.append(f"{name}={value:.4g}")
        return ", ".join(parts)

    def format_parameters(self) -> str:
        """Format parameters for display."""
        parts = []
        for name, value in self.parameter_values.items():
            if isinstance(value, float):
                parts.append(f"{name}={value:.4g}")
            else:
                parts.append(f"{name}={value}")
        return ", ".join(parts)
