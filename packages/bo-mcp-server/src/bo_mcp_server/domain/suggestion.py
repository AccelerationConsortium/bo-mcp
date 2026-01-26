"""Suggestion entity."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from bo_mcp_server.domain.utils import utcnow


class SuggestionStatus(str, Enum):
    """Status of a suggestion."""

    PENDING = "pending"  # Generated, awaiting execution
    ACCEPTED = "accepted"  # User accepted for execution
    REJECTED = "rejected"  # User rejected
    COMPLETED = "completed"  # Result submitted
    EXPIRED = "expired"  # Superseded by newer batch


class SuggestionProvenance(BaseModel):
    """Provenance information for a suggestion."""

    iteration: int
    batch_index: int  # Position within batch
    acquisition_value: float | None = None  # Expected improvement
    model_uncertainty: float | None = None  # Prediction uncertainty
    generation_method: str = "bo"  # "bo", "initial_design", "manual"

    # Enhanced provenance for transparency (per DESIGN_REVIEW.md Section 6)
    acquisition_function: str | None = None  # e.g., "qLogNEHVI"
    model_type: str | None = None  # e.g., "ModelListGP"
    random_seed: int | None = None  # For reproducibility
    model_version: int | None = None  # Model snapshot version
    confidence_level: str | None = None  # "high", "medium", "low"
    explanation: str | None = None  # Human-readable reason for this suggestion


class Suggestion(BaseModel):
    """Suggestion entity representing a recommended experiment."""

    id: UUID = Field(default_factory=uuid4)
    campaign_id: UUID
    parameter_values: dict[str, Any]  # Parameter name -> value
    status: SuggestionStatus = SuggestionStatus.PENDING
    provenance: SuggestionProvenance
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def with_status(self, status: SuggestionStatus) -> "Suggestion":
        """Create new suggestion with updated status."""
        return self.model_copy(
            update={
                "status": status,
                "updated_at": utcnow(),
            }
        )

    @property
    def is_actionable(self) -> bool:
        """Check if suggestion can be acted upon."""
        return self.status in (SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED)

    def format_parameters(self) -> str:
        """Format parameters for display."""
        parts = []
        for name, value in self.parameter_values.items():
            if isinstance(value, float):
                parts.append(f"{name}={value:.4g}")
            else:
                parts.append(f"{name}={value}")
        return ", ".join(parts)
