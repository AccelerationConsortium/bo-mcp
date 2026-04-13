"""Campaign entity."""

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from bo_mcp_server.domain.utils import utcnow

if TYPE_CHECKING:
    from bo_mcp_server.domain.campaign_spec import CampaignSpec


class CampaignStatus(StrEnum):
    """Campaign lifecycle status."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Campaign(BaseModel):
    """Campaign entity representing an optimization campaign.

    Includes version field for optimistic locking.
    """

    id: UUID = Field(default_factory=uuid4)
    spec_id: UUID  # Reference to CampaignSpec
    owner_id: UUID  # User who created the campaign
    status: CampaignStatus = CampaignStatus.CREATED
    version: int = Field(default=1, ge=1)  # For optimistic locking
    iteration: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    # Opaque backend state (e.g. TuRBO trust region, BayBE campaign JSON).
    # Stored in the DB column ``turbo_state_json`` for backward compatibility.
    backend_state: dict[str, Any] | None = None
    # v2.8: Hypervolume history for multi-objective convergence detection
    hypervolume_history: list[float] = Field(default_factory=list)

    # Cached spec for convenience (not persisted separately)
    _spec: "CampaignSpec | None" = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_active(self) -> bool:
        """Check if campaign is in an active state."""
        return self.status in (CampaignStatus.CREATED, CampaignStatus.RUNNING)

    @property
    def can_generate_suggestions(self) -> bool:
        """Check if new suggestions can be generated."""
        return self.status in (CampaignStatus.CREATED, CampaignStatus.RUNNING)

    @property
    def can_submit_results(self) -> bool:
        """Check if results can be submitted."""
        return self.status in (CampaignStatus.CREATED, CampaignStatus.RUNNING)

    def increment_version(self) -> "Campaign":
        """Create new campaign with incremented version."""
        return self.model_copy(
            update={
                "version": self.version + 1,
                "updated_at": utcnow(),
            }
        )

    def with_status(self, status: CampaignStatus) -> "Campaign":
        """Create new campaign with updated status."""
        updates: dict = {
            "status": status,
            "version": self.version + 1,
            "updated_at": utcnow(),
        }
        if status == CampaignStatus.COMPLETED:
            updates["completed_at"] = utcnow()
        return self.model_copy(update=updates)

    def advance_iteration(self) -> "Campaign":
        """Create new campaign with incremented iteration."""
        return self.model_copy(
            update={
                "iteration": self.iteration + 1,
                "version": self.version + 1,
                "updated_at": utcnow(),
            }
        )

    def with_backend_state(self, state: dict[str, Any] | None) -> "Campaign":
        """Create new campaign with updated backend state."""
        return self.model_copy(
            update={
                "backend_state": state,
                "version": self.version + 1,
                "updated_at": utcnow(),
            }
        )

    def with_hypervolume(self, hypervolume: float) -> "Campaign":
        """Create new campaign with appended hypervolume value.

        Used for tracking multi-objective convergence over time.
        """
        new_history = self.hypervolume_history + [hypervolume]
        return self.model_copy(
            update={
                "hypervolume_history": new_history,
                "version": self.version + 1,
                "updated_at": utcnow(),
            }
        )
