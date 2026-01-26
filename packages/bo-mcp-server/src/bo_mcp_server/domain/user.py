"""User entity."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from bo_mcp_server.domain.utils import utcnow


class User(BaseModel):
    """User entity for authentication and attribution."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=1)
    api_key_hash: str  # Hashed API key for authentication
    is_active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    last_active_at: datetime | None = None

    def record_activity(self) -> "User":
        """Create new user with updated last_active_at."""
        return self.model_copy(update={"last_active_at": utcnow()})

    def deactivate(self) -> "User":
        """Create new user with is_active=False."""
        return self.model_copy(update={"is_active": False})

    def activate(self) -> "User":
        """Create new user with is_active=True."""
        return self.model_copy(update={"is_active": True})
