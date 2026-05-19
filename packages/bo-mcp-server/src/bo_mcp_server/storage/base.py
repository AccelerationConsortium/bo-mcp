"""Abstract repository interfaces."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

T = TypeVar("T")


class Repository(ABC, Generic[T]):
    """Abstract base repository interface.

    Defines the core CRUD operations that all repositories must implement.
    Subclasses should add domain-specific list methods as needed.
    """

    @abstractmethod
    async def get(self, entity_id: UUID) -> T | None:
        """Get entity by ID.

        Args:
            entity_id: Entity UUID

        Returns:
            Entity if found, None otherwise
        """
        ...

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity (create or update).

        Args:
            entity: Entity to save

        Returns:
            Saved entity (may have updated fields like timestamps)
        """
        ...

    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID.

        Args:
            entity_id: Entity UUID to delete

        Returns:
            True if entity was deleted, False if not found
        """
        ...

    @abstractmethod
    async def list_all(self) -> list[T]:
        """List all entities.

        Returns:
            List of all entities
        """
        ...


def str_id(uuid: UUID) -> str:
    """Convert a UUID to its 36-char string representation.

    Centralises the ``str(uuid)`` conversion used throughout the
    repository layer so it can later be replaced with a SQLAlchemy
    TypeDecorator if needed.
    """
    return str(uuid)


class ConcurrentModificationError(Exception):
    """Raised when optimistic locking detects a conflict."""

    def __init__(self, entity_type: str, entity_id: UUID, expected_version: int) -> None:
        """Record the conflicting entity type, id and expected optimistic-lock version."""
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_version = expected_version
        super().__init__(
            f"{entity_type} {entity_id} was modified. Expected version {expected_version}."
        )
