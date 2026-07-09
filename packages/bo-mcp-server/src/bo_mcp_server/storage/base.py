"""Repository helpers shared across the storage layer."""

from uuid import UUID


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
