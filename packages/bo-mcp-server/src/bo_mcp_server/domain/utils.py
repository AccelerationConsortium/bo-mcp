"""Shared utility functions for domain models."""

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Get current UTC time (timezone-aware).

    Returns:
        Current datetime with UTC timezone
    """
    return datetime.now(UTC)
