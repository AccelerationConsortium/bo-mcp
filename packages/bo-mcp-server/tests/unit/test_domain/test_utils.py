"""Tests for domain utils module.

Reference: Testing timezone-aware datetime utilities based on Python datetime best practices.
See: https://docs.python.org/3/library/datetime.html#datetime.datetime.now
"""

from datetime import UTC, datetime

from bo_mcp_server.domain.utils import utcnow


class TestUtcnow:
    """Tests for utcnow utility function."""

    def test_returns_datetime(self) -> None:
        """Test that utcnow returns a datetime object."""
        result = utcnow()
        assert isinstance(result, datetime)

    def test_is_timezone_aware(self) -> None:
        """Test that returned datetime is timezone-aware."""
        result = utcnow()
        assert result.tzinfo is not None

    def test_is_utc_timezone(self) -> None:
        """Test that returned datetime has UTC timezone."""
        result = utcnow()
        assert result.tzinfo == UTC

    def test_is_close_to_now(self) -> None:
        """Test that returned time is close to current time."""
        before = datetime.now(UTC)
        result = utcnow()
        after = datetime.now(UTC)

        assert before <= result <= after

    def test_successive_calls_are_ordered(self) -> None:
        """Test that successive calls return increasing times."""
        first = utcnow()
        second = utcnow()

        assert first <= second

    def test_not_naive_datetime(self) -> None:
        """Test that the returned datetime is not naive (has tzinfo)."""
        result = utcnow()
        # A naive datetime would have tzinfo as None
        assert result.tzinfo is not None
        # Can convert to other timezones without error
        _ = result.astimezone(UTC)

    def test_microsecond_precision(self) -> None:
        """Test that datetime includes microsecond precision."""
        result = utcnow()
        # Microseconds should be available (though may be 0)
        assert hasattr(result, "microsecond")
        assert 0 <= result.microsecond < 1_000_000
