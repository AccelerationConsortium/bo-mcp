"""Cursor codec and helpers for keyset pagination.

Background: offset pagination is unsafe under concurrent inserts —
rows get skipped or duplicated across pages. This suite pins the
codec round-trip and the rejection contract for malformed cursors.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bo_mcp_server.pagination import (
    Cursor,
    CursorError,
    cursor_from_row,
    decode_cursor,
    encode_cursor,
    parse_optional_cursor,
)


def test_cursor_round_trip_preserves_tuple() -> None:
    """Encoding then decoding restores the original ``(created_at, id)`` pair."""
    when = datetime(2026, 5, 13, 12, 0, 0, tzinfo=UTC)
    original = Cursor(created_at=when, entity_id="abc-123")
    decoded = decode_cursor(encode_cursor(original))
    assert decoded == original


def test_cursor_is_urlsafe() -> None:
    """The cursor token must not contain characters that break URL/query usage."""
    when = datetime(2026, 5, 13, 12, 0, 0, tzinfo=UTC)
    token = cursor_from_row(when, "abc-123")
    # urlsafe base64 alphabet is A-Z, a-z, 0-9, '-', '_', '='.
    assert all(c.isalnum() or c in "-_=" for c in token)


def test_decode_rejects_malformed_base64() -> None:
    with pytest.raises(CursorError):
        decode_cursor("not!base64!")


def test_decode_rejects_missing_fields() -> None:
    """Token decodes to JSON but lacks required keys."""
    import base64

    payload = base64.urlsafe_b64encode(b'{"foo": "bar"}').decode("ascii")
    with pytest.raises(CursorError):
        decode_cursor(payload)


def test_decode_rejects_non_iso_created_at() -> None:
    import base64

    payload = base64.urlsafe_b64encode(b'{"created_at": "not-a-date", "id": "x"}').decode("ascii")
    with pytest.raises(CursorError):
        decode_cursor(payload)


def test_parse_optional_cursor_none_yields_empty_tuple() -> None:
    """``None`` cursor maps to ``(None, None)`` — caller stays in offset mode."""
    assert parse_optional_cursor(None) == (None, None)


def test_parse_optional_cursor_error_returns_message() -> None:
    """Malformed cursor returns the error string for the caller to surface."""
    result = parse_optional_cursor("garbage")
    assert isinstance(result, str)
    assert "cursor" in result.lower() or "json" in result.lower() or "base64" in result.lower()


def test_parse_optional_cursor_round_trip() -> None:
    when = datetime(2026, 5, 13, 12, 0, 0, tzinfo=UTC)
    token = cursor_from_row(when, "abc-123")
    parsed = parse_optional_cursor(token)
    assert parsed == (when, "abc-123")
