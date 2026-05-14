"""Opaque cursor codec for keyset pagination.

Offset-based pagination is unsafe under concurrent inserts on a live
campaign: rows added between page reads shift the offset window so the
agent silently sees the same row twice or skips one entirely. The fix
is keyset (a.k.a. seek) pagination on a strictly-monotonic key — we
use ``(created_at, id)`` because both columns are already indexed and
``id`` is a UUID string, giving the deterministic tiebreaker we need.

The cursor we hand back is opaque on the wire: a base64-url-encoded
JSON blob containing the last row's ``(created_at, id)``. Encoding
keeps clients from poking at the internals (forwards-compatibility) and
keeps the response shape JSON-safe. There is no signature — the cursor
is just a position pointer, not a capability — and a malformed cursor
returns a structured validation error so the caller does not get an
empty page silently.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Cursor:
    """Decoded keyset position.

    Attributes:
        created_at: The ``created_at`` value of the last returned row.
        entity_id: The string-form id of the last returned row. UUIDs
            are compared lexicographically, which is enough for the
            tiebreaker as long as the column is stored as a fixed-length
            string (which the ORM does).
    """

    created_at: datetime
    entity_id: str


class CursorError(ValueError):
    """Raised when an inbound cursor cannot be decoded."""


def encode_cursor(cursor: Cursor) -> str:
    """Serialize a :class:`Cursor` to its on-the-wire form.

    Uses URL-safe base64 of the canonical JSON so the cursor can ride
    inside a query string without further encoding.
    """
    payload = {
        "created_at": cursor.created_at.isoformat(),
        "id": cursor.entity_id,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def decode_cursor(token: str) -> Cursor:
    """Parse a cursor token into the structured position.

    Raises:
        CursorError: If the token is malformed (bad base64, bad JSON,
            missing required keys, or an unparseable ``created_at``).
    """
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise CursorError(f"Malformed cursor token: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CursorError(f"Cursor token is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise CursorError("Cursor token must decode to a JSON object")

    if "created_at" not in payload or "id" not in payload:
        raise CursorError("Cursor token missing required keys 'created_at' and 'id'")

    try:
        created_at = datetime.fromisoformat(str(payload["created_at"]))
    except ValueError as exc:
        raise CursorError(f"Cursor 'created_at' is not ISO-8601: {exc}") from exc

    return Cursor(created_at=created_at, entity_id=str(payload["id"]))


def cursor_from_row(created_at: datetime, entity_id: str) -> str:
    """Build an encoded cursor pointing at this row."""
    return encode_cursor(Cursor(created_at=created_at, entity_id=entity_id))


def build_page_cursor(items: list[Any], created_at_attr: str, id_attr: str) -> str | None:
    """Return the cursor for the last item in ``items``, or ``None`` for an empty page.

    Centralized so the operation layer never has to remember the field
    names. ``created_at_attr`` / ``id_attr`` are passed as strings so the
    helper does not need a typed protocol for every entity shape.
    """
    if not items:
        return None
    last = items[-1]
    return cursor_from_row(getattr(last, created_at_attr), str(getattr(last, id_attr)))


def parse_optional_cursor(
    cursor: str | None,
) -> tuple[datetime | None, str | None] | str:
    """Decode ``cursor`` to ``(created_at, id)`` or return a human-readable error.

    Helper consumed by every list-operation that supports cursor
    pagination so the validation lives in one place. Returns ``(None,
    None)`` when the cursor is ``None`` (caller proceeds in
    offset-pagination mode), the decoded tuple on success, or a string
    describing the failure when the cursor is malformed.
    """
    if cursor is None:
        return None, None
    try:
        decoded = decode_cursor(cursor)
    except CursorError as exc:
        return str(exc)
    return decoded.created_at, decoded.entity_id
