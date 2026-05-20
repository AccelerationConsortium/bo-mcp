"""Add ``reservation_token`` column to ``idempotency_cache``.

Without per-reservation tokens, a slow operation whose reservation
expired (and was reclaimed by a concurrent retry) could later finalize
over the newer pending row — same key, same payload, indistinguishable
from the original holder. The token closes that race: the row carries
a UUID generated on insert, and ``finalize_reservation`` /
``drop_reservation`` only proceed when the token matches the one the
caller originally received. Stale owners observe an "affected rows
== 0" outcome and back off without touching the newer reservation.

The column is nullable so legacy rows (pre-migration) remain valid;
new inserts always carry a token. Lookups that need exact-match
ownership pass the token through the ``WHERE`` clause.

Revision ID: 011_idempotency_token
Revises: 010_idempotency_cache
Create Date: 2026-05-14 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "011_idempotency_token"
down_revision: str = "010_idempotency_cache"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable ``reservation_token`` text column."""
    op.add_column(
        "idempotency_cache",
        sa.Column("reservation_token", sa.String(36), nullable=True),
    )


def downgrade() -> None:
    """Drop the ``reservation_token`` column."""
    op.drop_column("idempotency_cache", "reservation_token")
