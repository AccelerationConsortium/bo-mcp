"""Add ``idempotency_cache`` table.

Persists a 24-hour cache of ``(tool, key) -> response`` so retries of a
state-mutating MCP tool with the same ``idempotency_key`` return the
prior response instead of re-executing the side effect (TODO 1.46).

The ``request_hash`` column stores a SHA256 of the canonical request
payload so a deliberately-repeated key with a different payload is
flagged as a conflict instead of silently masking a logic bug.

Revision ID: 010_idempotency_cache
Revises: 009_advanced_options
Create Date: 2026-05-13 00:00:02.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "010_idempotency_cache"
down_revision: str = "009_advanced_options"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ``idempotency_cache`` table."""
    op.create_table(
        "idempotency_cache",
        sa.Column("tool_name", sa.String(255), primary_key=True),
        sa.Column("idempotency_key", sa.String(255), primary_key=True),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("response_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_idempotency_cache_expires_at",
        "idempotency_cache",
        ["expires_at"],
    )


def downgrade() -> None:
    """Drop the ``idempotency_cache`` table."""
    op.drop_index("ix_idempotency_cache_expires_at", table_name="idempotency_cache")
    op.drop_table("idempotency_cache")
