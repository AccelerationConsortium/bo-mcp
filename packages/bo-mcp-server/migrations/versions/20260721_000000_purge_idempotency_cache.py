"""Purge the idempotency cache for the suggestion_id-only payload contract.

Suggestion payloads no longer carry the ``id`` key; ``suggestion_id``
is the only identity key and is required by the response contracts.
Cached generate responses persisted under the previous shape would
replay without ``suggestion_id`` and fail the new contract, so the
cache is emptied here. The table is a pure replay cache with a short
TTL — dropping rows only means a retried request re-executes instead
of replaying, and only for requests issued before this migration ran.

Revision ID: 017_purge_idempotency_cache
Revises: 016_default_backend_baybe
Create Date: 2026-07-21 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "017_purge_idempotency_cache"
down_revision: str = "016_default_backend_baybe"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Delete every cached idempotency entry."""
    op.execute("DELETE FROM idempotency_cache")


def downgrade() -> None:
    """Nothing to restore — the purged rows were a replay cache."""
