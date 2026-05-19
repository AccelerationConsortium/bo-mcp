"""Replace the non-unique ``ix_users_api_key_hash`` with a UNIQUE index.

The REST auth path resolves the caller with
``UserRepository.get_by_api_key_hash`` (``scalar_one_or_none()``).
Without a uniqueness constraint, accidentally provisioning two users
with the same hash would surface as a ``MultipleResultsFound`` 500 on
every request that user makes — not a hypothetical edge case once we
start handing out real keys.

The upgrade refuses to run when pre-existing duplicates are present;
the operator must reconcile them (typically by retiring all but one
account) before the unique index can be created. This mirrors the
preflight pattern used by the earlier
``ix_results_suggestion_id_unique`` migration.

The old non-unique ``ix_users_api_key_hash`` is dropped at the same
time. Keeping both would leave PostgreSQL deployments with two indexes
covering the same column — wasted writes and a permanent diff against
``alembic --autogenerate`` of the matching ORM declaration in
``UserModel.__table_args__``.

Revision ID: 012_user_api_key_unique
Revises: 011_idempotency_token
Create Date: 2026-05-16 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "012_user_api_key_unique"
down_revision: str = "011_idempotency_token"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_DUPLICATE_PREFLIGHT_SQL = sa.text(
    """
    SELECT api_key_hash, COUNT(*) AS n
    FROM users
    GROUP BY api_key_hash
    HAVING COUNT(*) > 1
    ORDER BY api_key_hash
    """
)


class DuplicateApiKeyHashError(RuntimeError):
    """Raised when pre-existing ``users.api_key_hash`` duplicates are found.

    The migration cannot create the unique index while duplicates exist;
    the operator must retire all but one account per hash first. The
    message lists each colliding hash with its row count so the operator
    can target the cleanup precisely.
    """


def upgrade() -> None:
    """Swap the legacy index for a UNIQUE one after a duplicate preflight."""
    bind = op.get_bind()
    duplicates = bind.execute(_DUPLICATE_PREFLIGHT_SQL).fetchall()
    if duplicates:
        listing = ", ".join(f"{row.api_key_hash}={row.n}" for row in duplicates)
        msg = (
            "Cannot create ix_users_api_key_hash_unique: "
            f"{len(duplicates)} api_key_hash value(s) appear more than once "
            f"in `users` ({listing}). Retire all but one account per hash, "
            "then re-run this migration."
        )
        raise DuplicateApiKeyHashError(msg)
    op.drop_index("ix_users_api_key_hash", table_name="users")
    op.create_index(
        "ix_users_api_key_hash_unique",
        "users",
        ["api_key_hash"],
        unique=True,
    )


def downgrade() -> None:
    """Restore the legacy non-unique index."""
    op.drop_index("ix_users_api_key_hash_unique", table_name="users")
    op.create_index("ix_users_api_key_hash", "users", ["api_key_hash"])
