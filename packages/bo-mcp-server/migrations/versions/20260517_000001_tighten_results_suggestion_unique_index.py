"""Restrict the ``results.suggestion_id`` unique index to active rows.

Audit follow-up to TODO 8.11: the partial unique index introduced in
migration ``012_results_suggestion_id_unique`` keyed only on
``suggestion_id IS NOT NULL``. After the soft-delete column landed in
``013_soft_delete``, that predicate kept a *soft-deleted* result row
blocking a replacement INSERT against the same suggestion — application
reads treated the row as gone, but the database constraint still
rejected the replacement at the SQL layer. The new predicate adds
``deleted_at IS NULL`` so the unique slot is released when a result
is soft-deleted, matching the read-side semantics.

Scope: this migration fixes the *database constraint* only. The
application-level ``submit_results`` flow additionally marks the
underlying suggestion ``COMPLETED`` and later rejects non-actionable
suggestions, so a real client still cannot resubmit against the same
suggestion without an additional admin path that also restores the
suggestion's status. The unblock here is the admin / forensics
replacement story; the user-facing submission story is intentionally
unchanged.

Revision ID: 014_results_unique_active
Revises: 013_soft_delete
Create Date: 2026-05-17 00:00:01.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "014_results_unique_active"
down_revision: str = "013_soft_delete"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_INDEX_NAME = "ix_results_suggestion_id_unique"
_ACTIVE_PREDICATE = "suggestion_id IS NOT NULL AND deleted_at IS NULL"
_LEGACY_PREDICATE = "suggestion_id IS NOT NULL"

# Preflight for the downgrade: under the active-only predicate, a
# soft-deleted result and an active replacement can legitimately share
# the same ``suggestion_id``. Re-creating the legacy
# ``WHERE suggestion_id IS NOT NULL`` index over that state would fail
# with a duplicate-key error mid-migration, leaving the schema in an
# unknown state. The preflight surfaces the offending IDs up-front so
# the operator can reconcile them (typically by hard-deleting the
# soft-deleted rows that were superseded) before re-running the
# downgrade. Mirrors the duplicate-preflight pattern used by migration
# ``012_user_api_key_unique``.
_DUPLICATE_PREFLIGHT_SQL = sa.text(
    """
    SELECT suggestion_id, COUNT(*) AS n
    FROM results
    WHERE suggestion_id IS NOT NULL
    GROUP BY suggestion_id
    HAVING COUNT(*) > 1
    ORDER BY suggestion_id
    """
)


class DuplicateSuggestionIdError(RuntimeError):
    """Raised when downgrade would violate the legacy unique predicate.

    The legacy ``WHERE suggestion_id IS NOT NULL`` index treats every
    row referencing a suggestion as a uniqueness candidate, including
    soft-deleted rows. The active-only predicate the upgrade installs
    allows soft-deleted + active to coexist; the downgrade cannot,
    and the duplicate set must be reconciled manually first.
    """


def upgrade() -> None:
    """Recreate the unique index with the soft-delete-aware predicate."""
    op.drop_index(_INDEX_NAME, table_name="results")
    op.create_index(
        _INDEX_NAME,
        "results",
        ["suggestion_id"],
        unique=True,
        sqlite_where=sa.text(_ACTIVE_PREDICATE),
        postgresql_where=sa.text(_ACTIVE_PREDICATE),
    )


def downgrade() -> None:
    """Restore the legacy predicate after preflighting duplicates."""
    bind = op.get_bind()
    duplicates = bind.execute(_DUPLICATE_PREFLIGHT_SQL).fetchall()
    if duplicates:
        listing = ", ".join(f"{row.suggestion_id}={row.n}" for row in duplicates)
        msg = (
            "Cannot restore the legacy ix_results_suggestion_id_unique "
            f"predicate: {len(duplicates)} suggestion_id value(s) reference "
            f"more than one results row ({listing}). The upgrade allowed a "
            "soft-deleted result and an active replacement to share the "
            "same suggestion; hard-delete the superseded soft-deleted rows "
            "before re-running this downgrade."
        )
        raise DuplicateSuggestionIdError(msg)
    op.drop_index(_INDEX_NAME, table_name="results")
    op.create_index(
        _INDEX_NAME,
        "results",
        ["suggestion_id"],
        unique=True,
        sqlite_where=sa.text(_LEGACY_PREDICATE),
        postgresql_where=sa.text(_LEGACY_PREDICATE),
    )
