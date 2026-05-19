"""Add partial-unique index on ``results.suggestion_id``.

Guarantees at most one Result row can reference any given suggestion.
``NULL`` values (free-floating results) are excluded so manual/imported
rows can coexist without sharing a single global NULL slot. The
server-side phase-1 check rejects duplicates inside a single batch before
they reach the index; the index itself is the safety net against
cross-batch / cross-process duplicates.

A pre-existing bug in the submission path could have produced duplicate
``results.suggestion_id`` rows. We refuse to create the index silently in
that state -- instead, the upgrade aborts with the offending IDs so the
operator can decide how to clean up (typically: keep the earliest result
per suggestion and null-out / delete the rest, depending on policy).

Revision ID: 007_suggestion_unique
Revises: 006_acq_opt_columns
Create Date: 2026-05-12 00:00:02.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "007_suggestion_unique"
down_revision: str = "006_acq_opt_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


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
    """Raised when the upgrade detects pre-existing duplicate ``suggestion_id`` rows.

    The migration cannot create the partial-unique index while duplicates
    exist; the operator must reconcile them first. The exception message
    lists each offending ``suggestion_id`` with its row count so the
    operator can target their cleanup query precisely.
    """


def upgrade() -> None:
    """Create the partial-unique index after preflight cleanup check."""
    bind = op.get_bind()
    duplicates = bind.execute(_DUPLICATE_PREFLIGHT_SQL).fetchall()
    if duplicates:
        listing = ", ".join(f"{row.suggestion_id}={row.n}" for row in duplicates)
        msg = (
            "Cannot create ix_results_suggestion_id_unique: "
            f"{len(duplicates)} suggestion_id value(s) appear more than once "
            f"in `results` ({listing}). Resolve duplicates -- typically by "
            "keeping the earliest result per suggestion and null-ing or "
            "deleting the others -- then re-run this migration."
        )
        raise DuplicateSuggestionIdError(msg)
    op.create_index(
        "ix_results_suggestion_id_unique",
        "results",
        ["suggestion_id"],
        unique=True,
        sqlite_where=sa.text("suggestion_id IS NOT NULL"),
        postgresql_where=sa.text("suggestion_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Drop the partial-unique index."""
    op.drop_index("ix_results_suggestion_id_unique", table_name="results")
