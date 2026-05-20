"""Soft-delete columns, suggestion-provenance snapshot, RESTRICT cascades.

This migration hardens the data-integrity story around campaign / suggestion /
result cleanup:

* ``deleted_at`` is added to ``campaigns``, ``suggestions``, ``results``
  and ``events`` so cleanup paths can switch from hard ``DELETE`` to a
  timestamp flip. Repository read helpers filter ``deleted_at IS NULL``
  by default; admin / forensics queries opt in explicitly.

* ``results.suggestion_snapshot_json`` carries the originating
  suggestion's parameter values and provenance at submission time. The
  result-to-suggestion FK is ``ON DELETE SET NULL``, so even if the
  suggestion is later (soft- or hard-) removed, the result still
  reconstructs the provenance it was generated against.

* The previously cascading FKs ``suggestions.campaign_id`` →
  ``campaigns.id``, ``results.campaign_id`` → ``campaigns.id`` and
  ``events.campaign_id`` → ``campaigns.id`` are tightened from
  ``ON DELETE CASCADE`` to ``ON DELETE RESTRICT``. Any future cleanup
  path that forgets to soft-delete the children first now fails fast
  with an integrity error instead of silently wiping history.

The downgrade restores the previous ``CASCADE`` semantics for parity
with the original schema; the new columns are dropped.

Revision ID: 013_soft_delete
Revises: 012_user_api_key_unique
Create Date: 2026-05-17 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "013_soft_delete"
down_revision: str = "012_user_api_key_unique"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Centralized so the ``upgrade`` / ``downgrade`` pair cannot drift.
# Order is fixed by the dependency chain: child tables first so the
# parent's FKs can be re-bound without referential constraint races.
_CASCADE_TO_RESTRICT_FKS: tuple[tuple[str, str, str, str, str], ...] = (
    # (table, fk_name, local_column, referred_table, referred_column)
    ("suggestions", "suggestions_campaign_id_fkey", "campaign_id", "campaigns", "id"),
    ("results", "results_campaign_id_fkey", "campaign_id", "campaigns", "id"),
    ("events", "events_campaign_id_fkey", "campaign_id", "campaigns", "id"),
)


def _is_postgres() -> bool:
    """Return True when the active Alembic bind is PostgreSQL.

    SQLite cannot ``ALTER TABLE`` foreign keys in place; the on-disk
    schema is recreated from the ORM metadata for SQLite test runs
    (``Base.metadata.create_all``), so this migration is a no-op for
    that backend's FK changes. The ``deleted_at`` / snapshot columns
    are still added unconditionally because ``ADD COLUMN`` works on
    both backends.
    """
    bind = op.get_bind()
    return bind.dialect.name == "postgresql"


def _swap_fk(
    table: str,
    fk_name: str,
    local_column: str,
    referred_table: str,
    referred_column: str,
    *,
    new_ondelete: str,
) -> None:
    """Drop and recreate an FK with a different ``ON DELETE`` clause.

    Only invoked under PostgreSQL — SQLite recreates from metadata.
    """
    op.drop_constraint(fk_name, table, type_="foreignkey")
    op.create_foreign_key(
        fk_name,
        table,
        referred_table,
        [local_column],
        [referred_column],
        ondelete=new_ondelete,
    )


def upgrade() -> None:
    """Add soft-delete columns, snapshot blob, and tighten cascade FKs."""
    # ------------------------------------------------------------------
    # 1. Soft-delete columns on every history-bearing table.
    # ------------------------------------------------------------------
    for table in ("campaigns", "suggestions", "results", "events"):
        op.add_column(
            table,
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        )
        # Partial index on the active subset keeps the common
        # ``WHERE deleted_at IS NULL`` filter cheap as soft-deleted
        # history accumulates. Postgres + SQLite both honour the
        # ``WHERE`` clause; we mirror the syntax used by
        # ``ix_results_suggestion_id_unique``.
        op.create_index(
            f"ix_{table}_active",
            table,
            ["id"],
            sqlite_where=sa.text("deleted_at IS NULL"),
            postgresql_where=sa.text("deleted_at IS NULL"),
        )

    # ------------------------------------------------------------------
    # 2. Result provenance snapshot.
    # ------------------------------------------------------------------
    op.add_column(
        "results",
        sa.Column("suggestion_snapshot_json", sa.Text(), nullable=True),
    )

    # ------------------------------------------------------------------
    # 3. Tighten CASCADE FKs to RESTRICT. PostgreSQL only — SQLite
    #    rebuilds from ORM metadata in tests.
    # ------------------------------------------------------------------
    if _is_postgres():
        for table, fk_name, local, referred_table, referred_column in _CASCADE_TO_RESTRICT_FKS:
            _swap_fk(
                table,
                fk_name,
                local,
                referred_table,
                referred_column,
                new_ondelete="RESTRICT",
            )


def downgrade() -> None:
    """Restore the previous CASCADE FKs and drop the new columns."""
    if _is_postgres():
        for table, fk_name, local, referred_table, referred_column in _CASCADE_TO_RESTRICT_FKS:
            _swap_fk(
                table,
                fk_name,
                local,
                referred_table,
                referred_column,
                new_ondelete="CASCADE",
            )

    op.drop_column("results", "suggestion_snapshot_json")

    for table in ("campaigns", "suggestions", "results", "events"):
        op.drop_index(f"ix_{table}_active", table_name=table)
        op.drop_column(table, "deleted_at")
