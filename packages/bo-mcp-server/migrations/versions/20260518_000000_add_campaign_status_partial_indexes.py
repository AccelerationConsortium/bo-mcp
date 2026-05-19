"""Add per-status partial indexes on ``campaigns.status``.

Background
==========

The initial schema declared a plain BTree on ``campaigns.status``. With
five status values (``CREATED``, ``RUNNING``, ``PAUSED``, ``COMPLETED``,
``FAILED``) every query has to scan all values even though the dominant
traffic patterns target a small subset:

* The active-campaign poller (``list_filtered(status=RUNNING)``,
  dashboards) lives almost exclusively on ``CREATED`` + ``RUNNING``.
* Reporting / archive queries lives on the terminal subset
  ``COMPLETED`` + ``FAILED``.

A first iteration of this migration used two compound partial indexes
keyed on the IN-sets (``status IN ('CREATED','RUNNING')`` etc.). That
matched the audit's literal prescription but did **not** serve the
application's actual query shape — the repository emits single-status
equality filters (see
``CampaignRepository.list_filtered`` /
``CampaignRepository.list_keyset`` in
``packages/bo-mcp-server/src/bo_mcp_server/storage/repositories.py``):

.. code-block:: python

    query = query.where(CampaignModel.status == status)

SQLite's planner did not prove that ``status = 'RUNNING'`` subsumes
``status IN ('CREATED', 'RUNNING')``, so the IN-keyed partial index was
unused and the equality query fell back to a full table scan. (We
confirmed this empirically with ``EXPLAIN QUERY PLAN`` on a populated
schema.) The audit's intent — partition the working set so active and
terminal queries do not share index pages — required the partial
predicates to align with the equality query shape, not the conceptual
"active set" predicate.

This migration therefore creates one partial index per hot status:

* ``ix_campaigns_status_created_active`` → ``status = 'CREATED' AND
  deleted_at IS NULL``
* ``ix_campaigns_status_running_active`` → ``status = 'RUNNING' AND
  deleted_at IS NULL``
* ``ix_campaigns_status_completed_active`` → ``status = 'COMPLETED'
  AND deleted_at IS NULL``
* ``ix_campaigns_status_failed_active`` → ``status = 'FAILED' AND
  deleted_at IS NULL``

Each row only matches one of the four predicates, so steady-state
insert cost is one index write (plus four cheap predicate evaluations
to find which one). The legacy ``ix_campaigns_status`` is retained so
the ``PAUSED``-only operator query (rare but real) still has index
support and so existing query plans are unaffected for that subset.

References:
==========

* PostgreSQL "Partial Indexes":
  https://www.postgresql.org/docs/16/indexes-partial.html
* SQLite "Partial Indexes": https://www.sqlite.org/partialindex.html

Revision ID: 015_campaigns_status_partials
Revises: 014_results_unique_active
Create Date: 2026-05-18 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "015_campaigns_status_partials"
down_revision: str = "014_results_unique_active"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Per-status partial predicates. The status values are the *stored*
# representation of the enum — SQLAlchemy's ``Enum(CampaignStatus)``
# stores the enum *name* (uppercase) rather than the lowercase
# ``value``. A mismatch here would silently leave the index unusable
# (Postgres) or empty (SQLite); the value is pinned in
# ``test_campaigns_status_partial_indexes.py``.
#
# ``deleted_at IS NULL`` is part of every predicate because soft-deleted
# rows must not occupy slots in the hot-path index.
_HOT_STATUSES: tuple[tuple[str, str], ...] = (
    ("ix_campaigns_status_created_active", "CREATED"),
    ("ix_campaigns_status_running_active", "RUNNING"),
    ("ix_campaigns_status_completed_active", "COMPLETED"),
    ("ix_campaigns_status_failed_active", "FAILED"),
)


def _predicate_for(status: str) -> str:
    """Return the partial-index ``WHERE`` clause for a stored status name."""
    return f"status = '{status}' AND deleted_at IS NULL"


def upgrade() -> None:
    """Create one partial index per hot status."""
    for index_name, status in _HOT_STATUSES:
        predicate = _predicate_for(status)
        op.create_index(
            index_name,
            "campaigns",
            ["status"],
            sqlite_where=sa.text(predicate),
            postgresql_where=sa.text(predicate),
        )


def downgrade() -> None:
    """Drop the per-status partial indexes; ``ix_campaigns_status`` remains."""
    for index_name, _status in reversed(_HOT_STATUSES):
        op.drop_index(index_name, table_name="campaigns")
