"""Add budget / convergence stopping columns to campaign_specs.

Persists ``max_observations`` and ``convergence_tolerance`` alongside the
existing ``max_iterations`` so :func:`evaluate_stopping_decision` sees them
on every spec round-trip.

Revision ID: 005_budget_columns
Revises: 004_backend_column
Create Date: 2026-05-12 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "005_budget_columns"
down_revision: str = "004_backend_column"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add max_observations and convergence_tolerance columns."""
    op.add_column(
        "campaign_specs",
        sa.Column("max_observations", sa.Integer(), nullable=True),
    )
    op.add_column(
        "campaign_specs",
        sa.Column("convergence_tolerance", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Remove the budget stopping columns."""
    op.drop_column("campaign_specs", "convergence_tolerance")
    op.drop_column("campaign_specs", "max_observations")
