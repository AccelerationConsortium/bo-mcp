"""Add per-campaign acquisition-optimizer override columns.

Stores ``num_restarts`` and ``raw_samples`` overrides for the L-BFGS-B
multi-start. Both columns are nullable; when both are NULL the bo-engine
falls back to the dimension-adaptive defaults exposed by
``AcquisitionOptimizationConfig.resolve``.

Revision ID: 006_acq_opt_columns
Revises: 005_budget_columns
Create Date: 2026-05-12 00:00:01.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "006_acq_opt_columns"
down_revision: str = "005_budget_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ``acquisition_num_restarts`` / ``acquisition_raw_samples``."""
    op.add_column(
        "campaign_specs",
        sa.Column("acquisition_num_restarts", sa.Integer(), nullable=True),
    )
    op.add_column(
        "campaign_specs",
        sa.Column("acquisition_raw_samples", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    """Remove the acquisition-optimizer override columns."""
    op.drop_column("campaign_specs", "acquisition_raw_samples")
    op.drop_column("campaign_specs", "acquisition_num_restarts")
