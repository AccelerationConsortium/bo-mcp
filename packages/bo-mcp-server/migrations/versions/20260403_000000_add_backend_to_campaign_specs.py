"""Add backend column to campaign_specs for per-campaign backend selection.

Allows each campaign to specify which BO backend (botorch, baybe, etc.)
to use. Existing campaigns default to "botorch".

Revision ID: 004_backend_column
Revises: 003_fk_indexes
Create Date: 2026-04-03 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "004_backend_column"
down_revision: str = "003_fk_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add backend column to campaign_specs."""
    op.add_column(
        "campaign_specs",
        sa.Column("backend", sa.String(50), nullable=False, server_default="botorch"),
    )


def downgrade() -> None:
    """Remove backend column from campaign_specs."""
    op.drop_column("campaign_specs", "backend")
