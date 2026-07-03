"""Flip the campaign_specs.backend server default from botorch to baybe.

BayBE is now the default BO backend. Existing rows keep their explicit
backend value (the application always writes one); only the column
default for future inserts changes. BoTorch remains available as a
legacy fallback via an explicit ``backend="botorch"``.

Revision ID: 016_default_backend_baybe
Revises: 015_campaigns_status_partials
Create Date: 2026-07-03 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "016_default_backend_baybe"
down_revision: str = "015_campaigns_status_partials"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Change campaign_specs.backend server default to baybe."""
    with op.batch_alter_table("campaign_specs") as batch_op:
        batch_op.alter_column(
            "backend",
            existing_type=sa.String(50),
            existing_nullable=False,
            server_default="baybe",
        )


def downgrade() -> None:
    """Restore the botorch server default."""
    with op.batch_alter_table("campaign_specs") as batch_op:
        batch_op.alter_column(
            "backend",
            existing_type=sa.String(50),
            existing_nullable=False,
            server_default="botorch",
        )
