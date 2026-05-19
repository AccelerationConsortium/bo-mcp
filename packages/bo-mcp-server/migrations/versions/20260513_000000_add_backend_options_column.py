"""Add ``backend_options_json`` column to ``campaign_specs``.

Adds a nullable text column carrying the per-backend native-options
dictionary. The dictionary is JSON-encoded
because each entry is opaque to the neutral spec and can grow per
backend. Existing rows backfill to NULL so previously-created campaigns
keep their behavior.

Revision ID: 008_backend_options
Revises: 007_suggestion_unique
Create Date: 2026-05-13 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "008_backend_options"
down_revision: str = "007_suggestion_unique"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable ``backend_options_json`` text column."""
    op.add_column(
        "campaign_specs",
        sa.Column("backend_options_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Drop the ``backend_options_json`` column."""
    op.drop_column("campaign_specs", "backend_options_json")
