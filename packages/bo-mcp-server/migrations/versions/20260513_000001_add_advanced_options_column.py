"""Add ``advanced_options_json`` column to ``campaign_specs``.

Adds a nullable text column that carries the JSON-encoded subset of
:class:`CampaignSpec` advanced fields that previously had no storage:
``acquisition_method``, ``use_input_warping``, ``turbo_config``,
``outcome_constraints``, ``use_cost_aware``, ``fidelity_parameter``,
``transfer_learning``, ``saasbo_config``.

A single JSON blob keeps the schema stable as new advanced fields are
added (the values are consumed in-process, not queried in SQL).
Existing rows backfill to NULL so previously-created campaigns keep
their defaults.

Revision ID: 009_advanced_options
Revises: 008_backend_options
Create Date: 2026-05-13 00:00:01.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "009_advanced_options"
down_revision: str = "008_backend_options"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable ``advanced_options_json`` text column."""
    op.add_column(
        "campaign_specs",
        sa.Column("advanced_options_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Drop the ``advanced_options_json`` column."""
    op.drop_column("campaign_specs", "advanced_options_json")
