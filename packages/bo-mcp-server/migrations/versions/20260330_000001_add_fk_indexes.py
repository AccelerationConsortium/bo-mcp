"""Add missing index on campaigns.spec_id for query performance.

Step 5 item 2.3: list_by_campaign queries scan full tables without indexes.
Adds the missing index on campaigns.spec_id. Other FK indexes
(campaigns.owner_id, suggestions.campaign_id, results.campaign_id,
results.suggestion_id) already exist from migration 001_initial.
events.campaign_id already has an index from migration 002.

Revision ID: 003_fk_indexes
Revises: 002_events_and_uncertainty
Create Date: 2026-03-30 00:00:01.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "003_fk_indexes"
down_revision: str = "002_events_and_uncertainty"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add missing index on campaigns.spec_id foreign key."""
    op.create_index("ix_campaigns_spec_id", "campaigns", ["spec_id"])


def downgrade() -> None:
    """Remove campaigns.spec_id index."""
    op.drop_index("ix_campaigns_spec_id", "campaigns")
