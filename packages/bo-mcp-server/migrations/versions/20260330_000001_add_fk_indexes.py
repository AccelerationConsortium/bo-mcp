"""Add indexes on foreign key columns for query performance.

Step 5 item 2.3: list_by_campaign queries scan full tables without indexes.
Adds indexes on campaigns.spec_id, campaigns.owner_id, suggestions.campaign_id,
results.campaign_id, and results.suggestion_id.

Note: events.campaign_id already has an index from migration 002.

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
    """Add indexes on foreign key columns."""
    op.create_index("ix_campaigns_spec_id", "campaigns", ["spec_id"])
    op.create_index("ix_campaigns_owner_id", "campaigns", ["owner_id"])
    op.create_index("ix_suggestions_campaign_id", "suggestions", ["campaign_id"])
    op.create_index("ix_results_campaign_id", "results", ["campaign_id"])
    op.create_index("ix_results_suggestion_id", "results", ["suggestion_id"])


def downgrade() -> None:
    """Remove foreign key indexes."""
    op.drop_index("ix_results_suggestion_id", "results")
    op.drop_index("ix_results_campaign_id", "results")
    op.drop_index("ix_suggestions_campaign_id", "suggestions")
    op.drop_index("ix_campaigns_owner_id", "campaigns")
    op.drop_index("ix_campaigns_spec_id", "campaigns")
