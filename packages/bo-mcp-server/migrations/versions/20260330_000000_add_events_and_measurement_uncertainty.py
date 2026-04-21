"""Add events table and measurement_uncertainty to results.

Step 3 of the implementation plan:
- events: Audit log for MCP tool invocations (5.1)
- measurement_uncertainty_json: Per-objective noise estimates on results (5.2)
- predicted_objectives/predicted_std are stored in provenance_json (no schema change needed) (5.3)

Revision ID: 002_events_and_uncertainty
Revises: 001_initial
Create Date: 2026-03-30 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "002_events_and_uncertainty"
down_revision: str = "001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add events table and measurement_uncertainty column."""
    # Events table for audit logging
    op.create_table(
        "events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "campaign_id",
            sa.String(36),
            sa.ForeignKey("campaigns.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "event_type",
            sa.Enum("TOOL_CALL", "LIFECYCLE", "ERROR", name="eventtype"),
            default="TOOL_CALL",
            nullable=False,
        ),
        sa.Column("tool_name", sa.String(255), nullable=False),
        sa.Column("input_summary_json", sa.Text(), default="{}", nullable=False),
        sa.Column("output_summary_json", sa.Text(), default="{}", nullable=False),
        sa.Column("actor_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_events_campaign_id", "events", ["campaign_id"])
    op.create_index("ix_events_tool_name", "events", ["tool_name"])
    op.create_index("ix_events_created_at", "events", ["created_at"])

    # Add measurement_uncertainty to results
    op.add_column(
        "results",
        sa.Column("measurement_uncertainty_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Remove events table and measurement_uncertainty column."""
    op.drop_column("results", "measurement_uncertainty_json")
    op.drop_table("events")
    op.execute("DROP TYPE IF EXISTS eventtype")
