"""Tests for Campaign domain model."""

from uuid import uuid4

from bo_mcp_server.domain import Campaign, CampaignStatus


class TestCampaign:
    """Tests for Campaign entity."""

    def test_campaign_creation(self):
        """Campaign can be created with default values."""
        campaign = Campaign(
            spec_id=uuid4(),
            owner_id=uuid4(),
        )
        assert campaign.status == CampaignStatus.CREATED
        assert campaign.version == 1
        assert campaign.iteration == 0

    def test_campaign_is_active(self):
        """is_active returns correct value based on status."""
        campaign = Campaign(
            spec_id=uuid4(),
            owner_id=uuid4(),
        )
        assert campaign.is_active is True

        completed = campaign.with_status(CampaignStatus.COMPLETED)
        assert completed.is_active is False

    def test_campaign_with_status(self):
        """with_status creates new campaign with updated status."""
        campaign = Campaign(
            spec_id=uuid4(),
            owner_id=uuid4(),
        )
        running = campaign.with_status(CampaignStatus.RUNNING)

        # Original unchanged
        assert campaign.status == CampaignStatus.CREATED
        assert campaign.version == 1

        # New version updated
        assert running.status == CampaignStatus.RUNNING
        assert running.version == 2

    def test_campaign_advance_iteration(self):
        """advance_iteration creates new campaign with incremented iteration."""
        campaign = Campaign(
            spec_id=uuid4(),
            owner_id=uuid4(),
        )
        advanced = campaign.advance_iteration()

        # Original unchanged
        assert campaign.iteration == 0
        assert campaign.version == 1

        # New version updated
        assert advanced.iteration == 1
        assert advanced.version == 2

    def test_campaign_completed_sets_timestamp(self):
        """Completing a campaign sets completed_at timestamp."""
        campaign = Campaign(
            spec_id=uuid4(),
            owner_id=uuid4(),
        )
        assert campaign.completed_at is None

        completed = campaign.with_status(CampaignStatus.COMPLETED)
        assert completed.completed_at is not None
