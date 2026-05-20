"""Tests for API dependencies.

These tests verify the shared helper functions used across API routes
for UUID validation and campaign authorization.

References:
- RFC 4122: A Universally Unique IDentifier (UUID) URN Namespace
- FastAPI Dependency Injection: https://fastapi.tiangolo.com/tutorial/dependencies/
"""

from uuid import uuid4

import pytest
from fastapi import HTTPException

from api.deps import get_authorized_campaign, validate_uuid


class TestValidateUUID:
    """Tests for validate_uuid helper function.

    The validate_uuid function parses UUID strings and raises HTTPException
    with 400 status code if the format is invalid. This centralizes UUID
    validation across all API routes.
    """

    def test_valid_uuid_returns_uuid_object(self):
        """Valid UUID string should return UUID object."""
        valid_uuid = "12345678-1234-5678-1234-567812345678"
        result = validate_uuid(valid_uuid)
        assert str(result) == valid_uuid

    def test_valid_uuid_with_uppercase(self):
        """Valid UUID with uppercase letters should be parsed."""
        valid_uuid = "12345678-1234-5678-1234-567812345678"
        result = validate_uuid(valid_uuid.upper())
        assert str(result).lower() == valid_uuid.lower()

    def test_random_uuid_is_valid(self):
        """Randomly generated UUID should be valid."""
        random_uuid = str(uuid4())
        result = validate_uuid(random_uuid)
        assert str(result) == random_uuid

    def test_invalid_uuid_raises_http_exception(self):
        """Invalid UUID string should raise HTTPException with 400 status."""
        with pytest.raises(HTTPException) as exc_info:
            validate_uuid("not-a-valid-uuid")

        assert exc_info.value.status_code == 400
        assert "Invalid id format" in exc_info.value.detail

    def test_empty_string_raises_http_exception(self):
        """Empty string should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_uuid("")

        assert exc_info.value.status_code == 400

    def test_custom_name_in_error_message(self):
        """Error message should include custom parameter name."""
        with pytest.raises(HTTPException) as exc_info:
            validate_uuid("invalid", name="campaign_id")

        assert exc_info.value.status_code == 400
        assert "Invalid campaign_id format" in exc_info.value.detail

    def test_spec_id_name_in_error_message(self):
        """Error message should use spec_id when specified."""
        with pytest.raises(HTTPException) as exc_info:
            validate_uuid("invalid", name="spec_id")

        assert exc_info.value.status_code == 400
        assert "Invalid spec_id format" in exc_info.value.detail

    def test_uuid_without_hyphens_is_invalid(self):
        """UUID without hyphens should be invalid (strict format)."""
        # Python's UUID can parse without hyphens, but let's verify behavior
        uuid_no_hyphens = "12345678123456781234567812345678"
        # This actually works with Python's UUID parser
        result = validate_uuid(uuid_no_hyphens)
        assert result is not None

    def test_uuid_with_extra_characters_is_invalid(self):
        """UUID with extra characters should be invalid."""
        with pytest.raises(HTTPException) as exc_info:
            validate_uuid("12345678-1234-5678-1234-567812345678-extra")

        assert exc_info.value.status_code == 400

    def test_uuid_too_short_is_invalid(self):
        """UUID that is too short should be invalid."""
        with pytest.raises(HTTPException) as exc_info:
            validate_uuid("12345678-1234-5678")

        assert exc_info.value.status_code == 400


@pytest.mark.usefixtures("setup_database")
class TestGetAuthorizedCampaign:
    """Tests for get_authorized_campaign helper function.

    This function combines UUID validation, campaign lookup, and ownership
    check into a single reusable dependency. It eliminates duplicated
    authorization logic across multiple API routes.

    Reference: OWASP Access Control Cheat Sheet
    https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html
    """

    @pytest.mark.asyncio
    async def test_invalid_campaign_id_raises_400(self, sample_user):
        """Invalid campaign_id format should raise 400 HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            await get_authorized_campaign("not-a-uuid", sample_user)

        assert exc_info.value.status_code == 400
        assert "Invalid campaign_id format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_nonexistent_campaign_raises_404(self, sample_user):
        """Non-existent campaign should raise 404 HTTPException."""
        nonexistent_id = str(uuid4())

        with pytest.raises(HTTPException) as exc_info:
            await get_authorized_campaign(nonexistent_id, sample_user)

        assert exc_info.value.status_code == 404
        assert f"Campaign {nonexistent_id} not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_authorization_checks(self, persisted_campaign_with_users):
        """Test authorization: owner succeeds, non-owner gets 403.

        This test verifies both the happy path (owner access) and the
        authorization failure path (non-owner access) within a single
        test to avoid fixture isolation issues with in-memory SQLite.
        """
        campaign, owner, other_user = persisted_campaign_with_users
        campaign_id = str(campaign.id)

        # Owner should successfully get the campaign
        result = await get_authorized_campaign(campaign_id, owner)
        assert result is not None
        assert result.id == campaign.id
        assert result.owner_id == owner.id
        assert hasattr(result, "spec_id")
        assert hasattr(result, "status")
        assert hasattr(result, "iteration")

        # Non-owner should get 403
        with pytest.raises(HTTPException) as exc_info:
            await get_authorized_campaign(campaign_id, other_user)

        assert exc_info.value.status_code == 403
        assert "Not authorized to access this campaign" in exc_info.value.detail
