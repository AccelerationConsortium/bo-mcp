"""Tests for repository implementations.

Reference: Testing repository pattern implementations based on the Domain-Driven Design
repository pattern.
See: https://www.cosmicpython.com/book/chapter_02_repository.html
"""

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bo_mcp_server.domain import (
    CampaignStatus,
    Result,
    ResultSource,
    Suggestion,
    SuggestionProvenance,
    User,
)
from bo_mcp_server.storage.models import Base
from bo_mcp_server.storage.repositories import (
    ResultRepository,
    SuggestionRepository,
    UserRepository,
)

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def session() -> AsyncSession:
    """Create a test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with async_session() as session:
        yield session

    await engine.dispose()


class TestUserRepository:
    """Tests for UserRepository."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, session: AsyncSession) -> None:
        """Test saving and retrieving a user."""
        repo = UserRepository(session)
        user = User(
            name="Test User",
            email="test@example.com",
            api_key_hash="hash123",
        )

        saved = await repo.save(user)
        await session.commit()

        retrieved = await repo.get(saved.id)

        assert retrieved is not None
        assert retrieved.name == "Test User"
        assert retrieved.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_list_all(self, session: AsyncSession) -> None:
        """Test listing all users."""
        repo = UserRepository(session)

        # Create multiple users
        for i in range(3):
            user = User(
                name=f"User {i}",
                email=f"user{i}@example.com",
                api_key_hash=f"hash{i}",
            )
            await repo.save(user)
        await session.commit()

        users = await repo.list_all()

        assert len(users) == 3

    @pytest.mark.asyncio
    async def test_delete(self, session: AsyncSession) -> None:
        """Test deleting a user."""
        repo = UserRepository(session)
        user = User(
            name="Delete Me",
            email="delete@example.com",
            api_key_hash="hash",
        )

        saved = await repo.save(user)
        await session.commit()

        result = await repo.delete(saved.id)
        await session.commit()

        assert result is True
        assert await repo.get(saved.id) is None


class TestSuggestionRepository:
    """Tests for SuggestionRepository."""

    @pytest.fixture
    async def campaign_id(self, session: AsyncSession) -> uuid4:
        """Create a test campaign and return its ID."""
        # We need to create the necessary parent records
        from bo_mcp_server.storage.models import (
            CampaignModel,
            CampaignSpecModel,
            UserModel,
        )

        user = UserModel(
            id=str(uuid4()),
            name="Test",
            email="test@example.com",
            api_key_hash="hash",
        )
        session.add(user)

        spec = CampaignSpecModel(
            id=str(uuid4()),
            name="Test Spec",
            parameters_json='[{"name":"x","type":"continuous","bounds":[0,1]}]',
            objectives_json='[{"name":"y","direction":"minimize"}]',
        )
        session.add(spec)

        campaign_id = uuid4()
        campaign = CampaignModel(
            id=str(campaign_id),
            spec_id=spec.id,
            owner_id=user.id,
            status=CampaignStatus.RUNNING,
        )
        session.add(campaign)
        await session.commit()

        return campaign_id

    @pytest.mark.asyncio
    async def test_save_batch_empty_list(self, session: AsyncSession) -> None:
        """Test save_batch with empty list returns empty list."""
        repo = SuggestionRepository(session)

        result = await repo.save_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_save_batch_single_item(self, session: AsyncSession, campaign_id: uuid4) -> None:
        """Test save_batch with single suggestion."""
        repo = SuggestionRepository(session)
        suggestion = Suggestion(
            campaign_id=campaign_id,
            parameter_values={"x": 0.5},
            provenance=SuggestionProvenance(iteration=1, batch_index=0),
        )

        result = await repo.save_batch([suggestion])
        await session.commit()

        assert len(result) == 1
        assert result[0].parameter_values == {"x": 0.5}

        # Verify it was persisted
        retrieved = await repo.get(suggestion.id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_save_batch_multiple_items(
        self, session: AsyncSession, campaign_id: uuid4
    ) -> None:
        """Test save_batch with multiple suggestions is more efficient than individual saves."""
        repo = SuggestionRepository(session)
        suggestions = [
            Suggestion(
                campaign_id=campaign_id,
                parameter_values={"x": i / 10.0},
                provenance=SuggestionProvenance(iteration=1, batch_index=i),
            )
            for i in range(10)
        ]

        result = await repo.save_batch(suggestions)
        await session.commit()

        assert len(result) == 10

        # Verify all were persisted
        all_suggestions = await repo.list_by_campaign(campaign_id)
        assert len(all_suggestions) == 10

    @pytest.mark.asyncio
    async def test_list_all(self, session: AsyncSession, campaign_id: uuid4) -> None:
        """Test list_all returns all suggestions."""
        repo = SuggestionRepository(session)

        # Create suggestions
        for i in range(5):
            suggestion = Suggestion(
                campaign_id=campaign_id,
                parameter_values={"x": i / 10.0},
                provenance=SuggestionProvenance(iteration=1, batch_index=i),
            )
            await repo.save(suggestion)
        await session.commit()

        all_suggestions = await repo.list_all()

        assert len(all_suggestions) == 5


class TestResultRepository:
    """Tests for ResultRepository."""

    @pytest.fixture
    async def campaign_and_user(self, session: AsyncSession) -> tuple:
        """Create a test campaign and user."""
        from bo_mcp_server.storage.models import (
            CampaignModel,
            CampaignSpecModel,
            UserModel,
        )

        user_id = uuid4()
        user = UserModel(
            id=str(user_id),
            name="Test",
            email="test@example.com",
            api_key_hash="hash",
        )
        session.add(user)

        spec = CampaignSpecModel(
            id=str(uuid4()),
            name="Test Spec",
            parameters_json='[{"name":"x","type":"continuous","bounds":[0,1]}]',
            objectives_json='[{"name":"y","direction":"minimize"}]',
        )
        session.add(spec)

        campaign_id = uuid4()
        campaign = CampaignModel(
            id=str(campaign_id),
            spec_id=spec.id,
            owner_id=user.id,
            status=CampaignStatus.RUNNING,
        )
        session.add(campaign)
        await session.commit()

        return campaign_id, user_id

    @pytest.mark.asyncio
    async def test_save_batch_empty_list(self, session: AsyncSession) -> None:
        """Test save_batch with empty list returns empty list."""
        repo = ResultRepository(session)

        result = await repo.save_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_save_batch_single_item(
        self, session: AsyncSession, campaign_and_user: tuple
    ) -> None:
        """Test save_batch with single result."""
        campaign_id, user_id = campaign_and_user
        repo = ResultRepository(session)
        result_entity = Result(
            campaign_id=campaign_id,
            parameter_values={"x": 0.5},
            objective_values={"y": 1.0},
            source=ResultSource.API,
            submitted_by=user_id,
        )

        saved = await repo.save_batch([result_entity])
        await session.commit()

        assert len(saved) == 1
        assert saved[0].parameter_values == {"x": 0.5}

        # Verify it was persisted
        retrieved = await repo.get(result_entity.id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_save_batch_multiple_items(
        self, session: AsyncSession, campaign_and_user: tuple
    ) -> None:
        """Test save_batch with multiple results."""
        campaign_id, user_id = campaign_and_user
        repo = ResultRepository(session)
        results = [
            Result(
                campaign_id=campaign_id,
                parameter_values={"x": i / 10.0},
                objective_values={"y": float(i)},
                source=ResultSource.API,
                submitted_by=user_id,
            )
            for i in range(10)
        ]

        saved = await repo.save_batch(results)
        await session.commit()

        assert len(saved) == 10

        # Verify all were persisted
        all_results = await repo.list_by_campaign(campaign_id)
        assert len(all_results) == 10

    @pytest.mark.asyncio
    async def test_list_all(self, session: AsyncSession, campaign_and_user: tuple) -> None:
        """Test list_all returns all results."""
        campaign_id, user_id = campaign_and_user
        repo = ResultRepository(session)

        # Create results
        for i in range(5):
            result = Result(
                campaign_id=campaign_id,
                parameter_values={"x": i / 10.0},
                objective_values={"y": float(i)},
                source=ResultSource.API,
                submitted_by=user_id,
            )
            await repo.save(result)
        await session.commit()

        all_results = await repo.list_all()

        assert len(all_results) == 5
