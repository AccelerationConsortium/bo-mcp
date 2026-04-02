"""Pytest configuration and fixtures for bo-mcp-api tests."""

import os

# Set test database URL BEFORE importing storage module
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
# Disable Alembic for SQLite tests (use direct Base.metadata.create_all)
os.environ["USE_ALEMBIC"] = "false"

import hashlib
from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from bo_mcp_server.domain import (
    Bounds,
    Campaign,
    CampaignSpec,
    CampaignStatus,
    InputParameter,
    Objective,
    ParameterType,
    User,
)
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    UserRepository,
    close_database,
    get_session,
    init_database,
)
from httpx import ASGITransport, AsyncClient

from api.main import create_app


@pytest.fixture
def sample_user() -> User:
    """Create a sample user for testing."""
    api_key = "test-api-key-12345"
    return User(
        id=uuid4(),
        name="Test User",
        email="test@example.com",
        api_key_hash=hashlib.sha256(api_key.encode()).hexdigest(),
    )


@pytest.fixture
def another_user() -> User:
    """Create another user for testing authorization."""
    return User(
        id=uuid4(),
        name="Another User",
        email="another@example.com",
        api_key_hash=hashlib.sha256(b"another-api-key-67890").hexdigest(),
    )


@pytest.fixture
def sample_campaign_spec() -> CampaignSpec:
    """Create a sample campaign spec for testing."""
    return CampaignSpec(
        name="Test Campaign",
        description="A test campaign for unit tests",
        parameters=[
            InputParameter(
                name="temperature",
                type=ParameterType.CONTINUOUS,
                bounds=Bounds(lower=20.0, upper=100.0),
            ),
        ],
        objectives=[
            Objective(name="yield", direction="maximize"),
        ],
        batch_size=1,
    )


@pytest_asyncio.fixture
async def setup_database():
    """Initialize fresh in-memory database for each test.

    Ensures complete isolation between tests by resetting the
    database engine singleton. This is critical for test isolation
    when using in-memory SQLite databases.

    Reference: SQLAlchemy async engine lifecycle documentation
    https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
    """
    # Import here to avoid circular imports and to access module internals
    from bo_mcp_server.storage import database
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    # Dispose existing engine if present to release connections
    await close_database()

    # Reset lazy singletons to force fresh database creation
    # This ensures each test gets a completely fresh in-memory database
    database._engine = database._create_engine_with_options()
    database._session_factory = async_sessionmaker(
        database._engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    await init_database()

    yield

    # Cleanup: dispose engine to release connections
    await close_database()


@pytest_asyncio.fixture
async def persisted_user(setup_database, sample_user: User) -> User:
    """Create and persist a user in the database."""
    async with get_session() as session:
        user_repo = UserRepository(session)
        await user_repo.save(sample_user)
        await session.commit()
    return sample_user


@pytest_asyncio.fixture
async def persisted_another_user(setup_database, another_user: User) -> User:
    """Create and persist the secondary user in the database."""
    async with get_session() as session:
        user_repo = UserRepository(session)
        await user_repo.save(another_user)
        await session.commit()
    return another_user


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for the primary test user."""
    return {"X-API-Key": "test-api-key-12345"}


@pytest.fixture
def other_auth_headers() -> dict[str, str]:
    """Authentication headers for the secondary test user."""
    return {"X-API-Key": "another-api-key-67890"}


@pytest_asyncio.fixture
async def api_client(setup_database) -> AsyncGenerator[AsyncClient]:
    """Async HTTP client bound to the FastAPI app."""
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


@pytest_asyncio.fixture
async def persisted_campaign_with_users(
    setup_database,
    sample_campaign_spec: CampaignSpec,
) -> tuple[Campaign, User, User]:
    """Create and persist a campaign with owner and another user.

    Returns tuple of (campaign, owner_user, another_user).
    This ensures all entities are created in the same database context
    and avoids fixture dependency issues with in-memory SQLite.
    """
    spec_id = uuid4()
    campaign_id = uuid4()
    owner_id = uuid4()
    other_user_id = uuid4()

    owner = User(
        id=owner_id,
        name="Campaign Owner",
        email="owner@example.com",
        api_key_hash="owner_hash",
    )
    other_user = User(
        id=other_user_id,
        name="Other User",
        email="other@example.com",
        api_key_hash="other_hash",
    )

    async with get_session() as session:
        # Save users
        user_repo = UserRepository(session)
        await user_repo.save(owner)
        await user_repo.save(other_user)

        # Save spec
        spec_repo = CampaignSpecRepository(session)
        await spec_repo.save(sample_campaign_spec, spec_id)

        # Save campaign
        campaign = Campaign(
            id=campaign_id,
            spec_id=spec_id,
            owner_id=owner_id,
            status=CampaignStatus.CREATED,
            iteration=0,
        )
        campaign_repo = CampaignRepository(session)
        await campaign_repo.save(campaign)
        await session.commit()

    return campaign, owner, other_user
