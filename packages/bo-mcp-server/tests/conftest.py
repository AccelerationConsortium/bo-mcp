"""Pytest configuration and fixtures.

This module configures test isolation for the MCP server tests.
Each test gets a fresh in-memory SQLite database to ensure complete isolation.

Key design decisions:
- Database engine singleton is reset between tests to ensure fresh state
- Environment variables are set BEFORE any imports from bo_mcp_server
- USE_ALEMBIC=false ensures direct table creation (faster for SQLite tests)
"""

import os

# Set test database URL BEFORE importing storage module
# This ensures the engine is created with SQLite for testing
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
# Disable Alembic for SQLite tests (use direct Base.metadata.create_all)
os.environ["USE_ALEMBIC"] = "false"

import hashlib
from uuid import uuid4

import pytest
import pytest_asyncio

from bo_mcp_server.domain import (
    CampaignSpec,
    InputParameter,
    Objective,
    ParameterType,
    User,
)
from bo_mcp_server.storage import init_database


@pytest.fixture
def sample_continuous_param() -> InputParameter:
    """Create a sample continuous parameter."""
    return InputParameter(
        name="temperature",
        type=ParameterType.CONTINUOUS,
        bounds=(20.0, 100.0),
        description="Temperature in Celsius",
    )


@pytest.fixture
def sample_discrete_param() -> InputParameter:
    """Create a sample discrete parameter."""
    return InputParameter(
        name="pressure",
        type=ParameterType.DISCRETE,
        bounds=(1.0, 10.0),
        description="Pressure in bar",
    )


@pytest.fixture
def sample_categorical_param() -> InputParameter:
    """Create a sample categorical parameter."""
    return InputParameter(
        name="catalyst",
        type=ParameterType.CATEGORICAL,
        categories=["Pt", "Pd", "Rh"],
        description="Catalyst type",
    )


@pytest.fixture
def sample_objective_minimize() -> Objective:
    """Create a sample minimization objective."""
    return Objective(
        name="cost",
        direction="minimize",
        unit="USD",
    )


@pytest.fixture
def sample_objective_maximize() -> Objective:
    """Create a sample maximization objective."""
    return Objective(
        name="yield",
        direction="maximize",
        unit="%",
    )


@pytest.fixture
def sample_campaign_spec(
    sample_continuous_param: InputParameter,
    sample_discrete_param: InputParameter,
    sample_objective_minimize: Objective,
    sample_objective_maximize: Objective,
) -> CampaignSpec:
    """Create a sample campaign specification."""
    return CampaignSpec(
        name="Test Campaign",
        description="A test optimization campaign",
        parameters=[sample_continuous_param, sample_discrete_param],
        objectives=[sample_objective_minimize, sample_objective_maximize],
        batch_size=2,
    )


@pytest.fixture
def sample_user() -> User:
    """Create a sample user."""
    api_key = "test-api-key-12345"
    return User(
        id=uuid4(),
        name="Test User",
        email="test@example.com",
        api_key_hash=hashlib.sha256(api_key.encode()).hexdigest(),
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
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from bo_mcp_server.storage import database

    # Dispose existing engine if present to release connections
    if database.engine is not None:
        await database.engine.dispose()

    # Reset module-level singletons to force fresh database creation
    # This ensures each test gets a completely fresh in-memory database
    database.engine = database._create_engine_with_options()
    database.async_session_factory = async_sessionmaker(
        database.engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    await init_database()

    yield

    # Cleanup: dispose engine to release connections
    if database.engine is not None:
        await database.engine.dispose()
