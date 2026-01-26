"""PostgreSQL integration test fixtures using testcontainers.

These fixtures provide real PostgreSQL instances for integration testing,
ensuring database behavior matches production environments.

Reference: https://testcontainers-python.readthedocs.io/en/latest/modules/postgres/

Usage:
    Run PostgreSQL tests with: pytest -m postgres
    Skip PostgreSQL tests with: pytest -m "not postgres"

Note: Requires Docker to be running on the host machine.
"""

import os
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Mark all tests in files using these fixtures as postgres tests
pytestmark = pytest.mark.postgres


def pytest_configure(config: Any) -> None:
    """Register the postgres marker."""
    config.addinivalue_line(
        "markers", "postgres: marks tests as requiring PostgreSQL (deselect with '-m \"not postgres\"')"
    )


@pytest.fixture(scope="session")
def postgres_container():
    """Start PostgreSQL container for integration tests.

    This fixture uses testcontainers to spin up a real PostgreSQL instance.
    The container is shared across all tests in the session for efficiency.

    Yields:
        PostgresContainer: Running PostgreSQL container with connection details.
    """
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers[postgres] not installed. Run: uv pip install testcontainers[postgres]")

    with PostgresContainer(
        image="postgres:16-alpine",
        user="test_user",
        password="test_password",
        dbname="test_bo_mcp",
    ) as postgres:
        yield postgres


@pytest.fixture(scope="session")
def postgres_url(postgres_container) -> str:
    """Get async PostgreSQL connection URL from container.

    Converts the sync connection URL to an async one using asyncpg driver.

    Returns:
        str: PostgreSQL connection URL with asyncpg driver.
    """
    # Get sync URL and convert to async
    sync_url = postgres_container.get_connection_url()
    # Replace postgresql:// with postgresql+asyncpg://
    async_url = sync_url.replace("postgresql://", "postgresql+asyncpg://")
    # Also handle psycopg2 driver if present
    async_url = async_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    return async_url


@pytest.fixture(scope="session")
def postgres_engine(postgres_url: str):
    """Create async SQLAlchemy engine connected to PostgreSQL container.

    Returns:
        AsyncEngine: Configured async engine for PostgreSQL.
    """
    return create_async_engine(
        postgres_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    )


@pytest_asyncio.fixture(scope="session")
async def postgres_tables(postgres_engine):
    """Create all database tables in the PostgreSQL container.

    This fixture runs once per session and creates the schema.
    """
    from bo_mcp_server.storage.models import Base

    async with postgres_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    # Optionally drop tables after all tests (usually not needed with containers)
    # async with postgres_engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def postgres_session(
    postgres_engine, postgres_tables
) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for PostgreSQL integration tests.

    Each test gets a fresh session that is rolled back after the test,
    ensuring test isolation without recreating tables.

    Yields:
        AsyncSession: Database session connected to PostgreSQL.
    """
    async_session_factory = async_sessionmaker(
        postgres_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        # Use a savepoint for test isolation
        async with session.begin():
            yield session
            # Rollback after each test to maintain isolation
            await session.rollback()


@pytest_asyncio.fixture
async def postgres_session_committed(
    postgres_engine, postgres_tables
) -> AsyncGenerator[AsyncSession, None]:
    """Create a session that commits changes (for tests that need persistence).

    Use this fixture when you need changes to be visible across multiple
    queries or when testing transaction behavior.

    Warning: Tests using this fixture may affect other tests. Use sparingly.

    Yields:
        AsyncSession: Database session that commits changes.
    """
    async_session_factory = async_sessionmaker(
        postgres_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        yield session
        await session.commit()


# Sample data fixtures for PostgreSQL tests


@pytest.fixture
def pg_sample_user_data() -> dict:
    """Sample user data for PostgreSQL tests."""
    return {
        "id": str(uuid4()),
        "name": "PostgreSQL Test User",
        "email": f"pg_test_{uuid4().hex[:8]}@example.com",
        "api_key_hash": "pg_test_hash_" + uuid4().hex,
    }


@pytest.fixture
def pg_sample_campaign_spec_data() -> dict:
    """Sample campaign spec data for PostgreSQL tests."""
    return {
        "id": str(uuid4()),
        "name": "PostgreSQL Test Campaign",
        "description": "Integration test campaign for PostgreSQL",
        "parameters_json": '[{"name":"x","type":"continuous","bounds":[0,1]}]',
        "objectives_json": '[{"name":"y","direction":"minimize"}]',
        "constraints_json": "[]",
        "batch_size": 2,
    }
