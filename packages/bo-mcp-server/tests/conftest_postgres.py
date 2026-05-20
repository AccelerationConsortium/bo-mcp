"""PostgreSQL integration test fixtures using testcontainers.

These fixtures provide real PostgreSQL instances for integration testing,
ensuring database behavior matches production environments.

Test isolation uses the SQLAlchemy "savepoint + restart on commit" pattern
rather than re-running ``Base.metadata.create_all`` per test. The schema is
created once at session scope; each test runs inside an outer transaction
and a nested ``SAVEPOINT`` that is restarted whenever the application code
under test issues a ``commit``. At teardown the outer transaction is rolled
back, so no writes survive — even if the production code path committed
during the test.

This pattern is documented in the SQLAlchemy 2.0 ORM section
"Joining a Session into an External Transaction (such as for test suites)":
https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites

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
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Mark all tests in files using these fixtures as postgres tests
pytestmark = pytest.mark.postgres


def pytest_configure(config: Any) -> None:
    """Register the postgres marker."""
    config.addinivalue_line(
        "markers",
        "postgres: marks tests as requiring PostgreSQL (deselect with '-m \"not postgres\"')",
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
        from testcontainers.postgres import (  # ty: ignore[unresolved-import]
            PostgresContainer,  # type: ignore[import-not-found]
        )
    except ImportError:
        pytest.skip(
            "testcontainers[postgres] not installed. Run: uv pip install testcontainers[postgres]"
        )
        return  # unreachable, but helps type checker understand control flow

    with PostgresContainer(
        image="postgres:16-alpine",
        user="test_user",
        password="test_password",  # noqa: S106 - test container credentials
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
    return async_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")


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
    request: pytest.FixtureRequest, postgres_engine
) -> AsyncGenerator[AsyncSession]:
    """Create an async session for PostgreSQL integration tests.

    The session is bound to a dedicated connection that is wrapped in an
    outer transaction plus a nested ``SAVEPOINT``. When the production code
    under test calls ``commit`` the savepoint completes but the outer
    transaction stays open, and the ``after_transaction_end`` listener
    immediately starts a new savepoint so subsequent statements still see
    a transactional context. At teardown the outer transaction is rolled
    back unconditionally, so the schema is left untouched and the next
    test sees a clean slate without paying the cost of recreating tables.

    This pattern is the SQLAlchemy-recommended approach for test isolation
    against PostgreSQL: it is correct under application-side commits (which
    a flat ``async with session.begin()`` is not) and it is several orders
    of magnitude faster than recreating the engine + schema per test.

    Yields:
        AsyncSession: Database session connected to PostgreSQL.
    """
    request.getfixturevalue("postgres_tables")
    async with postgres_engine.connect() as connection:
        outer_transaction = await connection.begin()
        session = AsyncSession(bind=connection, expire_on_commit=True)
        await session.begin_nested()

        @event.listens_for(session.sync_session, "after_transaction_end")
        def _restart_savepoint(sync_session, transaction) -> None:  # noqa: ARG001
            # When the inner SAVEPOINT ends (commit or rollback inside the
            # test), immediately open a new one so subsequent statements
            # still execute inside a transaction we can roll back.
            sync_conn = connection.sync_connection
            if sync_conn is not None and not sync_conn.in_nested_transaction():
                sync_conn.begin_nested()

        try:
            yield session
        finally:
            await session.close()
            if outer_transaction.is_active:
                await outer_transaction.rollback()


@pytest_asyncio.fixture
async def postgres_session_committed(
    request: pytest.FixtureRequest, postgres_engine
) -> AsyncGenerator[AsyncSession]:
    """Create a session that commits changes (for tests that need persistence).

    Use this fixture when you need changes to be visible across multiple
    queries or when testing transaction behavior.

    Warning: Tests using this fixture may affect other tests. Use sparingly.

    Yields:
        AsyncSession: Database session that commits changes.
    """
    request.getfixturevalue("postgres_tables")
    async_session_factory = async_sessionmaker(
        postgres_engine,
        class_=AsyncSession,
        expire_on_commit=True,
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
