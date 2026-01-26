"""Database connection and session management.

This module provides async database connectivity for both PostgreSQL (production)
and SQLite (testing). It supports two initialization modes:

1. Alembic migrations (PostgreSQL): Schema versioning with upgrade/downgrade
2. Direct creation (SQLite): Fast setup for unit tests using Base.metadata.create_all()

The initialization mode is automatically selected based on DATABASE_URL.
"""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from bo_mcp_server.storage.models import Base

logger = logging.getLogger(__name__)

# Default to PostgreSQL (override via DATABASE_URL environment variable)
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://bo_user:bo_password@localhost:5432/bo_mcp"
)

# Use Alembic for PostgreSQL, direct creation for SQLite (testing)
USE_ALEMBIC = os.getenv("USE_ALEMBIC", "auto")  # "auto", "true", or "false"


def _create_engine_with_options() -> AsyncEngine:
    """Create async engine with database-specific options."""
    common_options = {
        "echo": os.getenv("SQL_ECHO", "false").lower() == "true",
    }

    if DATABASE_URL.startswith("postgresql"):
        # PostgreSQL-specific connection pool settings
        return create_async_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            **common_options,
        )
    else:
        # SQLite (used for testing)
        return create_async_engine(DATABASE_URL, **common_options)


engine = _create_engine_with_options()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


def _should_use_alembic() -> bool:
    """Determine whether to use Alembic migrations.

    Returns True for PostgreSQL (production), False for SQLite (testing).
    Can be overridden via USE_ALEMBIC environment variable.
    """
    if USE_ALEMBIC == "true":
        return True
    if USE_ALEMBIC == "false":
        return False
    # Auto-detect: use Alembic for PostgreSQL, direct creation for SQLite
    return DATABASE_URL.startswith("postgresql")


def _run_alembic_migrations() -> None:
    """Run Alembic migrations to upgrade database to latest version.

    This is a synchronous operation that runs in a subprocess-safe manner.
    """
    # Find alembic.ini relative to this file
    package_root = Path(__file__).parent.parent.parent.parent
    alembic_ini = package_root / "alembic.ini"

    if not alembic_ini.exists():
        logger.warning("alembic.ini not found at %s, skipping migrations", alembic_ini)
        return

    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)

    logger.info("Running Alembic migrations...")
    command.upgrade(alembic_cfg, "head")
    logger.info("Alembic migrations completed")


async def init_database() -> None:
    """Initialize database schema.

    For PostgreSQL: Uses Alembic migrations for versioned schema management.
    For SQLite: Uses direct Base.metadata.create_all() for fast test setup.
    """
    # Log with credentials masked (only shows host:port/db)
    logger.info("Initializing database: %s", DATABASE_URL.split("@")[-1])

    # Ensure data directory exists for SQLite (testing only)
    if DATABASE_URL.startswith("sqlite") and "memory" not in DATABASE_URL:
        data_dir = os.path.dirname(DATABASE_URL.replace("sqlite+aiosqlite:///", ""))
        if data_dir and data_dir != ".":
            os.makedirs(data_dir, exist_ok=True)

    if _should_use_alembic():
        # Use Alembic for PostgreSQL (production)
        _run_alembic_migrations()
    else:
        # Use direct creation for SQLite (testing)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized successfully")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_database() -> None:
    """Close database connections and dispose of the engine."""
    await engine.dispose()


@asynccontextmanager
async def lifespan() -> AsyncGenerator[None, None]:
    """Context manager for database lifecycle.

    Usage:
        async with lifespan():
            # database is initialized
            await do_work()
        # database connections are closed
    """
    await init_database()
    try:
        yield
    finally:
        await engine.dispose()
