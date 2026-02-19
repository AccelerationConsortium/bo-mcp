"""Alembic environment configuration for async SQLAlchemy.

This module configures Alembic for asynchronous database migrations with
PostgreSQL (production) and SQLite (testing) support.

Reference: https://alembic.sqlalchemy.org/en/latest/cookbook.html#using-asyncio-with-alembic
"""

import asyncio
import os
import threading
from logging.config import fileConfig

import dotenv
from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import models to ensure they're registered with Base.metadata
from bo_mcp_server.storage.models import Base

# Alembic Config object for access to .ini file values
config = context.config

# Configure Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Ensure Alembic sees the same .env configuration as application startup.
dotenv.load_dotenv()

# Target metadata for 'autogenerate' support
target_metadata = Base.metadata

# Get database URL from environment (consistent with database.py)
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://bo_user:bo_password@localhost:5432/bo_mcp"
)


def get_url() -> str:
    """Get database URL, handling async driver prefixes."""
    return DATABASE_URL


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine.
    Calls to context.execute() emit the given string to the script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with an active connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine.

    Creates an async Engine and associates a connection with the context.
    """
    # Build configuration dict for async engine
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.get_running_loop()

    def _run() -> None:
        asyncio.run(run_async_migrations())

    thread = threading.Thread(target=_run, name="alembic-migrations")
    thread.start()
    thread.join()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
