"""Database connection and session management.

This module provides async database connectivity for both PostgreSQL (production)
and SQLite (testing). It supports two initialization modes:

1. Alembic migrations (PostgreSQL): Schema versioning with upgrade/downgrade
2. Direct creation (SQLite): Fast setup for unit tests using Base.metadata.create_all()

The engine is lazily initialized on first use via get_session() or init_database(),
so importing this module does not require DATABASE_URL to be set.
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import dotenv
from alembic import command
from alembic.config import Config
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from bo_mcp_server.settings import (
    get_database_url,
    get_sql_echo,
    get_use_alembic_mode,
)
from bo_mcp_server.storage.models import Base

logger = logging.getLogger(__name__)

# Ensure .env values are available even when this module is imported directly.
dotenv.load_dotenv()

# Lazy-initialized engine and session factory. The engine is created at
# first use by ``_create_engine_with_options`` which reads the active
# :mod:`bo_mcp_server.settings` values at call time; tests that mutate
# ``os.environ`` via ``monkeypatch.setenv`` therefore only have to
# discard ``_engine`` (e.g. via ``close_database()``) to pick up the
# new URL on the next access.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _current_database_url() -> str:
    """Resolve ``DATABASE_URL`` at call time from the active settings."""
    return get_database_url()


def _current_use_alembic_mode() -> str:
    """Resolve ``USE_ALEMBIC`` at call time from the active settings."""
    return get_use_alembic_mode()


# Back-compat constants. These reflect the value at import time only.
# Code paths that need the live setting (``_create_engine_with_options``,
# ``init_database``, ``_run_alembic_migrations``, ``_should_use_alembic``)
# now call :func:`_current_database_url` / :func:`_current_use_alembic_mode`
# so test overrides applied via ``monkeypatch.setenv`` reach the engine
# factory. External callers that import :data:`DATABASE_URL` directly
# (rare) still see the import-time snapshot.
DATABASE_URL = get_database_url()
USE_ALEMBIC = get_use_alembic_mode()


def _get_engine() -> AsyncEngine:
    """Get or create the async engine (lazy initialization)."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        _engine = _create_engine_with_options()
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory (lazy initialization).

    ``expire_on_commit=True`` is the SQLAlchemy default and is restored
    here so any code that accidentally reads ORM attributes after a
    commit fails loudly (``MissingGreenlet`` on async lazy refresh)
    rather than silently returning stale field values. Repositories
    convert ORM models to frozen Pydantic domain entities before
    returning, so callers do not observe expired attributes. The
    ``cached_property`` parsed_* helpers on the ORM models are *not*
    cleared by ``expire_on_commit``; the immutability contract in
    ``storage/models.py`` already forbids re-reading them after a
    state mutation.
    """
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _get_engine(),
            class_=AsyncSession,
            expire_on_commit=True,
        )
    return _session_factory


def _create_engine_with_options() -> AsyncEngine:
    """Create async engine with database-specific options."""
    database_url = _current_database_url()
    common_options = {
        "echo": get_sql_echo(),
    }

    if database_url.startswith("postgresql"):
        # PostgreSQL-specific connection pool settings. ``pool_recycle`` is
        # set below the typical 1-hour PostgreSQL/PgBouncer idle timeout so
        # SQLAlchemy proactively retires stale connections instead of
        # surfacing "server closed the connection unexpectedly" errors on
        # the next checkout. See
        # https://docs.sqlalchemy.org/en/20/core/pooling.html#disconnect-handling-pessimistic
        # for the recommended pool_pre_ping + pool_recycle combination.
        return create_async_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=600,
            **common_options,
        )
    else:
        # SQLite (used for testing). FK enforcement is intentionally
        # *not* enabled engine-wide here: many pre-existing test
        # fixtures construct campaigns with synthetic ``owner_id`` /
        # ``spec_id`` UUIDs that have no matching parent row, and
        # turning the pragma on globally would surface those as
        # spurious failures unrelated to the change at hand. Tests
        # that specifically exercise the ``ON DELETE RESTRICT``
        # contract (see ``test_soft_delete_and_snapshot.py``) enable
        # the pragma on their own engine. Production runs PostgreSQL,
        # which enforces FKs unconditionally.
        return create_async_engine(database_url, **common_options)


def _should_use_alembic() -> bool:
    """Determine whether to use Alembic migrations.

    Returns True for PostgreSQL (production), False for SQLite (testing).
    Can be overridden via USE_ALEMBIC environment variable.
    """
    mode = _current_use_alembic_mode()
    if mode == "true":
        return True
    if mode == "false":
        return False
    # Auto-detect: use Alembic for PostgreSQL, direct creation for SQLite
    return _current_database_url().startswith("postgresql")


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
    alembic_cfg.set_main_option("sqlalchemy.url", _current_database_url())
    alembic_cfg.set_main_option("script_location", str(package_root / "migrations"))

    logger.info("Running Alembic migrations...")
    command.upgrade(alembic_cfg, "head")
    logger.info("Alembic migrations completed")


async def init_database() -> None:
    """Initialize database schema.

    For PostgreSQL: Uses Alembic migrations for versioned schema management.
    For SQLite: Uses direct Base.metadata.create_all() for fast test setup.
    """
    engine = _get_engine()
    database_url = _current_database_url()

    # Log with credentials masked (only shows host:port/db)
    logger.info("Initializing database: %s", database_url.split("@")[-1])

    # Ensure data directory exists for SQLite (testing only)
    if database_url.startswith("sqlite") and "memory" not in database_url:
        data_dir = os.path.dirname(database_url.replace("sqlite+aiosqlite:///", ""))
        if data_dir and data_dir != ".":
            os.makedirs(data_dir, exist_ok=True)

    if _should_use_alembic():
        # Use Alembic for PostgreSQL (production).
        # Run in a worker thread to avoid conflict with the running event loop —
        # Alembic's env.py uses asyncio.run() which requires no active loop.
        await asyncio.to_thread(_run_alembic_migrations)
    else:
        # Use direct creation for SQLite (testing)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized successfully")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession]:
    """Get a database session.

    Exits via either ``commit`` (no exception) or ``rollback`` followed by
    re-raise. Only ``SQLAlchemyError`` and ``RuntimeError`` are caught
    explicitly so storage-layer programming bugs (``AttributeError``,
    ``KeyError``, …) surface unchanged; ``AsyncSession.__aexit__`` still
    rolls back and disposes the connection for those cases. The bare
    ``raise`` preserves the originating traceback.
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except (SQLAlchemyError, RuntimeError):
            await session.rollback()
            raise


async def close_database() -> None:
    """Close database connections and dispose of the engine."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


@asynccontextmanager
async def lifespan() -> AsyncGenerator[None]:
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
        await close_database()
