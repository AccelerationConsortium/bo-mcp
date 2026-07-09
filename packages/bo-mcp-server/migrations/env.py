"""Alembic environment configuration for async SQLAlchemy.

This module configures Alembic for asynchronous database migrations with
PostgreSQL (production) and SQLite (testing) support.

Reference: https://alembic.sqlalchemy.org/en/latest/cookbook.html#using-asyncio-with-alembic
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from bo_mcp_server.settings import get_database_url, load_anchored_dotenv

# Import models to ensure they're registered with Base.metadata
from bo_mcp_server.storage.models import Base

# Alembic Config object for access to .ini file values
config = context.config

# Configure Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Ensure Alembic sees the same anchored .env configuration as application
# startup (never the invoking shell's CWD).
load_anchored_dotenv()

# Target metadata for 'autogenerate' support
target_metadata = Base.metadata

# Resolve the database URL through the shared settings accessor so the
# fallback default (package-anchored absolute SQLite path) cannot drift
# from database.py.
DATABASE_URL = get_database_url()


def get_url() -> str:
    """Get database URL, handling async driver prefixes."""
    return DATABASE_URL


_DEFAULT_TIMEOUT_SECONDS = 300.0


def _statement_timeout_seconds() -> float:
    """Resolve the per-statement timeout cap from the active environment.

    Mirrors :func:`bo_mcp_server.settings.get_database_init_timeout_seconds`,
    including its positive-value validation. Read directly from
    ``os.environ`` because Alembic CLI usage (``alembic upgrade head``
    invoked from a shell or from the subprocess kill-switch path) does
    not initialise the Settings module. Falls back to the same 300s
    default the application uses.

    Non-positive values are rejected with a fallback to the default:
    PostgreSQL treats ``SET
    statement_timeout = 0`` as 'unlimited' — exactly the opposite of
    the intended kill-switch behaviour — and negative values yield an
    invalid SET statement. Either silently disables the DB-side cap
    that the in-process wait_for cannot enforce alone, so we coerce
    back to the default and log a warning rather than honouring the
    bad value.
    """
    raw = os.environ.get("DATABASE_INIT_TIMEOUT_SECONDS")
    if raw is None:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_SECONDS
    if value <= 0.0:
        # Avoid log import here — env.py runs before logging is fully
        # configured during Alembic CLI usage. Falling back to the
        # default is the safest behaviour because the alternative is
        # an unbounded SET that defeats the kill switch.
        return _DEFAULT_TIMEOUT_SECONDS
    return value


def _apply_postgres_statement_timeout(connection: Connection) -> None:
    """Enforce DB-side timeouts on the migration connection.

    Python cannot cancel a worker thread once it is running
    (``asyncio.wait_for(asyncio.to_thread(...))`` returns to the caller
    on timeout but the thread keeps executing). Without DB-side
    timeouts, a stuck migration would continue mutating the database
    after the startup wrapper has already raised
    :class:`DatabaseInitializationError(stage='timeout')`. We set three
    complementary caps so a long-running migration is bounded along
    each dimension a single Alembic upgrade can spend time:

    * ``statement_timeout`` aborts any individual statement that runs
      past the configured budget (the typical DDL hang).
    * ``lock_timeout`` aborts a statement waiting for a row / table
      lock past the same budget (a concurrent transaction sitting on
      the table Alembic wants to alter).
    * ``idle_in_transaction_session_timeout`` aborts the connection
      itself if Alembic enters a transaction and then stalls between
      statements (Python-side work between DDL steps, runaway
      data-migration loop). Without this third cap, a migration with
      many under-budget statements separated by long Python pauses
      could still run past the orchestrator-side timeout — exactly
      the residual gap the second review pass flagged.

    Skipped for non-PostgreSQL dialects (SQLite has no equivalent and
    is only used for tests). The ``SET`` is connection-scoped so the
    caps apply across every per-revision transaction Alembic opens
    during the upgrade.

    The trailing ``connection.commit()`` is load-bearing. SQLAlchemy
    2.0 ``Connection`` autobegins an implicit transaction on the first
    ``execute()`` (the first SET above), so by the time Alembic calls
    ``context.begin_transaction()`` the connection is already inside a
    transaction. Alembic detects that and turns its own
    ``begin_transaction()`` into a no-op context manager
    (``_in_connection_transaction()`` branch). Every per-revision DDL
    statement then runs inside the implicit transaction started by the
    SETs — never inside a transaction Alembic owns. When the async
    ``connect()`` block in :func:`run_async_migrations` exits without
    an explicit commit, SQLAlchemy 2.0 rolls the implicit transaction
    back (documented "begin once" behaviour), discarding the schema
    changes silently while Alembic still exits 0. Committing here
    closes that implicit transaction so Alembic opens and commits its
    own. The session-scoped SETs survive the commit because they were
    issued with ``SET`` (not ``SET LOCAL``) — that is exactly why the
    helper does not wrap them in ``SET LOCAL``: ``SET LOCAL`` would
    bind the cap to the transaction we are about to commit and the
    timeouts would vanish for Alembic's subsequent transactions.

    Reference: PostgreSQL docs on these GUCs —
    https://www.postgresql.org/docs/current/runtime-config-client.html.
    SQLAlchemy 2.0 autobegin / commit-on-close contract —
    https://docs.sqlalchemy.org/en/20/core/connections.html#commit-as-you-go.
    """
    if connection.dialect.name != "postgresql":
        return
    timeout_ms = int(_statement_timeout_seconds() * 1000)
    # SET <timeout> = N (no quoting needed for integer milliseconds).
    connection.execute(text(f"SET statement_timeout = {timeout_ms}"))
    connection.execute(text(f"SET lock_timeout = {timeout_ms}"))
    connection.execute(text(f"SET idle_in_transaction_session_timeout = {timeout_ms}"))
    # Close the implicit transaction opened by the SETs so Alembic's
    # context.begin_transaction() opens a real, Alembic-owned
    # transaction whose commit persists the DDL. See the docstring
    # above for the autobegin / rollback-on-close trap this avoids.
    connection.commit()


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
    _apply_postgres_statement_timeout(connection)
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
    """Run migrations in 'online' mode.

    Uses asyncio.run() to execute async migrations. This function must be called
    from a thread without a running event loop. When invoked from an async context
    (e.g., FastAPI lifespan), the caller should use asyncio.to_thread().
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
