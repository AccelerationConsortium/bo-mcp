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
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import dotenv
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from bo_mcp_server.settings import (
    get_database_init_connect_timeout_seconds,
    get_database_init_timeout_seconds,
    get_database_url,
    get_db_max_overflow,
    get_db_pool_recycle_seconds,
    get_db_pool_size,
    get_sql_echo,
    get_use_alembic_mode,
)
from bo_mcp_server.storage.models import Base


class DatabaseInitializationError(RuntimeError):
    """Raised when ``init_database`` cannot bring the schema up to head.

    Three failure shapes collapse into one typed exception so the
    orchestrator (k8s, systemd) sees a single ``ExitCode != 0`` signal
    regardless of cause:

    * connectivity probe failed before Alembic ran (DNS, credentials,
      firewall, server down)
    * Alembic ``command.upgrade`` exceeded the configured timeout
    * Alembic itself raised (bad migration script, schema clash)

    ``cause`` is the original exception. ``stage`` is a short tag —
    ``connect`` / ``timeout`` / ``alembic`` — so the readiness probe
    can emit a structured error_class without inspecting the exception
    chain.
    """

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


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
            pool_size=get_db_pool_size(),
            max_overflow=get_db_max_overflow(),
            pool_pre_ping=True,
            pool_recycle=get_db_pool_recycle_seconds(),
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


def _run_alembic_in_subprocess(timeout_seconds: float) -> None:
    """Run ``python -m alembic upgrade head`` in a subprocess with a SIGKILL on timeout.

    Real wall-clock kill switch (TODO 8.22 third review pass). The
    earlier ``asyncio.to_thread(in_process_command.upgrade)`` could
    only abort the *await* — the worker thread kept running, and a
    migration made of many short individually-bounded statements with
    long Python work between them could outlive the orchestrator's
    declared timeout. :func:`subprocess.run` with ``timeout=N`` is
    the canonical Python kill switch: when the budget elapses the
    subprocess receives ``SIGKILL`` from the OS and ``run`` raises
    :class:`subprocess.TimeoutExpired`. The child cannot continue.

    Layering with the DB-side caps in :mod:`migrations.env`
    (statement / lock / idle_in_transaction) gives a belt-and-
    suspenders kill: PostgreSQL aborts individual statements / locks
    that hang, the process-level SIGKILL bounds the whole upgrade.

    The subprocess inherits ``DATABASE_URL`` (and any ``.env`` values
    loaded by the parent) so env.py resolves the same connection
    string. Stderr is captured and embedded in the raised
    :class:`RuntimeError` on non-zero exit so the typed envelope
    surfaces a useful triage hint without leaking the full Alembic
    traceback into client responses.
    """
    package_root = Path(__file__).parent.parent.parent.parent
    alembic_ini = package_root / "alembic.ini"

    if not alembic_ini.exists():
        logger.warning("alembic.ini not found at %s, skipping migrations", alembic_ini)
        return

    cmd = [
        sys.executable,
        "-m",
        "alembic",
        "-c",
        str(alembic_ini),
        "upgrade",
        "head",
    ]

    logger.info("Running Alembic migrations in subprocess (timeout=%.1fs)", timeout_seconds)
    try:
        result = subprocess.run(  # noqa: S603 - args are derived from trusted package layout
            cmd,
            timeout=timeout_seconds,
            check=False,
            capture_output=True,
            cwd=str(package_root),
        )
    except subprocess.TimeoutExpired as exc:
        # The subprocess has been SIGKILLed by subprocess.run; surface
        # a uniform TimeoutError so the caller's existing exception
        # routing maps it to ``stage='timeout'``.
        raise TimeoutError(
            f"Alembic upgrade exceeded {timeout_seconds:.1f}s and was SIGKILLed"
        ) from exc

    if result.returncode != 0:
        stderr_excerpt = result.stderr.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Alembic exited with code {result.returncode}: {stderr_excerpt}")
    logger.info("Alembic migrations completed")


async def _preflight_connectivity(engine: AsyncEngine, timeout_seconds: float) -> None:
    """Verify the configured DB is reachable before kicking off Alembic.

    A bad ``DATABASE_URL`` (wrong host, expired credentials, firewall
    block) would otherwise burn the full migration timeout on a
    connection that can never succeed. The probe is a single
    ``SELECT 1`` round-trip wrapped in :func:`asyncio.wait_for` so a
    hung TCP handshake cannot stall startup past
    ``DATABASE_INIT_CONNECT_TIMEOUT_SECONDS``.

    Skipped for SQLite — the in-memory / file-backed driver does not
    open a real socket and any "connectivity" failure there is really
    a permission / path issue that surfaces from the actual
    ``create_all`` step anyway.
    """
    if _current_database_url().startswith("sqlite"):
        return

    async def _probe() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    try:
        await asyncio.wait_for(_probe(), timeout=timeout_seconds)
    except TimeoutError as exc:
        raise DatabaseInitializationError(
            "connect",
            f"Database connectivity probe timed out after {timeout_seconds:.1f}s; "
            "check DATABASE_URL host/port and network reachability.",
        ) from exc
    except (SQLAlchemyError, OSError) as exc:
        # ``OSError`` covers the asyncpg connect path (``ConnectionRefusedError``,
        # ``socket.gaierror``) which SQLAlchemy does not always wrap into
        # ``SQLAlchemyError`` before raising. Both shapes are "DB unreachable"
        # to the operator.
        raise DatabaseInitializationError(
            "connect",
            f"Database connectivity probe failed ({type(exc).__name__}); "
            "verify DATABASE_URL credentials and that the server is up.",
        ) from exc


async def init_database() -> None:
    """Initialize database schema.

    For PostgreSQL: pre-flight ``SELECT 1`` connectivity check, then
    Alembic upgrade wrapped in :func:`asyncio.wait_for` so a hung
    migration script cannot stall lifespan indefinitely.
    For SQLite: direct ``Base.metadata.create_all()`` for fast test
    setup; no timeout because the in-process driver returns
    synchronously.

    Worker-thread caveat (TODO 8.22 review pass): ``asyncio.wait_for``
    aborts the coroutine it wraps, but Python cannot interrupt a
    worker thread spawned via :func:`asyncio.to_thread`. The Alembic
    upgrade runs in such a thread, so the timeout here is *advisory*
    for the orchestrator (we surface ``DatabaseInitializationError``
    fast) but does not guarantee the migration thread itself has
    stopped. The matching *real* abort happens DB-side: the migration
    connection sets ``statement_timeout`` and ``lock_timeout`` (see
    ``migrations/env.py``) to the same window, so a runaway statement
    is rolled back by PostgreSQL when the budget is exhausted.

    Raises:
        DatabaseInitializationError: when any of the three startup
            phases (connect probe, Alembic upgrade timeout, Alembic
            upgrade itself) fails. The exception carries a short
            ``stage`` tag so readiness probes can emit a structured
            error_class.
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
        connect_timeout = get_database_init_connect_timeout_seconds()
        upgrade_timeout = get_database_init_timeout_seconds()
        await _preflight_connectivity(engine, connect_timeout)
        # Wall-clock kill switch (TODO 8.22 third review pass): run
        # the migration as a child process so subprocess.run(timeout=N)
        # can SIGKILL it. ``asyncio.to_thread`` parks the blocking
        # call on the worker pool — the subprocess.run inside it
        # enforces the real kill, the wrapper just bridges back to
        # async. Layered with the DB-side timeouts in
        # ``migrations/env.py`` so individual statements / locks /
        # idle-in-transaction stalls all hit their own caps too.
        try:
            await asyncio.to_thread(_run_alembic_in_subprocess, upgrade_timeout)
        except TimeoutError as exc:
            raise DatabaseInitializationError(
                "timeout",
                f"Alembic schema upgrade exceeded the configured "
                f"{upgrade_timeout:.1f}s timeout; the subprocess was "
                "SIGKILLed. Check for a long-running migration or "
                "DB-side lock.",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - intentional translation boundary
            # Alembic raises a wide and undocumented set of exception
            # types: ``CommandError`` for CLI-level failures (missing
            # revision, ambiguous head), ``RuntimeError`` and
            # ``ValueError`` from env.py / script bodies, plus
            # ``SQLAlchemyError`` for DB-side faults. With the
            # subprocess wrapper a non-zero exit code surfaces as
            # ``RuntimeError`` carrying the captured stderr — the
            # catch stays broad so any future shape of failure still
            # maps cleanly to ``stage='alembic'``. ``TimeoutError``
            # has its own branch above; ``BaseException`` subclasses
            # (``KeyboardInterrupt``, ``SystemExit``) are unaffected.
            raise DatabaseInitializationError(
                "alembic",
                f"Alembic schema upgrade failed ({type(exc).__name__}); "
                "inspect the migration log for the offending revision.",
            ) from exc
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
