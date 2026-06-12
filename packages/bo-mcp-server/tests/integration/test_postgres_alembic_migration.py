"""End-to-end regression test for the Alembic silent-rollback bug.

The in-process tests in
``tests/unit/test_storage/test_alembic_statement_timeout.py`` verify
that ``migrations/env.py`` commits after the SET statements (the
fix) and that the SQLAlchemy 2.0 autobegin contract motivates the
commit (the why). What they cannot verify is "after the subprocess
wrapper in ``database._run_alembic_in_subprocess`` runs end-to-end
against a real PostgreSQL, the DDL is actually persisted". That is
exactly the failure mode the bug report described:

    api-1 | Alembic migrations completed
    api-1 | ProgrammingError: relation "idempotency_cache" does not exist

Alembic still exited 0 because no Python exception was raised — only
inspecting the database after the migration reveals the rolled-back
schema. This file fills that gap with a single ``postgres``-marked
test that:

1. Provisions a fresh database inside the shared testcontainers
   postgres (a ``CREATE DATABASE`` against the same container is
   ~100x cheaper than spinning up a second testcontainer and gives
   full isolation from the other postgres tests in this package).
2. Points ``DATABASE_URL`` at the fresh database and runs
   :func:`bo_mcp_server.storage.init_database` — the exact same
   code path the docker-compose api and mcp containers hit at
   startup.
3. Asserts on the persisted state through a raw ``asyncpg``
   connection (no SQLAlchemy caching as a confounder): every
   migration-created table is present in ``public`` schema, and
   ``alembic_version`` carries the head revision.

Run with: ``pytest -m postgres``.

References:
* Bug report: silent migration rollback observed in the docker
  compose stack on fresh installs with SQLAlchemy 2.0 — Alembic
  exits 0, schema is empty, idempotency GC trips over a missing
  table on first sweep.
* SQLAlchemy 2.0 commit-as-you-go contract —
  https://docs.sqlalchemy.org/en/20/core/connections.html#commit-as-you-go.
* testcontainers-python PostgresContainer —
  https://testcontainers-python.readthedocs.io/en/latest/modules/postgres/.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import asyncpg
import pytest
import pytest_asyncio
from alembic.config import Config
from alembic.script import ScriptDirectory

# Pull in postgres_container (session-scoped testcontainers fixture).
pytest_plugins = ["tests.conftest_postgres"]

# Gate the whole module behind ``-m postgres`` so the fast suite
# stays docker-free.
pytestmark = pytest.mark.postgres


def _parse_admin_dsn(container_url: str) -> dict[str, Any]:
    """Extract asyncpg connection kwargs for the maintenance ``postgres`` DB.

    ``PostgresContainer.get_connection_url`` returns a sync URL like
    ``postgresql+psycopg2://user:pw@host:port/test_bo_mcp``. To issue
    ``CREATE DATABASE`` / ``DROP DATABASE`` we need a connection to the
    server's default ``postgres`` maintenance database — those
    statements cannot run inside the target database itself and they
    cannot run inside a transaction block, which asyncpg respects.
    """
    sync_url = container_url.replace("postgresql+psycopg2://", "postgresql://")
    parsed = urlparse(sync_url)
    return {
        "host": parsed.hostname,
        "port": parsed.port,
        "user": parsed.username,
        "password": parsed.password,
        "database": "postgres",
    }


def _async_url_for(container_url: str, db_name: str) -> str:
    """Build a SQLAlchemy ``postgresql+asyncpg://`` URL for ``db_name``."""
    sync_url = container_url.replace("postgresql+psycopg2://", "postgresql://")
    parsed = urlparse(sync_url)
    return (
        f"postgresql+asyncpg://{parsed.username}:{parsed.password}"
        f"@{parsed.hostname}:{parsed.port}/{db_name}"
    )


def _expected_alembic_head_revision() -> str:
    """Resolve the head revision string from Alembic's script directory.

    Reading the head from ``ScriptDirectory`` rather than hardcoding a
    revision id keeps this test forward-compatible: when a new migration
    is added the head changes automatically, no test edit needed. It
    also strengthens the
    ``alembic_version`` assertion from "some non-empty value was
    stamped" to "the head of the migration tree was stamped" — a
    partial upgrade that stopped at an older revision would leave a
    truthy ``version_num`` and slip past a pure truthiness check.

    Reference: Alembic ``ScriptDirectory.from_config`` /
    ``get_current_head`` —
    https://alembic.sqlalchemy.org/en/latest/api/script.html#alembic.script.ScriptDirectory.get_current_head.
    """
    alembic_ini = Path(__file__).resolve().parents[2] / "alembic.ini"
    assert alembic_ini.exists(), (
        f"alembic.ini not found at {alembic_ini}; cannot resolve head "
        f"revision. The package layout may have changed."
    )
    config = Config(str(alembic_ini))
    # ``alembic.ini`` declares ``script_location = migrations`` as a
    # *relative* path. ``ScriptDirectory.from_config`` resolves it
    # against ``os.getcwd()`` — not against the ini file's directory —
    # so the lookup only succeeds when pytest is invoked from inside
    # ``packages/bo-mcp-server`` (cwd has a ``migrations/`` sibling).
    # Running ``uv run pytest packages/bo-mcp-server/...`` from the
    # repo root would resolve to ``<repo>/migrations`` and raise
    # ``CommandError: Path doesn't exist: migrations``. Overriding
    # the option to an absolute path makes the helper cwd-independent.
    config.set_main_option("script_location", str(alembic_ini.parent / "migrations"))
    script = ScriptDirectory.from_config(config)
    head = script.get_current_head()
    assert head is not None, (
        "Alembic script directory reports no head — the migration tree "
        "is broken (no revisions present, or branches without a merge). "
        "This is a development-time assertion: production cannot reach "
        "this state because the package would not boot without a valid "
        "migration tree."
    )
    return head


@pytest_asyncio.fixture
async def fresh_alembic_database(postgres_container: Any) -> AsyncGenerator[str]:
    """Provision a unique empty database for one Alembic upgrade.

    Per-test ``CREATE DATABASE`` rather than truncating the shared
    ``test_bo_mcp`` database for two reasons:

    * The other postgres tests use ``test_bo_mcp`` with tables created
      via ``Base.metadata.create_all`` (see ``conftest_postgres.py``).
      Dropping or truncating that schema would either tear down those
      tables or leave orphan rows behind depending on order. A fresh
      database has none of those concerns.
    * Alembic's bug surfaces only when starting from a *truly empty*
      schema (no ``alembic_version``, no application tables). Reusing
      a database that already has tables would mask the bug because
      the upgrade has nothing to do.

    Cleanup terminates any leftover backends in the target database
    before ``DROP DATABASE`` because PostgreSQL refuses to drop a
    database with active connections. The SQLAlchemy engine cache in
    ``bo_mcp_server.storage.database`` should already be disposed by
    the test, but defensive termination keeps the fixture robust
    against test-side leaks.
    """
    admin_kwargs = _parse_admin_dsn(postgres_container.get_connection_url())
    db_name = f"alembic_regression_{uuid4().hex[:12]}"

    admin = await asyncpg.connect(**admin_kwargs)
    try:
        # Identifier is uuid-derived (hex only); the double-quoting is
        # defence-in-depth so this fixture stays safe even if the name
        # scheme above ever changes.
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    try:
        yield _async_url_for(postgres_container.get_connection_url(), db_name)
    finally:
        admin = await asyncpg.connect(**admin_kwargs)
        try:
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = $1 AND pid <> pg_backend_pid()",
                db_name,
            )
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        finally:
            await admin.close()


async def test_alembic_upgrade_persists_application_tables_on_fresh_postgres(
    fresh_alembic_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Migration must leave the application tables present in the schema.

    This is the direct end-to-end regression for the bug report.
    The unit tests assert that env.py emits the right SQL and commits
    at the right moment; this test asserts that doing so actually
    causes PostgreSQL to keep the tables.

    Pinned tables include ``idempotency_cache`` specifically — the
    bug report's smoking-gun symptom was the idempotency GC sweep
    failing with ``relation "idempotency_cache" does not exist``
    moments after Alembic logged success. The other five tables are
    created by the initial migration (``20250116_000000``) and
    failing on any of them would also indicate the rollback bug.
    """
    monkeypatch.setenv("DATABASE_URL", fresh_alembic_database)
    monkeypatch.setenv("USE_ALEMBIC", "true")

    from bo_mcp_server.storage import close_database, init_database

    # Discard any engine a prior test cached against a different
    # DATABASE_URL. ``_create_engine_with_options`` reads
    # ``DATABASE_URL`` at engine-construction time only, so without
    # this reset the test could end up running migrations against the
    # wrong database.
    await close_database()
    try:
        await init_database()
    finally:
        # Drop our engine before the fixture tries to DROP DATABASE —
        # otherwise the engine's pooled connection would block the drop.
        await close_database()

    verify_kwargs = _parse_admin_dsn(fresh_alembic_database)
    verify_kwargs["database"] = urlparse(
        fresh_alembic_database.replace("postgresql+asyncpg://", "postgresql://")
    ).path.lstrip("/")
    verify = await asyncpg.connect(**verify_kwargs)
    try:
        rows = await verify.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        actual_tables = {row["tablename"] for row in rows}
    finally:
        await verify.close()

    expected_tables = {
        "users",
        "campaign_specs",
        "campaigns",
        "suggestions",
        "results",
        # idempotency_cache is the smoking-gun table from the bug
        # report — the GC sweep crashed on its absence post-migration.
        "idempotency_cache",
    }
    missing = expected_tables - actual_tables
    assert not missing, (
        f"After init_database() against a fresh PostgreSQL, expected the "
        f"application tables to be persisted, but the following are "
        f"missing from the public schema: {sorted(missing)}. "
        f"Tables found: {sorted(actual_tables)}. "
        f"This is the exact failure mode of the silent-rollback bug — "
        f"Alembic exits 0 because no Python exception was raised, but the "
        f"DDL never made it past the implicit transaction the SETs in "
        f"env.py opened."
    )


async def test_alembic_upgrade_stamps_head_revision_in_alembic_version(
    fresh_alembic_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``alembic_version`` must contain the head revision after upgrade.

    Complements the table-existence test above. The bug-report
    validation step was ``alembic current -v`` which returned "no
    current revision" — that is exactly the failure shape this test
    pins: a missing or empty ``alembic_version`` table.

    Asserting on ``len == 1`` (not just ``>= 1``) catches a related
    failure shape where Alembic stamps multiple times due to a
    misconfigured branch or partial rollback — both shapes would
    leave the database in a state where future ``alembic upgrade``
    cannot reliably resolve "head".
    """
    monkeypatch.setenv("DATABASE_URL", fresh_alembic_database)
    monkeypatch.setenv("USE_ALEMBIC", "true")

    from bo_mcp_server.storage import close_database, init_database

    await close_database()
    try:
        await init_database()
    finally:
        await close_database()

    verify_kwargs = _parse_admin_dsn(fresh_alembic_database)
    verify_kwargs["database"] = urlparse(
        fresh_alembic_database.replace("postgresql+asyncpg://", "postgresql://")
    ).path.lstrip("/")
    verify = await asyncpg.connect(**verify_kwargs)
    try:
        alembic_version_exists = await verify.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename = 'alembic_version')"
        )
        assert alembic_version_exists, (
            "alembic_version table is missing after init_database() against "
            "a fresh PostgreSQL. The canonical signature of the silent "
            "rollback bug is `alembic current -v` returning 'no current "
            "revision' — i.e. this table never persisted."
        )

        version_rows = await verify.fetch("SELECT version_num FROM alembic_version")
    finally:
        await verify.close()

    assert len(version_rows) == 1, (
        f"alembic_version must hold exactly one row after a clean upgrade "
        f"to head; found {len(version_rows)}. An empty row count is the "
        f"silent-rollback signature; more than one would indicate a "
        f"partial / branched upgrade."
    )
    stamped_revision = version_rows[0]["version_num"]
    expected_head = _expected_alembic_head_revision()
    assert stamped_revision == expected_head, (
        f"alembic_version stamped {stamped_revision!r} but the migration "
        f"tree's head is {expected_head!r}. A truthy-but-stale stamp would "
        f"sneak past a pure non-empty check while leaving the database "
        f"behind on an older revision — pinning against the script tree's "
        f"head catches partial upgrades as well as the silent-rollback "
        f"shape covered above."
    )
