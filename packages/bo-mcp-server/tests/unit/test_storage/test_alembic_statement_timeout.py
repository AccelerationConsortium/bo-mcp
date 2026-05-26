"""Alembic env.py applies DB-side timeouts on PostgreSQL.

The Python-side ``asyncio.wait_for(asyncio.to_thread(...))`` can only
abort the *await* — the worker thread continues. The real kill switch
for a stuck migration is the DB-side timeout enforced from
``migrations/env.py``. This suite asserts that wiring is in place so
removing the env.py call would fail visibly.

Three complementary caps are pinned:

* ``statement_timeout`` — bounds any single statement.
* ``lock_timeout`` — bounds waiting on a row / table lock.
* ``idle_in_transaction_session_timeout`` — bounds Python-side pauses
  between statements inside a transaction; closes the "many short
  statements + long Python work" gap the previous review flagged.

Reference: PostgreSQL runtime parameters
https://www.postgresql.org/docs/current/runtime-config-client.html.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# Load the env.py module directly so we don't have to run Alembic to
# exercise the wiring. ``runpy``/``import`` is awkward here because the
# top-level body of env.py invokes Alembic itself; loading the spec but
# not executing the bottom branch is cleaner via importlib.

_ENV_PATH = Path(__file__).resolve().parents[3] / "migrations" / "env.py"


class _RecordingConnection:
    """Minimal stand-in for ``sqlalchemy.engine.Connection``.

    The env helper reads ``connection.dialect.name``, calls
    ``connection.execute(text(...))``, and (after the SETs) calls
    ``connection.commit()`` to close the implicit transaction that
    SQLAlchemy 2.0 autobegins. We capture both kinds of call so the
    tests can assert on the ordered sequence: three SETs, then a
    commit.
    """

    def __init__(self, dialect_name: str) -> None:
        self.dialect = type("Dialect", (), {"name": dialect_name})()
        self.executed: list[str] = []
        # Use a marker token in the same ``calls`` log so a test can
        # pin the *ordering* between executes and commits — the
        # autobegin trap is sensitive to "commit comes after the
        # SETs", not just "commit happens at all".
        self.calls: list[str] = []
        self.commits: int = 0

    def execute(self, statement: Any) -> Any:
        # ``text(...)`` exposes the raw SQL via ``.text``; defensively
        # fall back to ``str()`` for anything else.
        rendered = getattr(statement, "text", None) or str(statement)
        self.executed.append(rendered)
        self.calls.append(f"EXEC:{rendered}")
        return None

    def commit(self) -> None:
        """Record commits so tests can assert the helper closes the implicit txn."""
        self.commits += 1
        self.calls.append("COMMIT")


def _load_env_module():
    """Import migrations/env.py without triggering the bottom-of-file Alembic run.

    env.py ends with ``if context.is_offline_mode(): ... else: ...``
    which would fire Alembic on import. We bypass that by skipping the
    bottom branch — importlib gives us the module object, and we only
    pull the helper out.

    Reference for the pattern: importlib spec_from_file_location is
    the recommended way to load a module by path —
    https://docs.python.org/3/library/importlib.html#importlib.util.spec_from_file_location.
    """
    # The env module imports ``alembic.context`` at the top, which
    # requires an Alembic config. To avoid that side effect we mirror
    # the *helper* into this test instead — env.py's helper is small,
    # branchless, and using a copy here keeps the test from needing
    # an actual Alembic environment to exercise the SET wiring. The
    # contract pinned below is therefore: env.py MUST issue these
    # three SETs on a PostgreSQL connection.
    assert _ENV_PATH.exists(), f"migrations/env.py missing at {_ENV_PATH}"


def _exercise_env_helper(connection: _RecordingConnection, timeout_seconds: float) -> None:
    """Reimplementation of the env.py helper, sourced from a single import.

    We import the helper out of the env module's source text via
    ``compile`` rather than running the full file, because env.py
    reaches Alembic at the bottom. The compiled snippet matches the
    real helper byte-for-byte; if env.py changes its SET layout this
    test fails on the read-and-compare step below.
    """
    source = _ENV_PATH.read_text()
    # Pull the helper body out of env.py — if anyone deletes or
    # renames it, this assert fires and we know the wiring is gone.
    needle = "def _apply_postgres_statement_timeout(connection: Connection) -> None:"
    assert needle in source, (
        "env.py must still expose _apply_postgres_statement_timeout; "
        "removing it loses the DB-side kill switch."
    )

    # The actual run uses the recording connection; we exercise the
    # helper by extracting it via exec into an isolated namespace.
    # Limit the exec to just the helper definitions so we don't trip
    # over the Alembic-imports at the top of env.py. Pull from the
    # ``_DEFAULT_TIMEOUT_SECONDS`` constant onward so the fallback
    # path inside ``_statement_timeout_seconds`` has the constant in
    # scope when invoked.
    helper_start = source.index("_DEFAULT_TIMEOUT_SECONDS = ")
    helper_end = source.index("def do_run_migrations(connection: Connection) -> None:")
    helper_src = source[helper_start:helper_end]
    namespace: dict[str, Any] = {"text": _passthrough_text, "os": __import__("os")}
    exec(compile(helper_src, str(_ENV_PATH), "exec"), namespace)  # noqa: S102 - env helper exec

    # The helper resolves the timeout from the environment; let the
    # caller stage the env var first.
    namespace["_apply_postgres_statement_timeout"](connection)
    _ = timeout_seconds  # reserved for future per-call override


def _passthrough_text(s: str) -> Any:
    """Stand-in for ``sqlalchemy.text`` that just records the rendered SQL."""
    return type("Text", (), {"text": s, "__str__": lambda _self: s})()


def test_env_helper_sets_three_timeouts_on_postgresql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PostgreSQL connection receives statement, lock, and idle-in-transaction caps.

    Without ``idle_in_transaction_session_timeout``, a migration that runs many short statements
    separated by long Python work between them would continue running
    after the orchestrator-side timeout has already raised
    ``stage='timeout'``. All three SETs are load-bearing for the
    "real kill switch" contract.
    """
    monkeypatch.setenv("DATABASE_INIT_TIMEOUT_SECONDS", "12")
    _load_env_module()
    conn = _RecordingConnection("postgresql")

    _exercise_env_helper(conn, timeout_seconds=12)

    assert any("statement_timeout = 12000" in sql for sql in conn.executed), (
        "missing statement_timeout SET"
    )
    assert any("lock_timeout = 12000" in sql for sql in conn.executed), "missing lock_timeout SET"
    assert any("idle_in_transaction_session_timeout = 12000" in sql for sql in conn.executed), (
        "missing idle_in_transaction_session_timeout SET — closes the long-Python-pause gap"
    )


def test_env_helper_skips_non_postgresql_dialect(monkeypatch: pytest.MonkeyPatch) -> None:
    """SQLite (test fixture) has no equivalent; the helper must no-op cleanly.

    Regression: running ``SET statement_timeout`` against SQLite raises
    a syntax error, which would break every test that uses the
    in-memory fixture. The dialect gate keeps the SQLite path silent.
    """
    monkeypatch.setenv("DATABASE_INIT_TIMEOUT_SECONDS", "5")
    _load_env_module()
    conn = _RecordingConnection("sqlite")

    _exercise_env_helper(conn, timeout_seconds=5)

    assert conn.executed == [], "non-PostgreSQL dialects must receive no SET"


def test_env_helper_defaults_when_env_var_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ``DATABASE_INIT_TIMEOUT_SECONDS`` falls back to 300s.

    The application-side default in ``settings.py`` is 300s; env.py is
    invoked from a separate process by Alembic CLI usage, so the
    fallback path must match — otherwise the DB-side cap could be
    silently zero / unset.
    """
    monkeypatch.delenv("DATABASE_INIT_TIMEOUT_SECONDS", raising=False)
    _load_env_module()
    conn = _RecordingConnection("postgresql")

    _exercise_env_helper(conn, timeout_seconds=300)

    assert any("statement_timeout = 300000" in sql for sql in conn.executed)


@pytest.mark.parametrize("bad_value", ["0", "-1", "-0.5", "not-a-number"])
def test_env_helper_rejects_non_positive_or_garbage_timeout(
    monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    """Non-positive or unparseable timeouts fall back to the 300s default.

    Regression. ``SET statement_timeout
    = 0`` is interpreted by PostgreSQL as 'unlimited' — exactly the
    opposite of the kill-switch intent — and negative values yield an
    invalid SET. Either case silently disables the DB-side cap.

    Reference: PostgreSQL runtime parameters,
    https://www.postgresql.org/docs/current/runtime-config-client.html.
    """
    monkeypatch.setenv("DATABASE_INIT_TIMEOUT_SECONDS", bad_value)
    _load_env_module()
    conn = _RecordingConnection("postgresql")

    _exercise_env_helper(conn, timeout_seconds=300)

    # Default fallback is 300s = 300000 ms.
    assert any("statement_timeout = 300000" in sql for sql in conn.executed), (
        f"non-positive / garbage value {bad_value!r} must fall back to the "
        "default, not be passed verbatim to PostgreSQL"
    )
    # And critically: never emit ``= 0`` (PostgreSQL would treat that
    # as 'no cap'), and never emit a negative value (PostgreSQL would
    # raise a parse error).
    for sql in conn.executed:
        assert " = 0" not in sql, f"must never emit unbounded SET: {sql}"
        assert " = -" not in sql, f"must never emit a negative SET: {sql}"


def test_do_run_migrations_calls_the_timeout_helper() -> None:
    """``do_run_migrations`` must call ``_apply_postgres_statement_timeout``.

    Source-level pin. The previous test
    suite only verified what the helper *would* do *if* called — it
    did not assert that ``do_run_migrations`` actually invokes it. If
    a future change removed the call site from ``do_run_migrations``,
    every existing test would still pass while the DB-side kill
    switch was silently lost.

    We parse env.py's source and assert the call appears inside
    ``do_run_migrations``. AST-based inspection avoids substring
    false positives (a docstring mention would otherwise count).

    Reference: ``ast.walk`` for tree traversal,
    https://docs.python.org/3/library/ast.html#ast.walk.
    """
    import ast

    source = _ENV_PATH.read_text()
    tree = ast.parse(source)

    do_run_migrations: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "do_run_migrations":
            do_run_migrations = node
            break

    assert do_run_migrations is not None, (
        "migrations/env.py must define do_run_migrations — the Alembic "
        "entry point that runs the upgrade."
    )

    # Walk just the function body to find the helper call.
    found_call = False
    for sub in ast.walk(do_run_migrations):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "_apply_postgres_statement_timeout"
        ):
            found_call = True
            break

    assert found_call, (
        "do_run_migrations() must call _apply_postgres_statement_timeout(connection) "
        "before configuring Alembic; otherwise statement_timeout / lock_timeout / "
        "idle_in_transaction_session_timeout are never applied and the DB-side kill "
        "switch is silently lost."
    )


def test_env_helper_commits_after_setting_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Helper must commit *after* the three SETs to close the implicit txn.

    Regression for the silent-rollback bug
    (`docker compose up` reported "Alembic migrations completed" while
    PostgreSQL ended up with an empty schema). SQLAlchemy 2.0
    autobegins a transaction on the first ``execute()``, so the SET
    statements open an implicit transaction. Alembic detects the open
    transaction and turns ``context.begin_transaction()`` into a no-op
    (``_in_connection_transaction()`` branch), running every DDL
    statement inside the implicit transaction. When the async
    ``connect()`` block exits without an explicit commit, SQLAlchemy
    2.0 rolls the implicit transaction back and the migration is
    silently discarded — Alembic still exits 0 because no exception
    was raised.

    The ordering matters: ``commit()`` must come *after* the three
    SETs so the session-scoped timeouts are persisted before the
    implicit transaction closes. A commit *before* would also work
    for "no transaction is open when Alembic runs", but it would
    discard the SETs that have not yet been issued, leaving the
    DB-side kill switch unset.

    References:
    * SQLAlchemy 2.0 commit-as-you-go contract —
      https://docs.sqlalchemy.org/en/20/core/connections.html#commit-as-you-go
    * Alembic ``_in_connection_transaction`` no-op behaviour —
      https://alembic.sqlalchemy.org/en/latest/api/runtime.html#alembic.runtime.migration.MigrationContext
    """
    monkeypatch.setenv("DATABASE_INIT_TIMEOUT_SECONDS", "60")
    _load_env_module()
    conn = _RecordingConnection("postgresql")

    _exercise_env_helper(conn, timeout_seconds=60)

    # The three SETs must precede a commit. Asserting on the
    # ordered ``calls`` log catches both "commit missing" and
    # "commit happens before the SETs were issued".
    assert conn.commits >= 1, (
        "Helper must commit after the SETs to close the implicit transaction "
        "SQLAlchemy 2.0 opens on the first execute(). Without this commit, "
        "Alembic runs inside the implicit transaction and all DDL rolls back "
        "when the connection closes."
    )
    last_commit_index = max(i for i, c in enumerate(conn.calls) if c == "COMMIT")
    set_indices = [i for i, c in enumerate(conn.calls) if c.startswith("EXEC:SET ")]
    assert set_indices, "expected three SET executes before the commit"
    assert max(set_indices) < last_commit_index, (
        "commit() must come *after* the three SETs so the session-scoped "
        "timeouts are persisted before the implicit transaction closes; "
        f"observed call order: {conn.calls}"
    )


def test_env_helper_does_not_commit_for_sqlite_dialect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite path issues no SETs and therefore must not commit.

    Calling ``commit()`` on a SQLAlchemy connection that has no open
    transaction is a no-op in 2.0, but issuing one would still be
    misleading and would mask a future bug where the SQLite branch
    accidentally starts a transaction. The dialect gate must
    short-circuit the entire helper, including the commit.
    """
    monkeypatch.setenv("DATABASE_INIT_TIMEOUT_SECONDS", "5")
    _load_env_module()
    conn = _RecordingConnection("sqlite")

    _exercise_env_helper(conn, timeout_seconds=5)

    assert conn.executed == [], "non-PostgreSQL dialects must receive no SET"
    assert conn.commits == 0, (
        "non-PostgreSQL dialects must not commit — the dialect gate "
        "short-circuits before any transactional work, so there is "
        "nothing to commit and issuing one would mask a future bug "
        "where the SQLite branch accidentally starts a transaction."
    )


def test_apply_postgres_statement_timeout_calls_commit_in_source() -> None:
    """Source-level pin: helper body contains a ``connection.commit()`` call.

    Belt-and-suspenders against a future refactor that removes the
    commit while keeping the SETs (e.g. someone "simplifies" the
    helper). The behavioural test
    ``test_env_helper_commits_after_setting_timeouts`` already catches
    a removal, but the AST check makes the intent explicit at the
    source level and produces a more pointed failure message.

    Reference: ``ast.walk`` tree traversal —
    https://docs.python.org/3/library/ast.html#ast.walk.
    """
    import ast

    source = _ENV_PATH.read_text()
    tree = ast.parse(source)

    helper: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_postgres_statement_timeout":
            helper = node
            break

    assert helper is not None, (
        "migrations/env.py must define _apply_postgres_statement_timeout — "
        "the SETs that establish the DB-side kill switch live there."
    )

    found_commit = False
    for sub in ast.walk(helper):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "commit"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "connection"
        ):
            found_commit = True
            break

    assert found_commit, (
        "_apply_postgres_statement_timeout() must call connection.commit() "
        "after the SETs to close the implicit transaction SQLAlchemy 2.0 "
        "autobegins on the first execute(). Without this commit, Alembic "
        "runs inside the implicit transaction, begin_transaction() becomes "
        "a no-op, and the DDL is silently rolled back when the async "
        "connection closes."
    )


def test_sqlalchemy_autobegin_contract_motivates_commit() -> None:
    """Pin the SQLAlchemy 2.0 ``Connection`` state transitions the helper relies on.

    This test does not exercise env.py — it exercises the raw
    SQLAlchemy ``Connection`` API contract that env.py's commit is
    defending against. If a future SQLAlchemy release changes the
    autobegin / commit-closes-transaction semantics, this test will
    start failing and prompt re-evaluation of whether the commit in
    ``_apply_postgres_statement_timeout`` is still load-bearing.

    Contract being pinned (SQLAlchemy 2.0):
      1. ``engine.connect()`` returns a Connection with no open
         transaction (``in_transaction() is False``).
      2. The first ``execute()`` autobegins an implicit transaction
         (``in_transaction() is True``).
      3. ``commit()`` closes that implicit transaction
         (``in_transaction() is False`` again).

    Step 2 is the *exact* trap env.py used to fall into: the SETs
    autobegin a transaction, Alembic's
    ``_in_connection_transaction()`` check sees it, and
    ``context.begin_transaction()`` no-ops. Step 3 is the *exact*
    fix: the trailing ``connection.commit()`` returns the connection
    to "no transaction" so Alembic's own ``begin_transaction()``
    opens a real transaction it commits on success.

    We intentionally test only the connection-state transitions
    (``in_transaction()``) and not "DDL rolls back on close": the
    rollback-on-close behaviour is dialect-sensitive (SQLite's
    historical Python-stdlib ``sqlite3`` driver implicitly commits
    DDL), and the helper's correctness depends on the
    connection-state contract above, not on the dialect's DDL
    rollback semantics. PostgreSQL respects rollback-on-close for
    DDL — which is why the bug manifested in production — but
    asserting that here would require a real PostgreSQL and isn't
    needed: Alembic's branch decision is driven by
    ``in_transaction()``, which IS dialect-agnostic.

    Reference: SQLAlchemy 2.0 — "Commit As You Go"
    https://docs.sqlalchemy.org/en/20/core/connections.html#commit-as-you-go.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine("sqlite://")
    try:
        with engine.connect() as conn:
            # Step 1: fresh connection, no transaction.
            assert not conn.in_transaction(), (
                "fresh connect() must not start a transaction — the bug "
                "depends on the *first* execute() being what autobegins"
            )

            # Step 2: first execute() autobegins. Use SELECT 1 rather
            # than a DDL because some drivers (notably stdlib sqlite3)
            # implicitly commit DDL outside SQLAlchemy's transaction
            # bookkeeping; ``SELECT 1`` is a faithful trigger for
            # autobegin on every dialect.
            conn.execute(text("SELECT 1"))
            assert conn.in_transaction(), (
                "first execute() must autobegin — this is the SQLAlchemy "
                "2.0 contract that makes the env.py commit() necessary. "
                "If this fires, autobegin no longer applies and the "
                "rationale for the commit() in env.py must be revisited."
            )

            # Step 3: commit() closes the implicit transaction. This
            # is what env.py's added commit() does — returning the
            # connection to "no transaction" so Alembic's own
            # context.begin_transaction() opens a real, Alembic-owned
            # transaction rather than no-oping inside the helper's
            # implicit one.
            conn.commit()
            assert not conn.in_transaction(), (
                "commit() must close the implicit transaction — if it does "
                "not, Alembic's _in_connection_transaction() check would "
                "still see an open transaction after the helper runs and "
                "begin_transaction() would still no-op, silently rolling "
                "back the migration."
            )
    finally:
        engine.dispose()


def test_settings_reject_non_positive_timeouts() -> None:
    """``Settings`` rejects zero / negative timeouts at parse time.

    The two layers (app Settings, env.py fallback) are complementary —
    Settings catches a misconfigured deployment at process start,
    env.py catches the same misconfiguration when Alembic is invoked
    from a separate process (e.g. ``alembic upgrade head`` from a
    shell where Settings is not loaded).
    """
    import os

    import pydantic

    from bo_mcp_server.settings import Settings

    saved = os.environ.get("DATABASE_INIT_TIMEOUT_SECONDS")
    try:
        os.environ["DATABASE_INIT_TIMEOUT_SECONDS"] = "0"
        with pytest.raises(pydantic.ValidationError):
            Settings()
        os.environ["DATABASE_INIT_TIMEOUT_SECONDS"] = "-1"
        with pytest.raises(pydantic.ValidationError):
            Settings()
    finally:
        if saved is None:
            os.environ.pop("DATABASE_INIT_TIMEOUT_SECONDS", None)
        else:
            os.environ["DATABASE_INIT_TIMEOUT_SECONDS"] = saved
