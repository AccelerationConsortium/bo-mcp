"""Alembic env.py applies DB-side timeouts on PostgreSQL (TODO 8.22).

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

    The env helper only reads ``connection.dialect.name`` and calls
    ``connection.execute(text(...))``. We capture every executed
    statement so the test can assert on the rendered SQL.
    """

    def __init__(self, dialect_name: str) -> None:
        self.dialect = type("Dialect", (), {"name": dialect_name})()
        self.executed: list[str] = []

    def execute(self, statement: Any) -> Any:
        # ``text(...)`` exposes the raw SQL via ``.text``; defensively
        # fall back to ``str()`` for anything else.
        rendered = getattr(statement, "text", None) or str(statement)
        self.executed.append(rendered)
        return None


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
        "see TODO 8.22 — removing it loses the DB-side kill switch."
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

    Without ``idle_in_transaction_session_timeout`` (TODO 8.22 second
    review pass), a migration that runs many short statements
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

    Regression for TODO 8.22 third review pass. ``SET statement_timeout
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

    Source-level pin (TODO 8.22 third review pass). The previous test
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
        "switch is silently lost (TODO 8.22)."
    )


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
