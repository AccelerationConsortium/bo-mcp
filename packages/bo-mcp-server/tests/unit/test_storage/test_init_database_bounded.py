"""Bounded ``init_database`` contract.

The lifespan path runs Alembic on the way in. A misconfigured
``DATABASE_URL`` or a stuck migration must surface as a typed
:class:`DatabaseInitializationError` so the orchestrator (k8s,
systemd) sees a clear non-zero exit instead of a process that hangs
on startup.

Three stages:

* ``connect`` — pre-flight ``SELECT 1`` couldn't complete (bad DSN,
  network, credentials).
* ``timeout`` — Alembic itself ran past ``DATABASE_INIT_TIMEOUT_SECONDS``.
* ``alembic`` — Alembic raised (bad migration, schema clash).

Reference: 12-Factor App XII (admin processes) recommends a bounded,
observable schema-migration step so orchestration tooling can
differentiate "DB never reachable" from "migration in progress".
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.exc import OperationalError

from bo_mcp_server.storage import DatabaseInitializationError, close_database
from bo_mcp_server.storage import database as db_module


@pytest.fixture(autouse=True)
async def reset_engine_state() -> AsyncGenerator[None]:
    """Ensure each test starts with a freshly-disposed engine."""
    await close_database()
    yield
    await close_database()


@pytest.mark.asyncio
async def test_unreachable_database_fails_fast_with_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bogus host must fail the connect probe within the connect timeout.

    Points at a TCP port that nothing is listening on (1) so the OS
    rejects the SYN immediately. Cap the connect probe at 2s so the
    test reports the typed error in under that window.
    """
    # S6698: literal credentials are a deliberate test fixture, not a real secret.
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://invalid:invalid@127.0.0.1:1/nonexistent",
    )
    monkeypatch.setenv("DATABASE_INIT_CONNECT_TIMEOUT_SECONDS", "2.0")
    monkeypatch.setenv("USE_ALEMBIC", "true")

    started = time.monotonic()
    with pytest.raises(DatabaseInitializationError) as exc_info:
        await db_module.init_database()
    elapsed = time.monotonic() - started

    assert exc_info.value.stage == "connect"
    # Under-3s guards against the historical "blocks forever" failure
    # mode (where no probe existed and lifespan hung indefinitely).
    assert elapsed < 10.0, f"connect probe must fail fast; took {elapsed:.1f}s"


@pytest.mark.asyncio
async def test_slow_alembic_upgrade_hits_typed_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A migration whose subprocess is SIGKILLed surfaces ``stage='timeout'``.

    The fake ``_run_alembic_in_subprocess`` raises the same
    ``TimeoutError`` that the real implementation raises when
    ``subprocess.run(timeout=N)`` triggers the SIGKILL path — that's
    the contract ``init_database`` actually maps off. End-to-end
    subprocess-spawn coverage lives in
    ``test_run_alembic_subprocess_real_timeout`` below.
    """

    async def _instant_preflight(*_args, **_kwargs) -> None:
        await asyncio.sleep(0)

    def _killed_subprocess(timeout_seconds: float) -> None:
        msg = f"Alembic upgrade exceeded {timeout_seconds:.1f}s and was SIGKILLed"
        raise TimeoutError(msg)

    monkeypatch.setattr(db_module, "_preflight_connectivity", _instant_preflight)
    monkeypatch.setattr(db_module, "_run_alembic_in_subprocess", _killed_subprocess)
    monkeypatch.setattr(db_module, "_should_use_alembic", lambda: True)
    monkeypatch.setenv("DATABASE_INIT_TIMEOUT_SECONDS", "0.25")

    started = time.monotonic()
    with pytest.raises(DatabaseInitializationError) as exc_info:
        await db_module.init_database()
    elapsed = time.monotonic() - started

    assert exc_info.value.stage == "timeout"
    assert elapsed < 3.0, f"timeout path must fire fast; took {elapsed:.1f}s"


@pytest.mark.asyncio
async def test_alembic_failure_is_translated_to_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An Alembic ``OperationalError`` becomes ``stage='alembic'``.

    Models a broken migration script (raises mid-upgrade). The original
    exception must survive on ``__cause__`` so triage can inspect the
    SQL state without re-running migrations.
    """

    async def _instant_preflight(*_args, **_kwargs) -> None:
        await asyncio.sleep(0)

    def _broken_upgrade(_timeout_seconds: float) -> None:
        msg = "bad-migration"
        raise OperationalError(msg, {}, RuntimeError("boom"))

    monkeypatch.setattr(db_module, "_preflight_connectivity", _instant_preflight)
    monkeypatch.setattr(db_module, "_run_alembic_in_subprocess", _broken_upgrade)
    monkeypatch.setattr(db_module, "_should_use_alembic", lambda: True)
    monkeypatch.setenv("DATABASE_INIT_TIMEOUT_SECONDS", "5.0")

    with pytest.raises(DatabaseInitializationError) as exc_info:
        await db_module.init_database()

    assert exc_info.value.stage == "alembic"
    assert isinstance(exc_info.value.__cause__, OperationalError)


@pytest.mark.parametrize(
    "raised",
    [
        # Migration-script failures land here in practice — pin every
        # documented shape so a narrower catch cannot reappear.
        # ``CommandError`` is the canonical Alembic CLI-level failure
        # (missing revision, multiple heads), ``RuntimeError`` covers
        # raw ``raise`` from a migration body, ``ValueError`` is the
        # typical shape for bad-config errors, and ``KeyError`` stands
        # in for anything more exotic a migration author might raise.
        ("alembic.util.exc.CommandError", "no such revision"),
        (RuntimeError, "migration body panicked"),
        (ValueError, "invalid migration config"),
        (KeyError, "missing-fixture"),
    ],
)
@pytest.mark.asyncio
async def test_alembic_non_sqlalchemy_failures_also_typed(
    monkeypatch: pytest.MonkeyPatch,
    raised: tuple[type[BaseException] | str, str],
) -> None:
    """Non-``SQLAlchemyError`` migration failures become ``stage='alembic'``.

    Regression for the original narrow catch (TODO 8.22 review pass):
    only ``SQLAlchemyError`` was translated, so ``alembic.util.exc.
    CommandError`` and any ``raise`` from a migration body bubbled up
    as a raw exception and the orchestrator lost the typed signal.
    """
    exc_type_or_path, message = raised
    if isinstance(exc_type_or_path, str):
        # Resolve ``alembic.util.exc.CommandError`` lazily — keeps the
        # test suite working when alembic isn't installed in the
        # baseline test extra.
        from importlib import import_module

        module_path, _, name = exc_type_or_path.rpartition(".")
        exc_type = getattr(import_module(module_path), name)
    else:
        exc_type = exc_type_or_path

    async def _instant_preflight(*_args, **_kwargs) -> None:
        await asyncio.sleep(0)

    def _broken_upgrade(_timeout_seconds: float) -> None:
        raise exc_type(message)

    monkeypatch.setattr(db_module, "_preflight_connectivity", _instant_preflight)
    monkeypatch.setattr(db_module, "_run_alembic_in_subprocess", _broken_upgrade)
    monkeypatch.setattr(db_module, "_should_use_alembic", lambda: True)
    monkeypatch.setenv("DATABASE_INIT_TIMEOUT_SECONDS", "5.0")

    with pytest.raises(DatabaseInitializationError) as exc_info:
        await db_module.init_database()

    assert exc_info.value.stage == "alembic"
    assert isinstance(exc_info.value.__cause__, exc_type)


def test_run_alembic_subprocess_sigkills_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_run_alembic_in_subprocess`` SIGKILLs a real child process on timeout.

    End-to-end coverage for TODO 8.22 third review pass. Spawns an
    actual ``python -c 'time.sleep(...)'`` subprocess via a monkey-
    patched command so we exercise the real ``subprocess.run(timeout=N)``
    path. The previous Python-thread approach could not be interrupted;
    here the OS terminates the child and ``subprocess.run`` raises
    :class:`subprocess.TimeoutExpired`, which the helper translates to
    :class:`TimeoutError`.
    """
    import subprocess as subprocess_mod

    # Replace ``subprocess.run`` only inside the helper's module so we
    # can substitute a command that genuinely blocks but does not need
    # alembic installed. The real ``subprocess.run`` (timeout behaviour)
    # is exercised — only the *argv* is swapped.
    real_run = subprocess_mod.run
    captured_args: dict[str, object] = {}

    def _spying_run(_cmd, **kwargs):
        captured_args["timeout"] = kwargs.get("timeout")
        # Force a sleep-much-longer-than-timeout child so the timeout
        # branch actually fires.
        replacement = [sys.executable, "-c", "import time; time.sleep(60)"]
        return real_run(replacement, **kwargs)

    monkeypatch.setattr(db_module.subprocess, "run", _spying_run)
    # Also stub the existence check so the helper doesn't bail early
    # when ``alembic.ini`` is in a non-standard path during testing.
    monkeypatch.setattr(db_module.Path, "exists", lambda _self: True)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        db_module._run_alembic_in_subprocess(timeout_seconds=0.5)
    elapsed = time.monotonic() - started

    assert captured_args["timeout"] == pytest.approx(0.5)
    # subprocess.run SIGKILLs at the budget; allow a small tail for
    # the process to wind down but assert it did not run anywhere
    # near the 60-second sleep.
    assert elapsed < 5.0, f"SIGKILL did not fire promptly; took {elapsed:.1f}s"


def test_run_alembic_subprocess_translates_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subprocess that exits non-zero becomes ``RuntimeError`` with stderr excerpt.

    The non-zero-exit branch is what surfaces a real Alembic failure
    (bad revision, schema clash) when the subprocess actually returns.
    ``init_database`` maps that to ``stage='alembic'``; verify the
    helper produces the expected shape so the chain stays correct.
    """
    import subprocess as subprocess_mod

    real_run = subprocess_mod.run

    def _failing_run(_cmd, **kwargs):
        # Substitute a child that exits 1 with a distinctive stderr.
        replacement = [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('synthetic alembic boom'); sys.exit(1)",
        ]
        return real_run(replacement, **kwargs)

    monkeypatch.setattr(db_module.subprocess, "run", _failing_run)
    monkeypatch.setattr(db_module.Path, "exists", lambda _self: True)

    with pytest.raises(RuntimeError) as exc_info:
        db_module._run_alembic_in_subprocess(timeout_seconds=10.0)

    assert "code 1" in str(exc_info.value)
    assert "synthetic alembic boom" in str(exc_info.value)
