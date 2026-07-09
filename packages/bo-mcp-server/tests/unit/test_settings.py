"""Tests for the consolidated :mod:`bo_mcp_server.settings` module.

Reference: the pydantic-settings ``BaseSettings`` pattern is documented at
https://docs.pydantic.dev/latest/concepts/pydantic_settings/ and is the
official replacement for the previous ``BaseSettings`` that lived in
pydantic core (now removed in v2).
"""

from pathlib import Path
from typing import Any, cast

import pytest

from bo_mcp_server.settings import (
    Settings,
    get_bo_compute_timeout_seconds,
    get_database_url,
    get_default_backend_name,
    get_idempotency_heartbeat_max_total_extension_seconds,
    get_settings,
    get_sql_echo,
    get_use_alembic_mode,
)


@pytest.fixture
def no_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detach the anchored ``.env`` candidates for hermetic default tests.

    The ``env_file`` anchors are absolute paths (CWD-independent by
    design), so ``monkeypatch.chdir`` no longer isolates a test from a
    developer's repo-root ``.env``. Tests that assert *documented
    defaults* after ``delenv`` clear the source explicitly instead.
    """
    # ``model_config`` is a TypedDict; cast to a plain mapping so the
    # ``None`` sentinel (meaning "no env file") type-checks.
    monkeypatch.setitem(cast("dict[str, Any]", Settings.model_config), "env_file", None)


@pytest.mark.usefixtures("no_env_file")
def test_default_values_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """All knobs fall back to documented defaults when nothing is set."""
    for key in ("DATABASE_URL", "USE_ALEMBIC", "SQL_ECHO", "BO_BACKEND"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings()
    assert settings.database_url.get_secret_value().startswith("sqlite+aiosqlite://")
    assert settings.use_alembic == "auto"
    assert settings.sql_echo is False
    assert settings.bo_backend == "baybe"


@pytest.mark.usefixtures("no_env_file")
def test_default_database_url_is_launch_context_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The default SQLite URL resolves the same absolute file from any CWD.

    MCP clients spawn stdio servers from arbitrary working directories;
    a CWD-relative default would silently select a different database
    per launch context ("my campaigns disappeared") or fail with
    ``PermissionError`` when the CWD is not writable. The default must
    therefore be an absolute path anchored at the package.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)

    cwd_a = tmp_path / "launch_a"
    cwd_b = tmp_path / "launch_b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    monkeypatch.chdir(cwd_a)
    url_from_a = get_database_url()
    monkeypatch.chdir(cwd_b)
    url_from_b = get_database_url()

    assert url_from_a == url_from_b
    db_path = url_from_a.removeprefix("sqlite+aiosqlite:///")
    assert db_path.startswith("/"), f"default SQLite path is not absolute: {db_path!r}"


def test_source_tree_anchor_matches_source_layout() -> None:
    """Running from the checkout, the anchor lands on this project's root.

    ``_source_project_root`` derives ``parents[2]`` of ``settings.py``,
    which encodes the src layout
    (``<project>/src/bo_mcp_server/settings.py``). If the module is ever
    moved (or the layout flattened), the default database path and the
    ``.env`` anchors would silently point somewhere else — this pins the
    assumption so such a refactor fails loudly.
    """
    from bo_mcp_server import settings as settings_module

    project_root = settings_module._source_project_root(Path(settings_module.__file__))
    assert project_root is not None, "src-layout assumption broken — see _source_project_root"
    assert 'name = "bo-mcp-server"' in (project_root / "pyproject.toml").read_text()
    assert settings_module._DEFAULT_DATA_DIR == project_root / "data"


def test_installed_wheel_defaults_to_user_data_dir(tmp_path) -> None:
    """A wheel-installed package must not default the DB under the venv lib dir.

    Simulates the installed layout
    (``…/lib/python3.13/site-packages/bo_mcp_server/settings.py``): no
    ``pyproject.toml`` sits two levels up, so the default database goes
    to the OS user data directory (platformdirs) instead of an
    interpreter-internal path, and no ambient ``.env`` candidates are
    honored — installed deployments configure via real environment
    variables.

    Reference: platformdirs is the maintained successor of appdirs for
    OS-appropriate per-user data locations —
    https://platformdirs.readthedocs.io/en/latest/.
    """
    import platformdirs

    from bo_mcp_server.settings import _default_data_dir, _env_file_candidates

    fake_wheel_settings = (
        tmp_path / "venv" / "lib" / "python3.13" / "site-packages" / "bo_mcp_server" / "settings.py"
    )
    fake_wheel_settings.parent.mkdir(parents=True)
    fake_wheel_settings.touch()

    data_dir = _default_data_dir(fake_wheel_settings)
    assert data_dir == Path(platformdirs.user_data_dir(appname="bo-mcp-server"))
    assert not data_dir.is_relative_to(tmp_path), (
        f"default data dir {data_dir} must not live inside the interpreter tree"
    )
    assert _env_file_candidates(fake_wheel_settings) == ()


def test_source_tree_layout_resolves_project_paths(tmp_path) -> None:
    """A simulated src-layout checkout anchors data and .env at the project."""
    from bo_mcp_server.settings import _default_data_dir, _env_file_candidates

    project = tmp_path / "repo" / "packages" / "bo-mcp-server"
    settings_file = project / "src" / "bo_mcp_server" / "settings.py"
    settings_file.parent.mkdir(parents=True)
    settings_file.touch()
    (project / "pyproject.toml").write_text('[project]\nname = "bo-mcp-server"\n')

    assert _default_data_dir(settings_file) == project / "data"
    assert _env_file_candidates(settings_file) == (
        str(tmp_path / "repo" / ".env"),
        str(project / ".env"),
    )


def test_foreign_env_file_in_launch_cwd_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A ``.env`` sitting in the launch CWD must not repoint the database.

    ``.env`` discovery is anchored to the package/repo roots (absolute
    paths), so a foreign file in whatever directory the MCP client
    happens to spawn the server from is never read — while an explicit
    environment variable keeps its priority over every ``.env`` source.
    """
    foreign_cwd = tmp_path / "foreign_launch"
    neutral_cwd = tmp_path / "neutral_launch"
    foreign_cwd.mkdir()
    neutral_cwd.mkdir()
    (foreign_cwd / ".env").write_text("DATABASE_URL=sqlite+aiosqlite:///stolen/foreign.db\n")

    monkeypatch.delenv("DATABASE_URL", raising=False)

    monkeypatch.chdir(neutral_cwd)
    url_from_neutral = get_database_url()
    monkeypatch.chdir(foreign_cwd)
    url_from_foreign = get_database_url()

    assert "foreign.db" not in url_from_foreign
    assert url_from_foreign == url_from_neutral

    # Explicit environment variables still outrank any .env candidate.
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://explicit@db.host/bo")
    assert get_database_url() == "postgresql+asyncpg://explicit@db.host/bo"


def test_env_override_is_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    """``monkeypatch.setenv`` mutations are picked up by the accessors."""
    monkeypatch.setenv("BO_BACKEND", "baybe")
    monkeypatch.setenv("SQL_ECHO", "true")
    assert get_default_backend_name() == "baybe"
    assert get_sql_echo() is True


@pytest.mark.usefixtures("no_env_file")
def test_compute_timeout_default_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compute timeout is on by default so a hung backend is bounded out-of-the-box.

    The M36 wedged-backend protection must not depend on every deployment
    setting the env var; the default is a conservative 30-minute hang
    detector. ``0`` remains the explicit opt-out.
    """
    monkeypatch.delenv("BO_COMPUTE_TIMEOUT_SECONDS", raising=False)
    assert Settings().bo_compute_timeout_seconds == 30 * 60
    assert get_bo_compute_timeout_seconds() == 30 * 60

    monkeypatch.setenv("BO_COMPUTE_TIMEOUT_SECONDS", "0")
    assert get_bo_compute_timeout_seconds() == 0.0


@pytest.mark.usefixtures("no_env_file")
def test_heartbeat_extension_cap_default_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reservation-heartbeat extension is capped by default (30 minutes).

    Without a finite cap a wedged backend could hold a reservation slot
    indefinitely via the heartbeat; the default bounds it.
    """
    monkeypatch.delenv("IDEMPOTENCY_HEARTBEAT_MAX_TOTAL_EXTENSION_SECONDS", raising=False)
    assert get_idempotency_heartbeat_max_total_extension_seconds() == 30 * 60


def test_invalid_use_alembic_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typoed alembic-mode values raise instead of being silently coerced."""
    monkeypatch.setenv("USE_ALEMBIC", "maybe")
    with pytest.raises(ValueError, match="USE_ALEMBIC"):
        Settings()


def test_accessors_return_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The thin accessors expose the typed Settings fields as their declared scalars."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x@y/z")
    monkeypatch.setenv("USE_ALEMBIC", "true")
    assert get_database_url() == "postgresql+asyncpg://x@y/z"
    assert get_use_alembic_mode() == "true"


def test_get_settings_returns_fresh_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each call returns a fresh instance reflecting the latest env."""
    monkeypatch.setenv("BO_BACKEND", "first")
    first = get_settings()
    monkeypatch.setenv("BO_BACKEND", "second")
    second = get_settings()
    assert first.bo_backend == "first"
    assert second.bo_backend == "second"


def test_repr_does_not_leak_database_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """``repr(Settings)`` masks the password component of a PostgreSQL URL.

    Pydantic v2's ``SecretStr`` is the documented way to keep sensitive
    fields out of ``repr`` / ``str`` output. Any incidental log line
    that prints a ``Settings`` instance — common in startup tracebacks
    or dashboard introspection — must not leak the username/password
    embedded in a ``postgresql://user:pass@host`` URL.

    Reference:
        https://docs.pydantic.dev/latest/api/types/#pydantic.types.SecretStr
    """
    import re

    fake_user = "fake-user"  # synthetic fixture — never matches a real account
    fake_pw = "fake-password-not-a-secret"
    url = f"postgresql+asyncpg://{fake_user}:{fake_pw}@db.host/bo"
    monkeypatch.setenv("DATABASE_URL", url)
    settings = Settings()

    rendered = repr(settings)
    assert fake_pw not in rendered
    assert fake_user not in rendered
    assert "SecretStr" in rendered
    # Belt-and-suspenders: no ``user:pass@host`` literal anywhere in the
    # repr, regardless of how the underlying library decides to surface
    # SecretStr.
    assert re.search(r"://[^/\s']+:[^/\s']+@", rendered) is None
    # The accessor still returns the raw URL for the storage layer.
    assert settings.database_url.get_secret_value() == url


def test_storage_engine_picks_up_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``monkeypatch.setenv`` reaches the engine factory at engine-create time.

    Before this fix the storage module captured ``DATABASE_URL`` at
    import time, so a test that overrode the env var via ``setenv``
    would never see it inside ``_create_engine_with_options``. The
    dynamic resolver now reads settings on every call so the override
    flows into ``create_async_engine``.
    """
    from bo_mcp_server.storage.database import _current_database_url

    # The module-level constant must keep its import-time value so any
    # callers that read it directly still see the original (the dynamic
    # accessor is what the engine factory consults now).
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:?override=1")
    assert _current_database_url().endswith("override=1")
