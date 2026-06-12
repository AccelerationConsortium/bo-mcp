"""Tests for the consolidated :mod:`bo_mcp_server.settings` module.

Reference: the pydantic-settings ``BaseSettings`` pattern is documented at
https://docs.pydantic.dev/latest/concepts/pydantic_settings/ and is the
official replacement for the previous ``BaseSettings`` that lived in
pydantic core (now removed in v2).
"""

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


def test_default_values_when_env_unset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """All knobs fall back to documented defaults when nothing is set."""
    for key in ("DATABASE_URL", "USE_ALEMBIC", "SQL_ECHO", "BO_BACKEND"):
        monkeypatch.delenv(key, raising=False)
    # Point to a fresh empty env file so a developer-local ``.env`` does not
    # leak overrides into this assertion (pydantic-settings parameterizes the
    # ``env_file`` via ``model_config``).
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.database_url.get_secret_value().startswith("sqlite+aiosqlite://")
    assert settings.use_alembic == "auto"
    assert settings.sql_echo is False
    assert settings.bo_backend == "botorch"


def test_env_override_is_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    """``monkeypatch.setenv`` mutations are picked up by the accessors."""
    monkeypatch.setenv("BO_BACKEND", "baybe")
    monkeypatch.setenv("SQL_ECHO", "true")
    assert get_default_backend_name() == "baybe"
    assert get_sql_echo() is True


def test_compute_timeout_default_is_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The compute timeout is on by default so a hung backend is bounded out-of-the-box.

    The M36 wedged-backend protection must not depend on every deployment
    setting the env var; the default is a conservative 30-minute hang
    detector. ``0`` remains the explicit opt-out.
    """
    monkeypatch.delenv("BO_COMPUTE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.chdir(tmp_path)
    assert Settings().bo_compute_timeout_seconds == 30 * 60
    assert get_bo_compute_timeout_seconds() == 30 * 60

    monkeypatch.setenv("BO_COMPUTE_TIMEOUT_SECONDS", "0")
    assert get_bo_compute_timeout_seconds() == 0.0


def test_heartbeat_extension_cap_default_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The reservation-heartbeat extension is capped by default (30 minutes).

    Without a finite cap a wedged backend could hold a reservation slot
    indefinitely via the heartbeat; the default bounds it.
    """
    monkeypatch.delenv("IDEMPOTENCY_HEARTBEAT_MAX_TOTAL_EXTENSION_SECONDS", raising=False)
    monkeypatch.chdir(tmp_path)
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
