"""Centralized configuration for bo-mcp-server.

Every environment-driven knob that previously lived as ``os.getenv(...)``
spread across modules is consolidated here so a single place authoritatively
documents the operational surface area. Modules still import the resolved
values through small adapter functions
(:func:`get_database_url`, :func:`get_default_backend_name`, …) so test
overrides applied via the standard ``monkeypatch.setenv`` fixture are
observed without any cache plumbing — the accessors instantiate
:class:`Settings` on every call, which re-reads ``os.environ``.

Reference: pydantic-settings ``BaseSettings`` is the idiomatic way to
bind environment variables to a typed object — see
https://docs.pydantic.dev/latest/concepts/pydantic_settings/.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Operational configuration sourced from environment variables.

    All knobs are optional and have safe defaults so local / SQLite
    development works without any env vars set.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/bo_mcp.db",
        alias="DATABASE_URL",
        description="SQLAlchemy async URL. SQLite is used for local + tests; "
        "production deployments override this with a PostgreSQL URL.",
    )
    use_alembic: Literal["auto", "true", "false"] = Field(
        default="auto",
        alias="USE_ALEMBIC",
        description="Schema bootstrap strategy. 'auto' uses Alembic for PostgreSQL "
        "and direct create_all for SQLite; explicit values force one path.",
    )
    sql_echo: bool = Field(
        default=False,
        alias="SQL_ECHO",
        description="If true, SQLAlchemy logs every emitted statement.",
    )
    bo_backend: str = Field(
        default="botorch",
        alias="BO_BACKEND",
        description="Default BO backend name when a campaign requests 'auto'.",
    )


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` instance.

    Each call re-reads ``os.environ`` so test overrides applied via
    ``monkeypatch.setenv`` take effect immediately. The cost is a
    Pydantic validation per call; if this ever becomes hot, switch the
    callers that don't need the dynamic semantics to module-level
    constants instead of caching the instance.
    """
    return Settings()


def get_database_url() -> str:
    """Convenience accessor used by the storage layer."""
    return get_settings().database_url


def get_use_alembic_mode() -> str:
    """Return the raw alembic-mode string (``auto`` / ``true`` / ``false``)."""
    return get_settings().use_alembic


def get_sql_echo() -> bool:
    """Return whether SQLAlchemy should log emitted statements."""
    return get_settings().sql_echo


def get_default_backend_name() -> str:
    """Return the configured default backend name."""
    return get_settings().bo_backend
