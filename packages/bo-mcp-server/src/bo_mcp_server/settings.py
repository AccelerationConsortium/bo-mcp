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
    audit_failures_fatal: bool = Field(
        default=False,
        alias="AUDIT_FAILURES_FATAL",
        description=(
            "If true, an audit-event persistence failure raises and the parent "
            "tool call returns an error envelope instead of completing silently. "
            "Compliance deployments flip this on so an audit gap cannot be hidden "
            "behind a misleading success response."
        ),
    )
    database_init_timeout_seconds: float = Field(
        default=300.0,
        gt=0.0,
        alias="DATABASE_INIT_TIMEOUT_SECONDS",
        description=(
            "Hard upper bound on Alembic schema-upgrade duration at startup. "
            "If exceeded, init_database() raises DatabaseInitializationError so "
            "the orchestrator (k8s, systemd) sees a clear non-zero exit instead "
            "of a process that silently blocks lifespan forever. Must be > 0: "
            "PostgreSQL's statement_timeout treats 0 as 'disabled' and asyncio."
            "wait_for(..., timeout=0) trips immediately, so a non-positive value "
            "would silently break both the orchestrator-side and the DB-side "
            "kill switches simultaneously."
        ),
    )
    database_init_connect_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        alias="DATABASE_INIT_CONNECT_TIMEOUT_SECONDS",
        description=(
            "Hard upper bound on the pre-Alembic connectivity probe. A network "
            "or credentials misconfiguration fails fast under this cap rather "
            "than burning the full migration timeout on a connection that will "
            "never succeed. Must be > 0."
        ),
    )
    idempotency_cache_gc_interval_seconds: float = Field(
        default=3600.0,
        ge=0.0,
        alias="IDEMPOTENCY_CACHE_GC_INTERVAL_SECONDS",
        description=(
            "Interval at which the lifespan-managed background task purges "
            "expired idempotency_cache rows. The opportunistic per-key purge in "
            "_read_existing only fires when that key is queried; rows for never-"
            "retried calls accumulate until this sweep runs. Set to 0 to disable "
            "the sweep entirely (test fixtures or deployments that already run "
            "pg_cron)."
        ),
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


def get_audit_failures_fatal() -> bool:
    """Return whether audit failures should fail the parent tool call."""
    return get_settings().audit_failures_fatal


def get_database_init_timeout_seconds() -> float:
    """Return the Alembic-upgrade timeout cap (seconds)."""
    return get_settings().database_init_timeout_seconds


def get_database_init_connect_timeout_seconds() -> float:
    """Return the pre-Alembic connectivity-probe timeout cap (seconds)."""
    return get_settings().database_init_connect_timeout_seconds


def get_idempotency_cache_gc_interval_seconds() -> float:
    """Return the idempotency-cache GC sweep interval (seconds).

    Zero disables the lifespan sweep; the opportunistic per-key purge in
    :func:`bo_mcp_server.idempotency._read_existing` still trims rows
    that are actually queried.
    """
    return get_settings().idempotency_cache_gc_interval_seconds
