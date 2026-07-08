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

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Operational configuration sourced from environment variables.

    All knobs are optional and have safe defaults so local / SQLite
    development works without any env vars set.

    Sensitive fields (``database_url``) are wrapped in
    :class:`pydantic.SecretStr` so any incidental ``repr(settings)`` —
    log breadcrumb, error envelope, dashboard introspection — emits
    ``database_url=SecretStr('**********')`` instead of leaking the
    embedded credentials. Call ``get_secret_value()`` (or the
    ``get_database_url`` accessor) when the live value is needed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    database_url: SecretStr = Field(
        default=SecretStr("sqlite+aiosqlite:///./data/bo_mcp.db"),
        alias="DATABASE_URL",
        description="SQLAlchemy async URL. SQLite is used for local + tests; "
        "production deployments override this with a PostgreSQL URL. Wrapped "
        "in SecretStr so credentials embedded in postgresql://user:pass@... "
        "URLs are not leaked by accidental repr(settings) calls.",
    )
    api_env: Literal["development", "staging", "production"] = Field(
        default="development",
        alias="API_ENV",
        description="Deployment environment; production refuses to start with DEV_AUTH=1.",
    )
    dev_auth: bool = Field(
        default=False,
        alias="DEV_AUTH",
        description=(
            "When true, startup may bootstrap the shared development user so "
            "MCP tools can resolve a real owner without exposing user ids to agents. "
            "Refused when API_ENV=production."
        ),
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
        description="If true, SQLAlchemy logs every emitted statement. "
        "Payload hazard: statements include bind parameters, and the "
        "campaign-state column can carry multi-MB serialized backend "
        "state — enabling this on a busy deployment floods the logs and "
        "may leak campaign data into log storage. Debug use only.",
    )
    bo_backend: str = Field(
        default="baybe",
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
    db_pool_size: int = Field(
        default=5,
        ge=1,
        alias="DB_POOL_SIZE",
        description=(
            "Steady-state SQLAlchemy connection-pool size for the production "
            "PostgreSQL engine. Trade-off: too low and request bursts queue on "
            "checkout; too high and PgBouncer / Postgres back-ends churn. The "
            "default of 5 matches the SQLAlchemy stock guidance for a single "
            "process serving moderate concurrency; bump via env when running "
            "a larger process pool."
        ),
    )
    db_max_overflow: int = Field(
        default=10,
        ge=0,
        alias="DB_MAX_OVERFLOW",
        description=(
            "Burst capacity on top of ``db_pool_size``. Connections beyond "
            "the steady-state pool are recycled after the request returns "
            "them; the cap protects the DB from a stampede during a "
            "synchronous fan-out."
        ),
    )
    db_pool_recycle_seconds: int = Field(
        default=600,
        ge=1,
        alias="DB_POOL_RECYCLE_SECONDS",
        description=(
            "Idle-connection recycle horizon. Kept below the typical "
            "PostgreSQL/PgBouncer 1-hour idle timeout so SQLAlchemy retires "
            "stale connections proactively instead of surfacing 'server "
            "closed the connection unexpectedly' on the next checkout."
        ),
    )
    diagnostics_cache_max_entries: int = Field(
        default=200,
        ge=1,
        alias="DIAGNOSTICS_CACHE_MAX_ENTRIES",
        description=(
            "Upper bound on the in-process diagnostics cache. Version-aware "
            "cache keys auto-invalidate after a mutation, so the cap mainly "
            "prevents unbounded growth from a long-running process that "
            "serves many distinct campaigns."
        ),
    )
    diagnostics_cache_ttl_seconds: int = Field(
        default=120,
        ge=1,
        alias="DIAGNOSTICS_CACHE_TTL_SECONDS",
        description=(
            "TTL for diagnostics-cache entries. The version-keyed scheme "
            "means cache entries become unreachable on mutation, so a "
            "relatively long TTL is safe and improves hit rate for "
            "repeatedly-polled campaigns."
        ),
    )
    idempotency_reservation_ttl_seconds: int = Field(
        default=10 * 60,
        ge=1,
        alias="IDEMPOTENCY_RESERVATION_TTL_SECONDS",
        description=(
            "How long an in-flight idempotency reservation can stay 'pending' "
            "before a retry is allowed to reclaim the slot. Without this "
            "bound, a worker that dies between mutation-commit and cache-"
            "finalize would poison the slot for the entire 24h response TTL. "
            "Tune up only for intentionally-long operations and accept that a "
            "longer block trades against possible duplicate execution."
        ),
    )
    idempotency_response_ttl_seconds: int = Field(
        default=24 * 60 * 60,
        ge=1,
        alias="IDEMPOTENCY_RESPONSE_TTL_SECONDS",
        description=(
            "How long a finalized idempotency response is retained for replay. "
            "Matches the IETF idempotency-key draft's recommended 24-hour "
            "window — clients retrying within this window get the cached "
            "response; after expiry, the same key can be reused for a new "
            "request."
        ),
    )
    idempotency_heartbeat_max_total_extension_seconds: float = Field(
        default=30 * 60,
        ge=0,
        alias="IDEMPOTENCY_HEARTBEAT_MAX_TOTAL_EXTENSION_SECONDS",
        description=(
            "Maximum wall-clock runtime of a reservation heartbeat, measured "
            "from when the heartbeat starts. The heartbeat extends a slow "
            "operation's reservation so it is not reclaimed mid-run, but without "
            "a bound a wedged backend (a hung GP fit on an un-cancellable worker "
            "thread) would hold the slot indefinitely. After running this many "
            "seconds the heartbeat stops, letting the reservation expire so a "
            "retry can reclaim the slot. Counted as elapsed runtime (one beat "
            "period at a time), NOT as the sum of requested extensions — the "
            "monotonic ``max(current, now+extension)`` expiry means early beats "
            "match the row without moving the deadline, and charging those "
            "no-ops against the bound would stop the heartbeat long before a "
            "legitimate long compute finishes. Keep this at or above "
            "BO_COMPUTE_TIMEOUT_SECONDS so a legitimately slow run keeps its "
            "slot. ``0`` disables the bound (unbounded — legacy behaviour)."
        ),
    )
    bo_compute_timeout_seconds: float = Field(
        default=30 * 60,
        ge=0,
        alias="BO_COMPUTE_TIMEOUT_SECONDS",
        description=(
            "Hard wall-clock budget for a single backend compute (GP fit + "
            "acquisition optimization) offloaded to a worker thread. When the "
            "budget elapses the server stops awaiting the thread and returns a "
            "retryable BACKEND_TRANSIENT_ERROR envelope, freeing the request "
            "(the orphaned thread is not cancellable and finishes in the "
            "background — its result is discarded). The 30-minute default is a "
            "conservative hang-detector: it sits beyond any realistic single "
            "GP-fit + acquisition run (seconds to a few minutes, even for "
            "SAASBO MCMC / large batches) so it only fires on a wedged backend, "
            "while still bounding an otherwise-indefinite request. Raise it for "
            "unusually heavy workloads; set ``0`` to disable (wait indefinitely)."
        ),
    )

    max_backend_state_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=0,
        alias="MAX_BACKEND_STATE_BYTES",
        description=(
            "Maximum serialized size (bytes) of a backend's persisted state "
            "envelope, enforced before the database write. PostgreSQL rejects "
            "any single wire-protocol message over 1 GiB by dropping the "
            "connection (the E199 incident class); this backstop converts an "
            "oversized state from any backend into a typed, non-retryable "
            "BACKEND_STATE_TOO_LARGE envelope with recovery guidance instead. "
            "The 256 MiB default sits far below the protocol limit while "
            "leaving generous headroom over the BayBE backend's own "
            "search-space budget. Set ``0`` to disable the check."
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
    """Convenience accessor used by the storage layer.

    Unwraps the ``SecretStr`` wrapper so callers receive the raw URL
    string. The wrapper exists to keep ``repr(settings)`` from leaking
    embedded credentials; downstream code that needs the actual URL
    (engine creation, log breadcrumbs that intentionally mask
    credentials themselves) reaches in through this accessor.
    """
    return get_settings().database_url.get_secret_value()


def get_api_env() -> str:
    """Return the deployment environment label."""
    return get_settings().api_env


def get_dev_auth() -> bool:
    """Return whether shared development authentication is enabled."""
    return get_settings().dev_auth


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


def get_db_pool_size() -> int:
    """Return the steady-state SQLAlchemy pool size."""
    return get_settings().db_pool_size


def get_db_max_overflow() -> int:
    """Return the SQLAlchemy pool overflow cap."""
    return get_settings().db_max_overflow


def get_db_pool_recycle_seconds() -> int:
    """Return the idle-connection recycle horizon (seconds)."""
    return get_settings().db_pool_recycle_seconds


def get_diagnostics_cache_max_entries() -> int:
    """Return the upper bound on the in-process diagnostics cache."""
    return get_settings().diagnostics_cache_max_entries


def get_diagnostics_cache_ttl_seconds() -> int:
    """Return the diagnostics-cache entry TTL (seconds)."""
    return get_settings().diagnostics_cache_ttl_seconds


def get_idempotency_reservation_ttl_seconds() -> int:
    """Return the idempotency reservation 'pending' TTL (seconds)."""
    return get_settings().idempotency_reservation_ttl_seconds


def get_idempotency_response_ttl_seconds() -> int:
    """Return the idempotency response replay TTL (seconds)."""
    return get_settings().idempotency_response_ttl_seconds


def get_idempotency_heartbeat_max_total_extension_seconds() -> float:
    """Return the max heartbeat runtime (elapsed seconds; 0 = unbounded)."""
    return get_settings().idempotency_heartbeat_max_total_extension_seconds


def get_bo_compute_timeout_seconds() -> float:
    """Return the per-compute wall-clock timeout (seconds; 0 = disabled)."""
    return get_settings().bo_compute_timeout_seconds


def get_max_backend_state_bytes() -> int:
    """Return the pre-persistence backend-state size limit (bytes; 0 = disabled)."""
    return get_settings().max_backend_state_bytes
