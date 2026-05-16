"""Database lifecycle helpers for transport layers.

Transports (FastAPI, MCP) call :func:`init_database` on startup and the
optional :func:`ping_database` from health-check endpoints. Both wrap
:mod:`bo_mcp_server.storage` so transports do not import from it
directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from bo_mcp_server.storage import get_session
from bo_mcp_server.storage import init_database as _init_database

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabasePingResult:
    """Outcome of a single ``SELECT 1`` health probe.

    ``error_class`` carries the exception type name on failure (e.g.
    ``OperationalError``) so health endpoints can surface it for
    operators without leaking exception arguments — which may include
    PII or query fragments.
    """

    healthy: bool
    error_class: str | None = None


async def init_database() -> None:
    """Initialize the shared database (create schema, run migrations)."""
    await _init_database()


async def ping_database() -> bool:
    """Return True when the configured database is reachable.

    Thin convenience wrapper around :func:`ping_database_detailed` that
    preserves the original bool-only contract for existing callers.
    New code should prefer :func:`ping_database_detailed` so the
    failure-class label is available for diagnostics.
    """
    return (await ping_database_detailed()).healthy


async def ping_database_detailed() -> DatabasePingResult:
    """Run the health probe and report the failure class on errors.

    Implemented as a single ``SELECT 1`` round-trip. We only swallow
    database-side failures (``SQLAlchemyError``) and connect-time
    timeouts so orchestrators (k8s, load balancers) keep their existing
    "DB down → unhealthy" behaviour. Other exception types — typically
    programming bugs such as ``AttributeError`` from a misconfigured
    engine — surface so they are not silently masked as transient
    connectivity failures.
    """
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
    except (SQLAlchemyError, TimeoutError) as err:
        error_class = type(err).__name__
        logger.warning("Database health probe failed: %s: %s", error_class, err)
        return DatabasePingResult(healthy=False, error_class=error_class)
    return DatabasePingResult(healthy=True)
