"""Database lifecycle helpers for transport layers.

Transports (FastAPI, MCP) call :func:`init_database` on startup and the
optional :func:`ping_database` from health-check endpoints. Both wrap
:mod:`bo_mcp_server.storage` so transports do not import from it
directly.
"""

from __future__ import annotations

from sqlalchemy import text

from bo_mcp_server.storage import get_session
from bo_mcp_server.storage import init_database as _init_database


async def init_database() -> None:
    """Initialize the shared database (create schema, run migrations)."""
    await _init_database()


async def ping_database() -> bool:
    """Return True when the configured database is reachable.

    Implemented as a single ``SELECT 1`` round-trip; suitable for health
    probes. Any exception is treated as "not connected" — health checks
    must never crash the surrounding endpoint.
    """
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - health probe swallows everything
        return False
    return True
