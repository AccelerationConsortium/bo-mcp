"""CLI entry point for bo-mcp-server."""

import argparse
import asyncio
import json
import sys

from bo_mcp_server import __version__
from bo_mcp_server.settings import load_anchored_dotenv

# Anchored (never CWD-relative) so an MCP client spawning this server
# from a directory containing a foreign ``.env`` cannot repoint the
# database or logging configuration.
load_anchored_dotenv()

from bo_mcp_server.logging_config import configure_logging  # noqa: E402

# Logging must be configured before the imports below run their side
# effects: backend discovery logs an INFO breadcrumb at import time,
# which would otherwise bypass BO_MCP_LOG_LEVEL / LOG_FORMAT and the
# PII/correlation filters. ``load_anchored_dotenv()`` above has already
# populated the environment. Handlers write to stderr, so the stdio
# JSON-RPC channel on stdout stays clean.
configure_logging()

from bo_mcp_server.backend import warm_default_backend  # noqa: E402
from bo_mcp_server.client import ensure_mcp_startup_user  # noqa: E402
from bo_mcp_server.idempotency_gc import idempotency_gc_lifespan  # noqa: E402
from bo_mcp_server.server import create_mcp_server  # noqa: E402
from bo_mcp_server.storage import close_database, init_database  # noqa: E402

# SSE binds loopback by default so a bare ``--transport sse`` never
# exposes the server on the network; ``--host 0.0.0.0`` opts in
# explicitly. Note: the SSE transport carries no authentication or
# per-tenant authorization yet (tracked as H19), so a non-loopback bind
# exposes every tool to anyone who can reach the port — keep it loopback
# or behind an authenticating proxy.
DEFAULT_SSE_HOST = "127.0.0.1"
DEFAULT_SSE_PORT = 8001


async def main_async(transport: str, host: str, port: int) -> None:
    """Async main function.

    The engine is disposed in the ``finally`` block on every exit path
    (clean shutdown, transport error, cancellation): without it,
    PostgreSQL logs per-connection EOF noise on every restart and
    in-flight GC transactions die server-side instead of being closed
    cleanly.
    """
    try:
        await init_database()
        await ensure_mcp_startup_user()
        # Load the default backend (torch import — seconds) on a worker
        # thread before anything else can pull it in, so the first tool
        # call can never stall the event loop on the import. Ordered before
        # create_mcp_server(), whose tool-schema enrichment loads every
        # discovered backend at tool-module import — that call therefore
        # runs on a worker thread too, keeping the remaining backend
        # imports (e.g. baybe) off the event loop as well.
        await warm_default_backend()
        mcp = await asyncio.to_thread(create_mcp_server)

        async with idempotency_gc_lifespan():
            if transport == "stdio":
                await mcp.run_stdio_async()
            else:
                mcp.settings.host = host
                mcp.settings.port = port
                await mcp.run_sse_async()
    finally:
        await close_database()


async def _verify_setup() -> None:
    """Verify server setup and print status JSON.

    Checks database connectivity and prints a JSON status report.
    Exits with code 0 on success, 1 on error.
    """
    mcp = create_mcp_server()
    status: dict[str, str | int] = {
        "status": "ok",
        "version": __version__,
        "tools": len(mcp._tool_manager.list_tools()),
    }

    try:
        await init_database()
        await ensure_mcp_startup_user()
        status["database"] = "connected"
    except Exception as e:  # noqa: BLE001 - DB drivers raise varied exception types
        status["status"] = "error"
        status["database"] = f"error: {e}"
    finally:
        try:
            await close_database()
        except Exception as e:  # noqa: BLE001 - teardown must not block exit
            status["database_close_warning"] = str(e)

    print(json.dumps(status, indent=2))
    sys.exit(0 if status["status"] == "ok" else 1)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    ``--host`` defaults to loopback so a bare ``--transport sse`` never
    exposes campaign data on the network; binding all interfaces
    requires explicitly passing ``--host 0.0.0.0``. The SSE transport
    has no authentication or per-tenant authorization yet (H19), so a
    non-loopback bind exposes every tool to anyone who can reach the
    port — keep it loopback or front it with an authenticating proxy.
    """
    parser = argparse.ArgumentParser(description="BO-MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_SSE_HOST,
        help=(
            f"SSE bind host (default: {DEFAULT_SSE_HOST}). Pass 0.0.0.0 explicitly "
            "to expose the server on the network; the SSE transport is "
            "unauthenticated (H19), so prefer loopback or an authenticating proxy."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SSE_PORT,
        help=f"SSE port (default: {DEFAULT_SSE_PORT})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify setup and exit (useful for agents to check installation)",
    )
    return parser


def main() -> None:
    """Main entry point."""
    # Re-run the bootstrap (idempotent: handlers are replaced, not
    # appended). The module-level call above covers import-time records;
    # this one evicts any handler a third-party import installed on the
    # root logger afterwards, and re-reads BO_MCP_LOG_LEVEL in case the
    # embedding process loaded the environment after importing this
    # module.
    configure_logging()

    args = build_arg_parser().parse_args()

    if args.verify:
        asyncio.run(_verify_setup())
        return

    asyncio.run(main_async(args.transport, args.host, args.port))


if __name__ == "__main__":
    main()
