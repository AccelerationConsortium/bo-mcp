"""CLI entry point for bo-mcp-server."""

import argparse
import asyncio
import json
import sys

import dotenv

from bo_mcp_server import __version__

dotenv.load_dotenv()

from bo_mcp_server.server import create_mcp_server  # noqa: E402
from bo_mcp_server.storage import close_database, init_database  # noqa: E402


async def main_async(transport: str, host: str, port: int) -> None:
    """Async main function."""
    await init_database()
    mcp = create_mcp_server()

    if transport == "stdio":
        await mcp.run_stdio_async()
    else:
        mcp.settings.host = host
        mcp.settings.port = port
        await mcp.run_sse_async()


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


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BO-MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="SSE host (default: 0.0.0.0)")  # noqa: S104
    parser.add_argument("--port", type=int, default=8001, help="SSE port (default: 8001)")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify setup and exit (useful for agents to check installation)",
    )

    args = parser.parse_args()

    if args.verify:
        asyncio.run(_verify_setup())
        return

    asyncio.run(main_async(args.transport, args.host, args.port))


if __name__ == "__main__":
    main()
