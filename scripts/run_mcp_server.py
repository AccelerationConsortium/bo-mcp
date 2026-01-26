#!/usr/bin/env python3
"""
Run the BO-MCP server as a standalone MCP server.

This script starts the MCP server without the FastAPI REST layer,
allowing direct MCP connections from Claude Code or other MCP clients.

Usage:
    # For Claude Code (stdio transport):
    uv run python scripts/run_mcp_server.py

    # For network access (SSE transport):
    uv run python scripts/run_mcp_server.py --transport sse --port 8001

Transports:
    stdio - Standard input/output (default). Use this for Claude Code
            and local CLI integrations. Communication happens via stdin/stdout.

    sse   - Server-Sent Events over HTTP. Use this for network access
            when you need remote clients to connect to the MCP server.
"""

import argparse
import asyncio

from bo_mcp_server import create_mcp_server
from bo_mcp_server.storage import init_database


async def setup():
    """Initialize the database before starting the server."""
    await init_database()


def main():
    parser = argparse.ArgumentParser(
        description="Run BO-MCP as a standalone MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with stdio transport (for Claude Code):
    uv run python scripts/run_mcp_server.py

    # Start with SSE transport on custom port:
    uv run python scripts/run_mcp_server.py --transport sse --port 8001

    # Start with SSE transport accessible from network:
    uv run python scripts/run_mcp_server.py --transport sse --host 0.0.0.0 --port 8001

Claude Code Configuration:
    Add to ~/.claude/claude_desktop_config.json:

    {
      "mcpServers": {
        "bo-mcp": {
          "command": "uv",
          "args": ["run", "python", "scripts/run_mcp_server.py"],
          "cwd": "/path/to/bo-mcp-ui"
        }
      }
    }
""",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol: 'stdio' for Claude Code (default), 'sse' for network access",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for SSE transport (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for SSE transport (default: 8001)",
    )

    args = parser.parse_args()

    # Initialize database
    asyncio.run(setup())

    # Create MCP server with all tools and resources registered
    mcp = create_mcp_server()

    if args.transport == "sse":
        print(f"Starting BO-MCP server with SSE transport on {args.host}:{args.port}")
        print("Available tools: validate_intake, create_campaign, generate_suggestions,")
        print("                 submit_results, get_diagnostics")
        print("Available resources: campaign://, campaigns://list, suggestions://,")
        print("                     suggestion://")
        print("-" * 60)
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        # stdio transport for Claude Code - no printing to stdout
        # as it would interfere with the MCP protocol
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
