"""BO-MCP Server - Bayesian Optimization via Model Context Protocol.

This package provides MCP tools for Bayesian Optimization campaigns.
"""

from bo_mcp_server.logging_config import configure_logging, get_logger
from bo_mcp_server.server import create_mcp_server, mcp
from bo_mcp_server.storage import get_session, init_database

__all__ = [
    # Logging
    "configure_logging",
    # Server
    "create_mcp_server",
    "get_logger",
    "get_session",
    # Database
    "init_database",
    "mcp",
]

__version__ = "0.1.0"
