"""Logging configuration for bo-mcp-server.

This module provides centralized logging configuration for the MCP server.
All tool and storage modules should import and use the logger from here.
"""

import logging
import os
import sys
from typing import Final

# Default log level from environment, defaulting to INFO
DEFAULT_LOG_LEVEL: Final[str] = os.environ.get("BO_MCP_LOG_LEVEL", "INFO")

# Default format for log messages
DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(
    level: str | None = None,
    format_string: str | None = None,
) -> None:
    """Configure logging for the bo-mcp-server package.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to BO_MCP_LOG_LEVEL env var or INFO.
        format_string: Log message format. Defaults to standard format.
    """
    log_level = level or DEFAULT_LOG_LEVEL
    log_format = format_string or DEFAULT_LOG_FORMAT

    # Configure the root logger for bo_mcp_server
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    # Set level specifically for bo_mcp_server package
    logger = logging.getLogger("bo_mcp_server")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance configured for the module.
    """
    return logging.getLogger(name)
