"""The REST API process must bootstrap logging at module init.

``api.main`` is the uvicorn import target, so its module-level
``configure_logging()`` call is what guarantees a root handler exists
in every API process. Without one, Python's last-resort handler drops
every record below WARNING and the documented ``BO_MCP_LOG_LEVEL`` /
``LOG_FORMAT`` switches have no runtime effect.

Runs in a subprocess: the contract is about a fresh interpreter's
import sequence, and in-process assertions on root-handler identity are
unreliable — pytest's capture machinery swaps ``sys.stderr`` per test,
and other tests legitimately reconfigure the root logger.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# Generous wall-clock bound: the api.main import chain pulls in
# fastapi/sqlalchemy/bo_mcp_server, a few seconds on a cold interpreter.
_API_IMPORT_TIMEOUT_SECONDS = 60

_BOOTSTRAP_PROBE = """\
import logging
import sys

import api.main  # noqa: F401 - importing executes the module-level bootstrap

root = logging.getLogger()
assert any(
    isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr
    for handler in root.handlers
), f"no configured stderr handler on root: {root.handlers!r}"
# INFO-level operational breadcrumbs must not be dropped; the
# pre-bootstrap behavior was an unconfigured root that only let
# WARNING+ through Python's last-resort handler.
assert logging.getLogger("bo_mcp_server").isEnabledFor(logging.INFO)
assert logging.getLogger("api").isEnabledFor(logging.INFO)
"""


def test_api_module_init_bootstraps_logging() -> None:
    """Importing ``api.main`` leaves a configured stderr handler on root."""
    env = {
        **os.environ,
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "USE_ALEMBIC": "false",
    }
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter path
        [sys.executable, "-c", _BOOTSTRAP_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=_API_IMPORT_TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_api_import_time_startup_logs_honor_log_format() -> None:
    """Import-time records must route through the configured handler.

    ``api.main`` configures logging before importing the
    ``bo_mcp_server`` modules whose import side effects log (backend
    discovery emits an INFO breadcrumb). A bootstrap placed after those
    imports lets the records bypass ``BO_MCP_LOG_LEVEL`` /
    ``LOG_FORMAT`` and the PII/correlation filters. Mirrors the
    equivalent CLI entry-point test in bo-mcp-server.
    """
    env = {
        **os.environ,
        "LOG_FORMAT": "json",
        "BO_MCP_LOG_LEVEL": "INFO",
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "USE_ALEMBIC": "false",
    }
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter path
        [sys.executable, "-c", "import api.main"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=_API_IMPORT_TIMEOUT_SECONDS,
    )
    discovery_lines = [
        line for line in result.stderr.splitlines() if "Discovered bo-mcp backends" in line
    ]
    assert discovery_lines, f"expected the discovery breadcrumb on stderr: {result.stderr!r}"
    # Every occurrence must be a JSON record — a plain-text duplicate
    # would mean a second, unconfigured handler is still attached.
    for line in discovery_lines:
        record = json.loads(line)
        assert record["level"] == "INFO"
        assert record["name"] == "bo_mcp_server.backend"
