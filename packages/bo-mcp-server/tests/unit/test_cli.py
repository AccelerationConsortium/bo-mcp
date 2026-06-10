"""Tests for the CLI entry point."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from bo_mcp_server import cli

# Generous wall-clock bound for the import-time subprocess test: the CLI
# import chain pulls in mcp/fastapi/sqlalchemy, which takes a few seconds
# on a cold interpreter but stays far below this on any CI runner.
_CLI_IMPORT_TIMEOUT_SECONDS = 60


@pytest.mark.asyncio
async def test_main_async_initializes_mcp_server(monkeypatch) -> None:
    calls: list[str] = []
    bind_all_host = "0.0.0.0"  # noqa: S104 - intentional test input for SSE bind-all configuration

    async def fake_init_database() -> None:
        calls.append("init_database")

    async def fake_run_sse_async() -> None:
        calls.append("run_sse_async")

    fake_mcp = SimpleNamespace(
        settings=SimpleNamespace(host="127.0.0.1", port=8000),
        run_sse_async=fake_run_sse_async,
    )

    monkeypatch.setattr(cli, "init_database", fake_init_database)
    monkeypatch.setattr(cli, "create_mcp_server", lambda: fake_mcp)

    await cli.main_async("sse", bind_all_host, 8001)

    assert calls == ["init_database", "run_sse_async"]
    assert fake_mcp.settings.host == bind_all_host
    assert fake_mcp.settings.port == 8001


def test_main_configures_logging_before_serving(monkeypatch) -> None:
    """The entry point must install the logging bootstrap before serving.

    Without a root handler, Python's last-resort handler drops every
    record below WARNING and the documented ``BO_MCP_LOG_LEVEL`` /
    ``LOG_FORMAT`` environment switches have no runtime effect.
    """
    calls: list[str] = []
    monkeypatch.setattr(cli, "configure_logging", lambda: calls.append("configure_logging"))

    async def fake_main_async(transport: str, _host: str, _port: int) -> None:
        calls.append(f"main_async:{transport}")

    monkeypatch.setattr(cli, "main_async", fake_main_async)
    monkeypatch.setattr(cli.sys, "argv", ["bo-mcp-server", "--transport", "stdio"])

    cli.main()

    assert calls == ["configure_logging", "main_async:stdio"]


def test_import_time_startup_logs_honor_log_format() -> None:
    """Import-time records must route through the configured handler.

    The CLI module configures logging before importing the modules whose
    import side effects log (backend discovery emits an INFO breadcrumb).
    A bootstrap placed after those imports lets the records bypass
    ``BO_MCP_LOG_LEVEL`` / ``LOG_FORMAT`` and the PII/correlation
    filters. Runs in a subprocess because the defect only exists during
    a fresh interpreter's import sequence.
    """
    env = {**os.environ, "LOG_FORMAT": "json", "BO_MCP_LOG_LEVEL": "INFO"}
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter path
        [sys.executable, "-c", "import bo_mcp_server.cli"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=_CLI_IMPORT_TIMEOUT_SECONDS,
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
