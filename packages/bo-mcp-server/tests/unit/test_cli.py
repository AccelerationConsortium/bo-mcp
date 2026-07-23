"""Tests for the CLI entry point."""

import json
import os
import subprocess
import sys
import threading
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
    sse_host = "127.0.0.1"

    async def fake_init_database() -> None:
        calls.append("init_database")

    async def fake_warm_default_backend() -> None:
        calls.append("warm_default_backend")

    async def fake_run_sse_async() -> None:
        calls.append("run_sse_async")

    fake_mcp = SimpleNamespace(
        settings=SimpleNamespace(host="placeholder", port=0),
        run_sse_async=fake_run_sse_async,
    )

    def fake_create_mcp_server():
        calls.append("create_mcp_server")
        return fake_mcp

    monkeypatch.setattr(cli, "init_database", fake_init_database)
    monkeypatch.setattr(cli, "create_mcp_server", fake_create_mcp_server)
    monkeypatch.setattr(cli, "warm_default_backend", fake_warm_default_backend)

    await cli.main_async("sse", sse_host, 8001)

    assert calls == ["init_database", "warm_default_backend", "create_mcp_server", "run_sse_async"]
    assert fake_mcp.settings.host == sse_host
    assert fake_mcp.settings.port == 8001


@pytest.mark.asyncio
async def test_main_async_dispatches_streamable_http_transport(monkeypatch) -> None:
    """The current (non-deprecated) MCP HTTP transport is reachable via the CLI."""
    calls: list[str] = []
    host = "127.0.0.1"

    async def fake_init_database() -> None:
        calls.append("init_database")

    async def fake_warm_default_backend() -> None:
        calls.append("warm_default_backend")

    async def fake_run_streamable_http_async() -> None:
        calls.append("run_streamable_http_async")

    fake_mcp = SimpleNamespace(
        settings=SimpleNamespace(host="placeholder", port=0),
        run_streamable_http_async=fake_run_streamable_http_async,
    )

    def fake_create_mcp_server():
        calls.append("create_mcp_server")
        return fake_mcp

    monkeypatch.setattr(cli, "init_database", fake_init_database)
    monkeypatch.setattr(cli, "create_mcp_server", fake_create_mcp_server)
    monkeypatch.setattr(cli, "warm_default_backend", fake_warm_default_backend)

    await cli.main_async("streamable-http", host, 8001)

    assert calls == [
        "init_database",
        "warm_default_backend",
        "create_mcp_server",
        "run_streamable_http_async",
    ]
    assert fake_mcp.settings.host == host
    assert fake_mcp.settings.port == 8001


@pytest.mark.asyncio
async def test_main_async_keeps_startup_imports_off_the_event_loop(monkeypatch) -> None:
    """Startup must warm before building the server, and build off the loop.

    Backend loading imports torch (seconds of synchronous work), and
    ``create_mcp_server()`` itself loads every remaining backend through
    tool-schema enrichment at tool-module import. Warming first and
    running the server build on a worker thread keeps the whole import
    chain off the event loop; a regression here re-blocks async startup
    for the full torch/baybe import.
    """
    calls: list[str] = []
    create_thread: list[threading.Thread] = []

    async def fake_init_database() -> None:
        calls.append("init_database")

    async def fake_warm_default_backend() -> None:
        calls.append("warm_default_backend")

    async def fake_run_stdio_async() -> None:
        calls.append("run_stdio_async")

    fake_mcp = SimpleNamespace(run_stdio_async=fake_run_stdio_async)

    def fake_create_mcp_server():
        calls.append("create_mcp_server")
        create_thread.append(threading.current_thread())
        return fake_mcp

    monkeypatch.setattr(cli, "init_database", fake_init_database)
    monkeypatch.setattr(cli, "create_mcp_server", fake_create_mcp_server)
    monkeypatch.setattr(cli, "warm_default_backend", fake_warm_default_backend)

    await cli.main_async("stdio", "127.0.0.1", 8001)

    assert calls == [
        "init_database",
        "warm_default_backend",
        "create_mcp_server",
        "run_stdio_async",
    ]
    assert create_thread[0] is not threading.main_thread(), (
        "create_mcp_server must run on a worker thread — its tool-schema "
        "enrichment imports every discovered backend"
    )


@pytest.mark.asyncio
async def test_main_async_disposes_engine_on_clean_return(monkeypatch) -> None:
    """``main_async`` disposes the DB engine when the transport exits cleanly.

    Without the disposal, PostgreSQL logs per-connection EOF noise on
    every restart and in-flight background transactions die server-side
    instead of closing cleanly.
    """
    closed: list[str] = []

    async def fake_init_database() -> None:
        pass

    async def fake_warm_default_backend() -> None:
        pass

    async def fake_close_database() -> None:
        closed.append("close_database")

    async def fake_run_stdio_async() -> None:
        pass

    fake_mcp = SimpleNamespace(run_stdio_async=fake_run_stdio_async)
    monkeypatch.setattr(cli, "init_database", fake_init_database)
    monkeypatch.setattr(cli, "warm_default_backend", fake_warm_default_backend)
    monkeypatch.setattr(cli, "close_database", fake_close_database)
    monkeypatch.setattr(cli, "create_mcp_server", lambda: fake_mcp)

    await cli.main_async("stdio", "127.0.0.1", 8001)

    assert closed == ["close_database"]


@pytest.mark.asyncio
async def test_main_async_disposes_engine_on_transport_error(monkeypatch) -> None:
    """``main_async`` disposes the DB engine even when the transport raises."""
    closed: list[str] = []

    async def fake_init_database() -> None:
        pass

    async def fake_warm_default_backend() -> None:
        pass

    async def fake_close_database() -> None:
        closed.append("close_database")

    async def failing_run_stdio_async() -> None:
        msg = "transport crashed"
        raise RuntimeError(msg)

    fake_mcp = SimpleNamespace(run_stdio_async=failing_run_stdio_async)
    monkeypatch.setattr(cli, "init_database", fake_init_database)
    monkeypatch.setattr(cli, "warm_default_backend", fake_warm_default_backend)
    monkeypatch.setattr(cli, "close_database", fake_close_database)
    monkeypatch.setattr(cli, "create_mcp_server", lambda: fake_mcp)

    with pytest.raises(RuntimeError, match="transport crashed"):
        await cli.main_async("stdio", "127.0.0.1", 8001)

    assert closed == ["close_database"]


def test_parsed_default_host_is_loopback() -> None:
    """A bare ``--transport sse`` must not expose the server on the network.

    The host default used to be ``0.0.0.0``, which served unauthenticated
    campaign data on every interface the moment SSE mode was selected.
    Binding all interfaces now requires passing ``--host 0.0.0.0``
    explicitly.
    """
    args = cli.build_arg_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.transport == "stdio"
    assert args.port == 8001


def test_explicit_host_overrides_loopback_default() -> None:
    bind_all_host = "0.0.0.0"  # noqa: S104 - intentional test input for explicit SSE bind-all opt-in
    args = cli.build_arg_parser().parse_args(["--transport", "sse", "--host", bind_all_host])

    assert args.host == bind_all_host


def test_streamable_http_is_an_accepted_transport_choice() -> None:
    """The current (non-deprecated) MCP HTTP transport is a valid ``--transport`` value."""
    args = cli.build_arg_parser().parse_args(["--transport", "streamable-http"])

    assert args.transport == "streamable-http"


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
    result = subprocess.run(
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
