"""Tests for the CLI entry point."""

from types import SimpleNamespace

import pytest

from bo_mcp_server import cli


@pytest.mark.asyncio
async def test_main_async_initializes_mcp_server(monkeypatch) -> None:
    calls: list[str] = []

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

    await cli.main_async("sse", "0.0.0.0", 8001)

    assert calls == ["init_database", "run_sse_async"]
    assert fake_mcp.settings.host == "0.0.0.0"
    assert fake_mcp.settings.port == 8001
