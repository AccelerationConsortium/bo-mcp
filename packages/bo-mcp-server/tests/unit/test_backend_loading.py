"""Tests for backend loading semantics (caching, async offload, warm-up).

Backend loading runs the entry-point import (torch / baybe — seconds of
synchronous work), so the provider must (a) cache instances, (b) load
each backend exactly once even under concurrent access, and (c) offer an
async path that resolves cached hits without a thread hop. The
event-loop responsiveness of the offload itself is covered by
``tests/integration/test_event_loop_responsiveness.py``.

Reference: the offload pattern follows the asyncio guidance for blocking
calls — https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from bo_mcp_server import backend as backend_module

_BACKEND_NAME = "botorch"
# Long enough that overlapping loads would reliably interleave without
# the lock, short enough to keep the test fast.
_LOAD_SECONDS = 0.05


@pytest.fixture
def fresh_backend_cache(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Isolate the module-level backend cache and pin the default name."""
    cache: dict[str, object] = {}
    monkeypatch.setattr(backend_module, "_backends", cache)
    monkeypatch.setenv("BO_BACKEND", _BACKEND_NAME)
    return cache


@pytest.fixture
def counting_loader(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the slow entry-point load with a countable, sleeping stub."""
    loaded: list[str] = []

    def fake_load(name: str) -> SimpleNamespace:
        time.sleep(_LOAD_SECONDS)
        loaded.append(name)
        return SimpleNamespace(name=name)

    monkeypatch.setattr(backend_module, "_load_backend", fake_load)
    return loaded


@pytest.mark.usefixtures("counting_loader")
class TestGetBackendAsync:
    @pytest.mark.asyncio
    async def test_cached_hit_returns_existing_instance(
        self, fresh_backend_cache: dict[str, object]
    ) -> None:
        sentinel = SimpleNamespace(name=_BACKEND_NAME)
        fresh_backend_cache[_BACKEND_NAME] = sentinel

        assert await backend_module.get_backend_async() is sentinel

    @pytest.mark.asyncio
    async def test_cache_miss_loads_and_caches(
        self, fresh_backend_cache: dict[str, object]
    ) -> None:
        backend = await backend_module.get_backend_async(_BACKEND_NAME)

        assert backend.name == _BACKEND_NAME
        assert fresh_backend_cache[_BACKEND_NAME] is backend

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("fresh_backend_cache")
    async def test_concurrent_misses_load_exactly_once(self, counting_loader: list[str]) -> None:
        """Two concurrent first requests must not double-initialize.

        Both calls miss the cache and offload to worker threads; the
        load lock must serialize them so the second observes the first's
        cached instance instead of running its own multi-second load.
        """
        first, second = await asyncio.gather(
            backend_module.get_backend_async(_BACKEND_NAME),
            backend_module.get_backend_async(_BACKEND_NAME),
        )

        assert first is second
        assert counting_loader == [_BACKEND_NAME]

    @pytest.mark.usefixtures("fresh_backend_cache")
    def test_concurrent_sync_misses_load_exactly_once(self, counting_loader: list[str]) -> None:
        """The sync accessor is itself race-free across worker threads."""
        results: list[object] = []

        def hit() -> None:
            results.append(backend_module.get_backend(_BACKEND_NAME))

        threads = [threading.Thread(target=hit) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert counting_loader == [_BACKEND_NAME]
        assert all(result is results[0] for result in results)


class TestWarmDefaultBackend:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("counting_loader")
    async def test_warms_the_default_backend_into_the_cache(
        self, fresh_backend_cache: dict[str, object]
    ) -> None:
        backend = await backend_module.warm_default_backend()

        assert backend.name == _BACKEND_NAME
        assert fresh_backend_cache[_BACKEND_NAME] is backend

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("fresh_backend_cache")
    async def test_unknown_default_backend_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A misconfigured BO_BACKEND must fail at startup, not first call.

        Uses the real ``_load_backend`` so the error path exercised is
        the actual entry-point lookup failure.
        """
        monkeypatch.setenv("BO_BACKEND", "does-not-exist")

        with pytest.raises(ValueError, match="does-not-exist"):
            await backend_module.warm_default_backend()
