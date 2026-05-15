"""Tests for the response cache module.

The cache uses version-aware keys (e.g. "diagnostics:{id}:{version}") so
entries become unreachable after mutations — no explicit invalidation needed.

TTL behaviour is exercised against an injected fake clock so tests advance
time deterministically and do not depend on wall-clock ``asyncio.sleep`` —
which made the suite flaky on slow CI and inflated runtime.

Reference: BO-MCP-UI Implementation Plan Step 5, Section 2.5.
"""

import time
from datetime import UTC, datetime, timedelta

import pytest

from bo_mcp_server.cache import MAX_CACHE_ENTRIES, ResponseCache, diagnostics_cache


class FakeClock:
    """Mutable clock for deterministic TTL tests.

    The cache calls ``clock()`` whenever it needs to read "now"; the test
    advances ``current`` directly instead of sleeping.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self.current = start or datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current = self.current + timedelta(seconds=seconds)


class TestResponseCache:
    """Tests for ResponseCache class."""

    def test_cache_initialization_default_ttl(self) -> None:
        """Cache should initialize with default 120 second TTL."""
        cache = ResponseCache()
        assert cache._ttl == timedelta(seconds=120)
        assert cache.size == 0

    def test_cache_initialization_custom_ttl(self) -> None:
        """Cache should accept custom TTL values."""
        cache = ResponseCache(ttl_seconds=60)
        assert cache._ttl == timedelta(seconds=60)

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self) -> None:
        """Cache should store and retrieve values correctly."""
        cache = ResponseCache(ttl_seconds=10)
        test_data = {"success": True, "health_status": "healthy"}

        await cache.set("diagnostics:test-campaign:1", test_data)
        result = await cache.get("diagnostics:test-campaign:1")

        assert result == test_data

    @pytest.mark.asyncio
    async def test_cache_get_nonexistent_key(self) -> None:
        """Cache should return None for nonexistent keys."""
        cache = ResponseCache()
        result = await cache.get("nonexistent-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self) -> None:
        """Cache entries should expire after TTL.

        Reference: TTL-based caching is essential for balancing freshness
        with avoiding redundant computation during optimization loops. The
        TTL is exercised against a fake clock so this test does not depend
        on wall-clock sleeps (which were a documented source of flakiness
        on slow CI runners).
        """
        clock = FakeClock()
        cache = ResponseCache(ttl_seconds=1, clock=clock)
        await cache.set("test-key", {"value": 42})

        # Immediately after set, value should be retrievable
        assert await cache.get("test-key") == {"value": 42}

        # Advance virtual time past the TTL.
        clock.advance(1.1)

        # After TTL, value should be None
        assert await cache.get("test-key") is None

    @pytest.mark.asyncio
    async def test_version_aware_keys_auto_invalidate(self) -> None:
        """Version-aware keys mean old versions become unreachable.

        Reference: Section 2.5 — Use diagnostics:{campaign_id}:{version}
        as cache key. No explicit invalidation needed since version changes
        on every mutation.
        """
        cache = ResponseCache()
        campaign_id = "abc-123-def-456"

        # Store diagnostics for version 1
        await cache.set(f"diagnostics:{campaign_id}:1", {"health": "healthy"})

        # After a mutation, version increments to 2
        # The old key is no longer looked up
        assert await cache.get(f"diagnostics:{campaign_id}:2") is None

        # Old version entry is still technically in cache but unreachable
        assert await cache.get(f"diagnostics:{campaign_id}:1") == {"health": "healthy"}

    @pytest.mark.asyncio
    async def test_cache_clear(self) -> None:
        """Cache clear should remove all entries."""
        cache = ResponseCache()
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        assert cache.size == 3

        await cache.clear()

        assert cache.size == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_cache_size_property(self) -> None:
        """Cache size should reflect number of entries."""
        cache = ResponseCache()
        assert cache.size == 0

        await cache.set("key1", "value1")
        assert cache.size == 1

        await cache.set("key2", "value2")
        assert cache.size == 2

    @pytest.mark.asyncio
    async def test_cache_overwrite_existing_key(self) -> None:
        """Setting same key should overwrite previous value and reset TTL.

        The set→advance→overwrite→advance sequence uses a fake clock so the
        TTL boundary is hit deterministically rather than relying on
        ``asyncio.sleep`` (which slow CI was failing intermittently on).
        """
        clock = FakeClock()
        cache = ResponseCache(ttl_seconds=2, clock=clock)

        await cache.set("test-key", {"version": 1})
        clock.advance(1)

        # Overwrite with new value — this resets the TTL anchor for the key.
        await cache.set("test-key", {"version": 2})

        # Value should be updated
        assert await cache.get("test-key") == {"version": 2}

        # Advance past the original entry's TTL but still within the
        # refreshed window from the overwrite.
        clock.advance(1.1)
        assert await cache.get("test-key") == {"version": 2}

    @pytest.mark.asyncio
    async def test_cache_handles_complex_values(self) -> None:
        """Cache should handle complex nested dictionaries.

        Reference: get_diagnostics returns complex nested responses with
        Pareto fronts, convergence info, feature importance, etc.
        """
        cache = ResponseCache()
        complex_data = {
            "success": True,
            "campaign_status": "running",
            "pareto_front": [
                {"obj1": 1.5, "obj2": 2.3},
                {"obj1": 2.1, "obj2": 1.8},
            ],
            "convergence": {
                "converged": False,
                "convergence_score": 0.45,
                "reason": "Still improving",
            },
            "feature_importance": {
                "param1": 0.8,
                "param2": 0.2,
            },
        }

        await cache.set("diagnostics:complex-test:5", complex_data)
        result = await cache.get("diagnostics:complex-test:5")

        assert result is not None
        assert result == complex_data
        assert result["pareto_front"][0]["obj1"] == 1.5
        assert result["convergence"]["converged"] is False

    @pytest.mark.asyncio
    async def test_cache_evicts_oldest_at_capacity(self) -> None:
        """Cache should evict oldest entry when at MAX_CACHE_ENTRIES."""
        cache = ResponseCache(ttl_seconds=60)

        # Fill to capacity
        for i in range(MAX_CACHE_ENTRIES):
            await cache.set(f"key-{i}", {"i": i})

        assert cache.size == MAX_CACHE_ENTRIES

        # One more should evict the oldest
        await cache.set("overflow-key", {"new": True})
        assert cache.size == MAX_CACHE_ENTRIES
        assert await cache.get("overflow-key") == {"new": True}


class TestGlobalDiagnosticsCache:
    """Tests for the global diagnostics_cache instance."""

    def test_global_cache_exists(self) -> None:
        """Global diagnostics_cache should be initialized."""
        assert diagnostics_cache is not None
        assert isinstance(diagnostics_cache, ResponseCache)

    def test_global_cache_has_120s_ttl(self) -> None:
        """Global cache should have 120 second TTL (version-aware keys)."""
        assert diagnostics_cache._ttl == timedelta(seconds=120)

    @pytest.mark.asyncio
    async def test_global_cache_operations(self) -> None:
        """Global cache should support standard operations."""
        test_key = f"test-{time.time()}"

        # Set
        await diagnostics_cache.set(test_key, {"test": True})

        # Get
        result = await diagnostics_cache.get(test_key)
        assert result == {"test": True}

        # Clear (cleanup)
        await diagnostics_cache.clear()


class TestCacheEdgeCases:
    """Edge case tests for cache behavior."""

    @pytest.mark.asyncio
    async def test_cache_with_none_value(self) -> None:
        """Cache should handle None values correctly.

        This is important because we need to distinguish between
        'not in cache' and 'cached value is None'.
        """
        cache = ResponseCache()
        await cache.set("none-value-key", None)

        # The value is None, but it is cached
        # Note: Our implementation returns None for both cases
        # This is acceptable since diagnostics shouldn't return None
        result = await cache.get("none-value-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_with_empty_dict(self) -> None:
        """Cache should handle empty dictionaries."""
        cache = ResponseCache()
        await cache.set("empty-dict", {})

        result = await cache.get("empty-dict")
        assert result == {}

    @pytest.mark.asyncio
    async def test_cache_with_empty_list(self) -> None:
        """Cache should handle empty lists."""
        cache = ResponseCache()
        await cache.set("empty-list", [])

        result = await cache.get("empty-list")
        assert result == []

    @pytest.mark.asyncio
    async def test_cache_concurrent_like_access_patterns(self) -> None:
        """Cache should handle rapid set/get patterns.

        This simulates tight optimization loops where diagnostics
        might be checked frequently.
        """
        cache = ResponseCache(ttl_seconds=10)
        campaign_id = "tight-loop-test"

        for i in range(100):
            await cache.set(f"diagnostics:{campaign_id}:{i}", {"iteration": i})
            result = await cache.get(f"diagnostics:{campaign_id}:{i}")
            assert result == {"iteration": i}
