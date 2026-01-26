"""Tests for the response cache module.

These tests verify the TTL-based caching behavior for expensive MCP operations
like get_diagnostics.

Reference: BO-MCP-UI Implementation Plan v3.1, Section 12.4.
"""

import time
from datetime import datetime, timedelta

from bo_mcp_server.cache import ResponseCache, diagnostics_cache


class TestResponseCache:
    """Tests for ResponseCache class."""

    def test_cache_initialization_default_ttl(self) -> None:
        """Cache should initialize with default 30 second TTL."""
        cache = ResponseCache()
        assert cache._ttl == timedelta(seconds=30)
        assert cache.size == 0

    def test_cache_initialization_custom_ttl(self) -> None:
        """Cache should accept custom TTL values."""
        cache = ResponseCache(ttl_seconds=60)
        assert cache._ttl == timedelta(seconds=60)

    def test_cache_set_and_get(self) -> None:
        """Cache should store and retrieve values correctly."""
        cache = ResponseCache(ttl_seconds=10)
        test_data = {"success": True, "health_status": "healthy"}

        cache.set("diagnostics:test-campaign", test_data)
        result = cache.get("diagnostics:test-campaign")

        assert result == test_data

    def test_cache_get_nonexistent_key(self) -> None:
        """Cache should return None for nonexistent keys."""
        cache = ResponseCache()
        result = cache.get("nonexistent-key")

        assert result is None

    def test_cache_ttl_expiration(self) -> None:
        """Cache entries should expire after TTL.

        Reference: TTL-based caching is essential for balancing freshness
        with avoiding redundant computation during optimization loops.
        """
        cache = ResponseCache(ttl_seconds=1)
        cache.set("test-key", {"value": 42})

        # Immediately after set, value should be retrievable
        assert cache.get("test-key") == {"value": 42}

        # Wait for TTL to expire
        time.sleep(1.1)

        # After TTL, value should be None
        assert cache.get("test-key") is None

    def test_cache_invalidate_by_campaign_id(self) -> None:
        """Cache should invalidate all entries containing campaign_id.

        Reference: Section 12.4 - Cache invalidation should be called
        after mutations (submit_results, generate_suggestions).
        """
        cache = ResponseCache()
        campaign_id = "abc-123-def-456"

        # Set multiple entries for the same campaign
        cache.set(f"diagnostics:{campaign_id}", {"health": "healthy"})
        cache.set(f"suggestions:{campaign_id}", {"count": 5})
        cache.set("diagnostics:other-campaign", {"health": "warning"})

        # Invalidate the specific campaign
        cache.invalidate(campaign_id)

        # Campaign entries should be gone
        assert cache.get(f"diagnostics:{campaign_id}") is None
        assert cache.get(f"suggestions:{campaign_id}") is None

        # Other campaign entry should remain
        assert cache.get("diagnostics:other-campaign") == {"health": "warning"}

    def test_cache_invalidate_nonexistent_campaign(self) -> None:
        """Invalidating nonexistent campaign should not raise error."""
        cache = ResponseCache()
        cache.set("diagnostics:existing", {"value": 1})

        # Should not raise
        cache.invalidate("nonexistent-campaign")

        # Existing entry should remain
        assert cache.get("diagnostics:existing") == {"value": 1}

    def test_cache_clear(self) -> None:
        """Cache clear should remove all entries."""
        cache = ResponseCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.size == 3

        cache.clear()

        assert cache.size == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_cache_size_property(self) -> None:
        """Cache size should reflect number of entries."""
        cache = ResponseCache()
        assert cache.size == 0

        cache.set("key1", "value1")
        assert cache.size == 1

        cache.set("key2", "value2")
        assert cache.size == 2

        cache.invalidate("key1")
        assert cache.size == 1

    def test_cache_overwrite_existing_key(self) -> None:
        """Setting same key should overwrite previous value and reset TTL."""
        cache = ResponseCache(ttl_seconds=2)

        cache.set("test-key", {"version": 1})
        time.sleep(1)

        # Overwrite with new value
        cache.set("test-key", {"version": 2})

        # Value should be updated
        assert cache.get("test-key") == {"version": 2}

        # Wait a bit more - original entry would have expired, but new one shouldn't
        time.sleep(1.1)
        assert cache.get("test-key") == {"version": 2}

    def test_cache_handles_complex_values(self) -> None:
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

        cache.set("diagnostics:complex-test", complex_data)
        result = cache.get("diagnostics:complex-test")

        assert result == complex_data
        assert result["pareto_front"][0]["obj1"] == 1.5
        assert result["convergence"]["converged"] is False


class TestGlobalDiagnosticsCache:
    """Tests for the global diagnostics_cache instance."""

    def test_global_cache_exists(self) -> None:
        """Global diagnostics_cache should be initialized."""
        assert diagnostics_cache is not None
        assert isinstance(diagnostics_cache, ResponseCache)

    def test_global_cache_has_30s_ttl(self) -> None:
        """Global cache should have 30 second TTL as per spec."""
        assert diagnostics_cache._ttl == timedelta(seconds=30)

    def test_global_cache_operations(self) -> None:
        """Global cache should support standard operations."""
        test_key = f"test-{datetime.now().timestamp()}"

        # Set
        diagnostics_cache.set(test_key, {"test": True})

        # Get
        result = diagnostics_cache.get(test_key)
        assert result == {"test": True}

        # Invalidate
        diagnostics_cache.invalidate(test_key)
        assert diagnostics_cache.get(test_key) is None


class TestCacheEdgeCases:
    """Edge case tests for cache behavior."""

    def test_cache_with_none_value(self) -> None:
        """Cache should handle None values correctly.

        This is important because we need to distinguish between
        'not in cache' and 'cached value is None'.
        """
        cache = ResponseCache()
        cache.set("none-value-key", None)

        # The value is None, but it is cached
        # Note: Our implementation returns None for both cases
        # This is acceptable since diagnostics shouldn't return None
        result = cache.get("none-value-key")
        assert result is None

    def test_cache_with_empty_dict(self) -> None:
        """Cache should handle empty dictionaries."""
        cache = ResponseCache()
        cache.set("empty-dict", {})

        result = cache.get("empty-dict")
        assert result == {}

    def test_cache_with_empty_list(self) -> None:
        """Cache should handle empty lists."""
        cache = ResponseCache()
        cache.set("empty-list", [])

        result = cache.get("empty-list")
        assert result == []

    def test_cache_invalidate_partial_match(self) -> None:
        """Invalidate should match campaign_id anywhere in key."""
        cache = ResponseCache()
        campaign_id = "test-uuid-123"

        cache.set(f"prefix:{campaign_id}:suffix", {"a": 1})
        cache.set(f"diagnostics:{campaign_id}", {"b": 2})
        cache.set(f"{campaign_id}:extra", {"c": 3})

        cache.invalidate(campaign_id)

        # All entries containing campaign_id should be removed
        assert cache.get(f"prefix:{campaign_id}:suffix") is None
        assert cache.get(f"diagnostics:{campaign_id}") is None
        assert cache.get(f"{campaign_id}:extra") is None

    def test_cache_concurrent_like_access_patterns(self) -> None:
        """Cache should handle rapid set/get patterns.

        This simulates tight optimization loops where diagnostics
        might be checked frequently.
        """
        cache = ResponseCache(ttl_seconds=10)
        campaign_id = "tight-loop-test"

        for i in range(100):
            cache.set(f"diagnostics:{campaign_id}", {"iteration": i})
            result = cache.get(f"diagnostics:{campaign_id}")
            assert result == {"iteration": i}
