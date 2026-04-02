"""Response caching for expensive MCP operations.

Provides TTL-based caching to avoid redundant computations for
frequently-called tools like get_diagnostics.

Cache keys include the campaign version, so entries are automatically
stale after any mutation — no explicit invalidation needed.

Uses asyncio.Lock to ensure safe concurrent access from async operations.

Usage:
    from bo_mcp_server.cache import diagnostics_cache

    # Get cached value if available (version-aware key)
    cache_key = f"diagnostics:{campaign_id}:{campaign.version}"
    cached = await diagnostics_cache.get(cache_key)
    if cached:
        return cached

    # Compute and cache
    result = compute_expensive_diagnostics()
    await diagnostics_cache.set(cache_key, result)
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

# Maximum number of entries before evicting oldest
MAX_CACHE_ENTRIES = 200


class ResponseCache:
    """TTL cache for expensive computations with async-safe access.

    Default TTL is 120 seconds for diagnostics. Version-aware keys mean
    entries become unreachable (and eventually evicted) after mutations,
    so a longer TTL is safe and improves hit rate.
    """

    def __init__(self, ttl_seconds: int = 120) -> None:
        """Initialize cache with TTL.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get cached value if not expired.

        Args:
            key: Cache key (e.g. "diagnostics:{campaign_id}:{version}").

        Returns:
            Cached value if present and not expired, None otherwise.
        """
        async with self._lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if datetime.now(UTC) - timestamp < self._ttl:
                    return value
                del self._cache[key]
            return None

    async def set(self, key: str, value: Any) -> None:
        """Store value in cache, evicting oldest entries if at capacity.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        async with self._lock:
            if len(self._cache) >= MAX_CACHE_ENTRIES:
                self._evict_oldest()
            self._cache[key] = (datetime.now(UTC), value)

    def _evict_oldest(self) -> None:
        """Remove the oldest cache entry. Must be called under lock."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
        del self._cache[oldest_key]

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)


# Global cache instance for diagnostics
# 120s TTL — version-aware keys auto-invalidate on mutations
diagnostics_cache = ResponseCache(ttl_seconds=120)
