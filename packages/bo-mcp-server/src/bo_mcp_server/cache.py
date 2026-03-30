"""Response caching for expensive MCP operations.

Provides TTL-based caching to avoid redundant computations for
frequently-called tools like get_diagnostics.

Cache keys include the campaign version, so entries are automatically
stale after any mutation — no explicit invalidation needed.

Usage:
    from bo_mcp_server.cache import diagnostics_cache

    # Get cached value if available (version-aware key)
    cache_key = f"diagnostics:{campaign_id}:{campaign.version}"
    cached = diagnostics_cache.get(cache_key)
    if cached:
        return cached

    # Compute and cache
    result = compute_expensive_diagnostics()
    diagnostics_cache.set(cache_key, result)
"""

from datetime import datetime, timedelta
from typing import Any

# Maximum number of entries before evicting oldest
MAX_CACHE_ENTRIES = 200


class ResponseCache:
    """TTL cache for expensive computations.

    Default TTL is 120 seconds for diagnostics. Version-aware keys mean
    entries become unreachable (and eventually evicted) after mutations,
    so a longer TTL is safe and improves hit rate.

    Attributes:
        _cache: Dictionary mapping cache keys to (timestamp, value) tuples.
        _ttl: Time-to-live for cache entries as a timedelta.
    """

    def __init__(self, ttl_seconds: int = 120) -> None:
        """Initialize cache with TTL.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: str) -> Any | None:
        """Get cached value if not expired.

        Args:
            key: Cache key (e.g. "diagnostics:{campaign_id}:{version}").

        Returns:
            Cached value if present and not expired, None otherwise.
        """
        if key in self._cache:
            timestamp, value = self._cache[key]
            if datetime.now() - timestamp < self._ttl:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache, evicting oldest entries if at capacity.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        if len(self._cache) >= MAX_CACHE_ENTRIES:
            self._evict_oldest()
        self._cache[key] = (datetime.now(), value)

    def _evict_oldest(self) -> None:
        """Remove the oldest cache entry."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
        del self._cache[oldest_key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)


# Global cache instance for diagnostics
# 120s TTL — version-aware keys auto-invalidate on mutations
diagnostics_cache = ResponseCache(ttl_seconds=120)
