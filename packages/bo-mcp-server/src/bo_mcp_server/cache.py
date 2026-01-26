"""Response caching for expensive MCP operations.

Provides TTL-based caching to avoid redundant computations for
frequently-called tools like get_diagnostics.

Usage:
    from bo_mcp_server.cache import diagnostics_cache

    # Get cached value if available
    cached = diagnostics_cache.get(f"diagnostics:{campaign_id}")
    if cached:
        return cached

    # Compute and cache
    result = compute_expensive_diagnostics()
    diagnostics_cache.set(f"diagnostics:{campaign_id}", result)

    # Invalidate after mutations
    diagnostics_cache.invalidate(campaign_id)
"""

from datetime import datetime, timedelta
from typing import Any


class ResponseCache:
    """TTL cache for expensive computations.

    Default TTL is 30 seconds for diagnostics - balances freshness
    with avoiding redundant computation during tight optimization loops.

    Attributes:
        _cache: Dictionary mapping cache keys to (timestamp, value) tuples.
        _ttl: Time-to-live for cache entries as a timedelta.
    """

    def __init__(self, ttl_seconds: int = 30) -> None:
        """Initialize cache with TTL.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: str) -> Any | None:
        """Get cached value if not expired.

        Args:
            key: Cache key (typically "diagnostics:{campaign_id}").

        Returns:
            Cached value if present and not expired, None otherwise.
        """
        if key in self._cache:
            timestamp, value = self._cache[key]
            if datetime.now() - timestamp < self._ttl:
                return value
            # Entry expired, remove it
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        self._cache[key] = (datetime.now(), value)

    def invalidate(self, campaign_id: str) -> None:
        """Invalidate all cache entries for a campaign.

        Call this after mutations (submit_results, generate_suggestions).

        Args:
            campaign_id: Campaign ID to invalidate entries for.
        """
        to_delete = [k for k in self._cache if campaign_id in k]
        for k in to_delete:
            del self._cache[k]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)


# Global cache instance for diagnostics
# 30 second TTL balances freshness with avoiding redundant computation
diagnostics_cache = ResponseCache(ttl_seconds=30)
