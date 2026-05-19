"""Response caching for expensive MCP operations.

Provides TTL-based caching to avoid redundant computations for
frequently-called tools like get_diagnostics.

Cache keys include the campaign version, so entries are automatically
stale after any mutation — no explicit invalidation needed.

Uses asyncio.Lock to ensure safe concurrent access from async operations.

The cache accepts an injectable ``clock`` callable so tests can advance time
without sleeping. Production code constructs the cache with the default clock
(``datetime.now(UTC)``); tests pass a callable backed by a mutable container
to deterministically simulate TTL expiry.

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

Rollback safety
---------------

The cache is only *written* by ``get_diagnostics_operation``; mutation
paths never touch it. ``campaign.version`` is bumped in the same
transaction as the underlying mutation, so a rolled-back mutation
leaves the version unchanged and the existing cache entry stays
semantically correct (it still represents the committed state).
Regression coverage:
``tests/unit/test_diagnostics_cache_rollback_safety.py``.
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from bo_mcp_server.settings import (
    get_diagnostics_cache_max_entries,
    get_diagnostics_cache_ttl_seconds,
)

# Compatibility alias retained for code/tests that import the symbol
# directly. New call sites should read the live value via
# :func:`bo_mcp_server.settings.get_diagnostics_cache_max_entries` so
# environment-driven tuning is observed without restarting the
# process at import time.
MAX_CACHE_ENTRIES = get_diagnostics_cache_max_entries()


def _default_clock() -> datetime:
    """Return the current UTC time. Default clock for ResponseCache."""
    return datetime.now(UTC)


class ResponseCache:
    """TTL cache for expensive computations with async-safe access.

    Default TTL is 120 seconds for diagnostics. Version-aware keys mean
    entries become unreachable (and eventually evicted) after mutations,
    so a longer TTL is safe and improves hit rate.

    The ``clock`` parameter is injectable so tests can advance time without
    relying on wall-clock sleeps. In production, leave it at the default;
    in tests, pass a callable that returns a controllable ``datetime``.
    """

    def __init__(
        self,
        ttl_seconds: int | None = None,
        clock: Callable[[], datetime] = _default_clock,
        max_entries: int | None = None,
    ) -> None:
        """Initialize cache with TTL.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds. When
                ``None``, falls back to
                :func:`bo_mcp_server.settings.get_diagnostics_cache_ttl_seconds`
                so the deployment knob is honoured at construction.
            clock: Callable returning the current ``datetime``. Tests can
                inject a fake clock to advance time deterministically.
            max_entries: Capacity before LRU-eviction kicks in. When
                ``None``, falls back to
                :func:`bo_mcp_server.settings.get_diagnostics_cache_max_entries`.
        """
        resolved_ttl = (
            ttl_seconds if ttl_seconds is not None else get_diagnostics_cache_ttl_seconds()
        )
        resolved_cap = (
            max_entries if max_entries is not None else get_diagnostics_cache_max_entries()
        )
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._ttl = timedelta(seconds=resolved_ttl)
        self._lock = asyncio.Lock()
        self._clock = clock
        self._max_entries = resolved_cap

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
                if self._clock() - timestamp < self._ttl:
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
            if len(self._cache) >= self._max_entries:
                self._evict_oldest()
            self._cache[key] = (self._clock(), value)

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


# Global cache instance for diagnostics. TTL and capacity are sourced
# from :class:`bo_mcp_server.settings.Settings` so a deployment can
# tune them via env without re-importing this module. Version-aware
# keys auto-invalidate on mutations regardless of the TTL.
diagnostics_cache = ResponseCache()
