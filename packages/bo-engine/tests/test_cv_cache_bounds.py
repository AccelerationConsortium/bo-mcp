"""Bounds, eviction, and synchronization of the module-level CV cache.

The CV result cache previously grew without limit in long-lived server
processes: entries were only checked for expiry on a key *hit* and
never removed, and concurrent worker threads (the server offloads CV
via ``asyncio.to_thread``) mutated the dict unsynchronized. The cache
now evicts expired entries on every insert, enforces
``CV_CACHE_MAX_ENTRIES`` (earliest-expiry evicted first), and guards
all access with a lock.

Reference: this is the standard TTL+size-bounded cache contract — see
e.g. functools.lru_cache's maxsize semantics
(https://docs.python.org/3/library/functools.html#functools.lru_cache)
for the bounded-cache precedent; TTL eviction on write keeps memory
proportional to the live working set rather than the process lifetime.
"""

from __future__ import annotations

import threading

import pytest

from bo_engine import cross_validation
from bo_engine.cross_validation import (
    CVConfig,
    CVMetrics,
    _check_cv_cache,
    _store_cv_cache,
    clear_cv_cache,
)


def _metrics(tag: str) -> CVMetrics:
    return CVMetrics(
        rmse=1.0,
        mae=1.0,
        r_squared=0.5,
        mean_standardized_error=0.0,
        coverage_95=0.95,
        per_fold_errors=[],
        computation_time=0.0,
        method=tag,
    )


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_cv_cache()


def test_expired_entries_are_evicted_on_insert() -> None:
    """A dead entry is removed by the next insert, not kept until a hit."""
    _store_cv_cache("stale", _metrics("stale"), cache_ttl=-1.0)  # already expired
    assert "stale" in cross_validation._cv_cache

    _store_cv_cache("fresh", _metrics("fresh"), cache_ttl=300.0)

    assert "stale" not in cross_validation._cv_cache
    assert "fresh" in cross_validation._cv_cache


def test_size_cap_evicts_entries_closest_to_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cross_validation, "CV_CACHE_MAX_ENTRIES", 3)

    for i in range(3):
        _store_cv_cache(f"key-{i}", _metrics(f"m{i}"), cache_ttl=100.0 + i)
    _store_cv_cache("key-new", _metrics("new"), cache_ttl=300.0)

    assert len(cross_validation._cv_cache) == 3
    # key-0 had the earliest expiry and is the one dropped.
    assert "key-0" not in cross_validation._cv_cache
    assert "key-new" in cross_validation._cv_cache


def test_expired_hit_is_removed_and_reported_as_miss() -> None:
    import torch

    train_x = torch.zeros(3, 1)
    train_y = torch.zeros(3, 1)
    bounds = torch.tensor([[0.0], [1.0]])
    config = CVConfig(cache_ttl=-1.0)  # everything inserted is instantly stale

    key = cross_validation._compute_cache_key(train_x, train_y, bounds, config, None)
    _store_cv_cache(key, _metrics("stale"), cache_ttl=config.cache_ttl)

    returned_key, cached = _check_cv_cache(train_x, train_y, bounds, config)

    assert returned_key == key
    assert cached is None
    assert key not in cross_validation._cv_cache


def test_concurrent_inserts_stay_within_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel writers never corrupt the dict or overshoot the cap."""
    monkeypatch.setattr(cross_validation, "CV_CACHE_MAX_ENTRIES", 8)

    def _writer(worker: int) -> None:
        for i in range(50):
            _store_cv_cache(f"w{worker}-k{i}", _metrics("m"), cache_ttl=300.0)

    threads = [threading.Thread(target=_writer, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(cross_validation._cv_cache) <= 8
