"""Lint contract: named offender files must not regress on hardcoded knobs.

Project guideline: tuning values must not be hardcoded in operational
code — they must live in constants or pydantic-settings so deployments
can override them via environment.

Three files are the load-bearing offenders for connection-pool sizing,
cache capacity, and idempotency TTLs:

* ``storage/database.py``      — pool_size / max_overflow / pool_recycle
* ``cache.py``                  — MAX_CACHE_ENTRIES, TTL
* ``idempotency.py``            — reservation + response TTLs

This module is a *scoped* lint: it asserts the named files contain no
bare numeric-literal arguments to the specific knobs we moved to
:class:`bo_mcp_server.settings.Settings`. A regression here means
either the value has crept back into a literal or someone has added a
new tuning knob without going through Settings — both shapes are
flagged so a reviewer can audit the decision.

A whole-repo numeric lint would fire on test fixtures, threshold
constants, and bench-mark seeds — this guard is deliberately limited
to the named offender files to keep the signal sharp.
"""

from __future__ import annotations

import re
from pathlib import Path


def _package_src() -> Path:
    """Locate ``packages/bo-mcp-server/src/bo_mcp_server``."""
    return Path(__file__).resolve().parents[2] / "src" / "bo_mcp_server"


# Patterns each offender file must NOT contain. Anchored to the
# specific kwarg / constant the audit named.
_DATABASE_FORBIDDEN: tuple[re.Pattern[str], ...] = (
    re.compile(r"pool_size\s*=\s*\d+\b"),
    re.compile(r"max_overflow\s*=\s*\d+\b"),
    re.compile(r"pool_recycle\s*=\s*\d+\b"),
)

# In cache.py, the live value goes through the Settings accessor;
# the module-level alias retained for back-compat is allowed to read
# from that accessor (it does not contain a bare literal).
_CACHE_FORBIDDEN: tuple[re.Pattern[str], ...] = (
    # ``MAX_CACHE_ENTRIES = 200`` (or any bare int) is the original
    # offender. The current source assigns from
    # ``get_diagnostics_cache_max_entries()`` instead.
    re.compile(r"^MAX_CACHE_ENTRIES\s*=\s*\d+\b", re.MULTILINE),
    # ResponseCache(ttl_seconds=120) — the global instance must no
    # longer wire a literal.
    re.compile(r"ResponseCache\([^)]*\bttl_seconds\s*=\s*\d+"),
)

# In idempotency.py, the constants ``DEFAULT_*_TTL_SECONDS`` once
# carried literal arithmetic (``24 * 60 * 60``). They now read from
# Settings accessors; reverting would silently re-hardcode the value.
_IDEMPOTENCY_FORBIDDEN: tuple[re.Pattern[str], ...] = (
    re.compile(r"^DEFAULT_IDEMPOTENCY_TTL_SECONDS\s*=\s*[\d* ]+$", re.MULTILINE),
    re.compile(r"^DEFAULT_RESERVATION_TTL_SECONDS\s*=\s*[\d* ]+$", re.MULTILINE),
)


def _assert_no_forbidden_patterns(path: Path, patterns: tuple[re.Pattern[str], ...]) -> None:
    body = path.read_text()
    for pattern in patterns:
        matches = pattern.findall(body)
        assert not matches, (
            f"{path.name}: forbidden hardcoded knob matched {pattern.pattern!r} "
            f"({len(matches)} hit(s)). Route the value through "
            f"bo_mcp_server.settings.Settings instead. Examples:\n  "
            + "\n  ".join(str(m) for m in matches[:5])
        )


def test_database_module_does_not_hardcode_pool_knobs() -> None:
    """``storage/database.py`` reads pool knobs from Settings, not literals."""
    _assert_no_forbidden_patterns(_package_src() / "storage" / "database.py", _DATABASE_FORBIDDEN)


def test_cache_module_does_not_hardcode_capacity_or_ttl() -> None:
    """``cache.py`` resolves capacity / TTL through Settings accessors."""
    _assert_no_forbidden_patterns(_package_src() / "cache.py", _CACHE_FORBIDDEN)


def test_idempotency_module_does_not_hardcode_ttls() -> None:
    """``idempotency.py`` defaults read from Settings, not literal arithmetic."""
    _assert_no_forbidden_patterns(_package_src() / "idempotency.py", _IDEMPOTENCY_FORBIDDEN)
