"""Lint contract: bare ``except Exception`` requires an attached justification.

The project guideline in :doc:`/CLAUDE.md` requires specific exceptions;
when the call site genuinely needs a catch-all (translator boundaries,
best-effort introspection, cleanup-then-reraise), the rule requires an
inline ``# noqa: BLE001 - <reason>`` so a reviewer can audit the
decision without re-discovering the rationale.

This test is a static scan over the package source — it is the lint
gate the audit (TODO 8.57) prescribed: any new bare-catch added
without an attached justification breaks the build, so the contract
cannot silently regress.

Scope: the scan walks the ``src/bo_mcp_server`` tree only. Tests can
catch broadly without a justification because their failure surface
is different (a flaky assertion is fixed by the next CI run rather
than swallowing a programming bug in production).
"""

from __future__ import annotations

import re
from pathlib import Path

# Match ``except Exception`` or ``except BaseException`` at line scope,
# optionally followed by ``as <name>:`` so we catch both shapes. The
# anchor on ``:`` keeps us from matching exception class definitions.
_BARE_EXCEPT = re.compile(r"\bexcept\s+(Exception|BaseException)(\s+as\s+\w+)?\s*:")
# Justifying noqa must include both the ``BLE001`` code *and* a
# trailing ``- <reason>`` clause. ``noqa: BLE001`` alone is rejected
# so the contract cannot silently regress to a code-only suppression.
# The ``[\w, ]*`` permits adjacent codes (e.g. ``noqa: BLE001, S110``)
# before the dash and reason text.
_BLE_NOQA = re.compile(r"noqa:\s*[\w, ]*\bBLE001\b[\w, ]*\s*-\s*\S")


def _server_src_root() -> Path:
    """Locate ``packages/bo-mcp-server/src`` from the test file."""
    return Path(__file__).resolve().parents[2] / "src"


def _api_src_root() -> Path | None:
    """Locate ``packages/bo-mcp-api/src`` if it is available alongside."""
    candidate = Path(__file__).resolve().parents[3] / "bo-mcp-api" / "src"
    return candidate if candidate.exists() else None


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _find_unjustified_bare_excepts(roots: list[Path]) -> list[tuple[Path, int, str]]:
    """Return ``(path, lineno, line)`` for each unjustified bare except."""
    offenders: list[tuple[Path, int, str]] = []
    for root in roots:
        for path in _iter_python_files(root):
            for idx, line in enumerate(path.read_text().splitlines(), start=1):
                if not _BARE_EXCEPT.search(line):
                    continue
                if _BLE_NOQA.search(line):
                    continue
                offenders.append((path, idx, line.strip()))
    return offenders


def test_no_unjustified_bare_except_exception() -> None:
    """Every ``except Exception`` in production source carries a justification.

    The contract is enforced by inline ``# noqa: BLE001 - <reason>``;
    reviewers should reject any new offender that does not explain
    *why* the catch-all is the right shape (translator boundary,
    cleanup-then-reraise, best-effort introspection, etc.).
    """
    roots = [_server_src_root()]
    api_root = _api_src_root()
    if api_root is not None:
        roots.append(api_root)

    offenders = _find_unjustified_bare_excepts(roots)
    formatted = "\n".join(f"  {path}:{lineno}: {line}" for path, lineno, line in offenders)
    assert not offenders, (
        "Bare 'except Exception' without an attached '# noqa: BLE001 - reason' "
        "justification was found in production source. Either narrow the catch "
        "to a specific exception type, or add the inline noqa with a reason. "
        f"Offenders:\n{formatted}"
    )


def test_regex_rejects_noqa_without_reason() -> None:
    """A bare ``noqa: BLE001`` (no ``- reason``) is treated as unjustified.

    Without this assertion the test message — which promises a
    ``- reason`` clause — would be a lie: the regex used to match any
    noqa with the BLE001 code, regardless of whether a justification
    followed. Pinning the negative case keeps the contract honest.
    """
    code_only = "except Exception:  # noqa: BLE001"
    code_and_reason = "except Exception:  # noqa: BLE001 - cleanup-then-reraise"
    bare = "except Exception:"

    assert _BARE_EXCEPT.search(code_only)
    assert _BARE_EXCEPT.search(code_and_reason)
    assert _BARE_EXCEPT.search(bare)

    # Code without a justification fails the noqa matcher.
    assert not _BLE_NOQA.search(code_only)
    # Code WITH a trailing reason passes.
    assert _BLE_NOQA.search(code_and_reason)
    # Adjacent codes plus a reason also pass.
    multi = "except Exception:  # noqa: BLE001, S110 - best-effort introspection"
    assert _BLE_NOQA.search(multi)
    # No noqa at all is obviously rejected.
    assert not _BLE_NOQA.search(bare)
