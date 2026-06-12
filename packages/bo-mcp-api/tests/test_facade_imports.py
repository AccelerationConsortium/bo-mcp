"""Enforce the ``bo_mcp_server.client`` facade boundary.

The REST API is a transport adapter; it must import from
:mod:`bo_mcp_server.client` only. Any other ``bo_mcp_server.*`` import is a
leaked internal — storage / operations / domain refactors then ripple into
REST contracts and the boundary becomes fictitious (see project guidance on
the API/server boundary).

This guard is an **allowlist**, not a denylist. A denylist of forbidden
prefixes lets every newly-added server module through by default and
silently leaks (e.g. ``bo_mcp_server.idempotency_gc`` does not prefix-match
``bo_mcp_server.idempotency``). The allowlist inverts that: only
:mod:`bo_mcp_server.client` (and its submodules) is permitted, plus one
narrowly-scoped, documented exception:

* :mod:`bo_mcp_server.logging_config` — imported once in ``api.main`` to
  call ``configure_logging()`` *before* importing the facade, whose import
  chain emits a backend-discovery log record at import time. Routing this
  through the facade would import the facade first and defeat the ordering,
  so the direct import is intentional.

Every other transport-support symbol the API needs (idempotency GC
lifespan, trace-context binding, OpenAPI schema augmentation) is re-exported
from :mod:`bo_mcp_server.client` so the API imports it via the facade.

Test-suite files are intentionally exempt: they exercise storage fixtures
and persisted-row helpers that are not part of the public facade contract
(only ``src/api`` is scanned).

Reference: scanning project source for forbidden imports is a standard
architecture-fitness function; see e.g. ArchUnit (Java) or the
``importlinter`` Python package.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

API_SRC = Path(__file__).resolve().parent.parent / "src" / "api"

# The only ``bo_mcp_server.*`` modules the REST API may import directly. Any
# other ``bo_mcp_server`` import is a leaked internal. See the module
# docstring for why ``logging_config`` is permitted.
ALLOWED_MODULES = (
    "bo_mcp_server.client",
    "bo_mcp_server.logging_config",
)


def _iter_python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _is_bo_mcp_server_module(module: str) -> bool:
    """True when ``module`` is the server package or one of its submodules."""
    return module == "bo_mcp_server" or module.startswith("bo_mcp_server.")


def _is_allowed(module: str) -> bool:
    """True when ``module`` is the facade or a documented allowlist exception."""
    return any(module == allowed or module.startswith(allowed + ".") for allowed in ALLOWED_MODULES)


def _imported_modules(node: ast.AST) -> list[tuple[str, str]]:
    """Return ``(module, rendered)`` for each import statement on ``node``."""
    if isinstance(node, ast.ImportFrom) and node.module:
        names = ", ".join(alias.name for alias in node.names)
        return [(node.module, f"{node.lineno} from {node.module} import {names}")]
    if isinstance(node, ast.Import):
        return [(alias.name, f"{node.lineno} import {alias.name}") for alias in node.names]
    return []


def _collect_forbidden_imports(path: Path) -> list[str]:
    """Return ``bo_mcp_server`` imports in ``path`` that bypass the facade."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        for module, rendered in _imported_modules(node):
            if _is_bo_mcp_server_module(module) and not _is_allowed(module):
                offenders.append(f"{path.name}:{rendered}")
    return offenders


@pytest.mark.parametrize(
    "py_file",
    _iter_python_files(API_SRC),
    ids=lambda p: str(p.relative_to(API_SRC)),
)
def test_api_module_imports_only_from_client_facade(py_file: Path) -> None:
    """Every API source file routes ``bo_mcp_server`` access through ``.client``."""
    offenders = _collect_forbidden_imports(py_file)
    assert not offenders, (
        f"{py_file.relative_to(API_SRC)} reaches into bo_mcp_server internals. "
        "Import via bo_mcp_server.client instead (or, for bootstrap ordering, "
        "bo_mcp_server.logging_config):\n  " + "\n  ".join(offenders)
    )
