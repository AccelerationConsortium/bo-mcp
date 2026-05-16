"""Enforce the ``bo_mcp_server.client`` facade boundary.

The REST API is a transport adapter; it must import from
:mod:`bo_mcp_server.client` only. Direct imports from
:mod:`bo_mcp_server.storage`, :mod:`bo_mcp_server.operations`,
:mod:`bo_mcp_server.domain`, :mod:`bo_mcp_server.errors`,
:mod:`bo_mcp_server.response_formatter`, or
:mod:`bo_mcp_server.result_upload_parser` indicate a leaked internal —
storage refactors then ripple into REST contracts and the boundary
becomes fictitious (see project guidance on the API/server boundary).

Test-suite files are intentionally exempt: they exercise storage
fixtures and persisted-row helpers that are not part of the public
facade contract.

Reference: scanning project source for forbidden imports is a standard
architecture-fitness function; see e.g. ArchUnit (Java) or the
``importlinter`` Python package.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

API_SRC = Path(__file__).resolve().parent.parent / "src" / "api"

# Every internal submodule the API must not import from directly.
FORBIDDEN_PREFIXES = (
    "bo_mcp_server.storage",
    "bo_mcp_server.operations",
    "bo_mcp_server.domain",
    "bo_mcp_server.errors",
    "bo_mcp_server.response_formatter",
    "bo_mcp_server.result_upload_parser",
    "bo_mcp_server.tools",
    "bo_mcp_server.resources",
    "bo_mcp_server.converters",
    "bo_mcp_server.audit",
    "bo_mcp_server.cache",
    "bo_mcp_server.idempotency",
    "bo_mcp_server.pagination",
    "bo_mcp_server.progress_bridge",
    "bo_mcp_server.constants",
)


def _iter_python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _collect_forbidden_imports(path: Path) -> list[str]:
    """Return the offending ``from ... import ...`` statements in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for prefix in FORBIDDEN_PREFIXES:
                if node.module == prefix or node.module.startswith(prefix + "."):
                    names = ", ".join(alias.name for alias in node.names)
                    offenders.append(f"{path.name}:{node.lineno} from {node.module} import {names}")
                    break
        elif isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in FORBIDDEN_PREFIXES:
                    if alias.name == prefix or alias.name.startswith(prefix + "."):
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
                        break
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
        "Import via bo_mcp_server.client instead:\n  " + "\n  ".join(offenders)
    )
