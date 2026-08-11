"""Canonical BO-MCP operating manual, shipped as package data.

``MANPAGE.md`` in this sub-package is the single narrative manual for
operating BO-MCP campaigns. It lives inside ``bo_mcp_server`` (rather
than at a repository path) so every transport that imports the shared
operations layer — the REST API, the FastMCP server, and the test
suites — loads the exact same bytes via :mod:`importlib.resources`,
whether installed from a wheel, running in Docker, or in editable mode.

This module is the single load path; do not open the file directly.
"""

from functools import cache
from importlib import resources

_MANPAGE_FILENAME = "MANPAGE.md"


@cache
def load_manpage_markdown() -> str:
    """Return the canonical manual as Markdown text.

    Cached for the process lifetime: the content is package data and
    can only change on deploy.
    """
    return resources.files(__package__).joinpath(_MANPAGE_FILENAME).read_text(encoding="utf-8")
