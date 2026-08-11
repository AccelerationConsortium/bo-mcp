"""The canonical operating manual as an MCP resource.

``docs://manpage`` serves the exact same Markdown the REST API exposes
at ``/manpage.md`` — both load through
:func:`bo_mcp_server.docs.load_manpage_markdown`, so MCP-native agents
and REST clients read one canonical text. Static content: no
parameters, no storage access, cannot fail.
"""

from bo_mcp_server.docs import load_manpage_markdown
from bo_mcp_server.server import mcp

MANPAGE_RESOURCE_URI = "docs://manpage"


@mcp.resource(MANPAGE_RESOURCE_URI, mime_type="text/markdown")
async def get_manpage() -> str:
    """The BO-MCP operating manual (canonical Markdown).

    Read this before operating campaigns: it documents the end-to-end
    procedure, state ownership and continuation rules, idempotency and
    error handling, and the REST ↔ MCP tool mapping.
    """
    return load_manpage_markdown()
