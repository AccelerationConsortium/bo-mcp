"""Backend provider for the MCP server.

Provides a single point of access to the active BO backend. Currently
hardcoded to BoTorchBackend. Step 8 P3 (future) will add a plugin
registry with entry-point discovery.

Usage:
    from bo_mcp_server.backend import get_backend
    backend = get_backend()
    result = backend.generate_suggestions(spec, observations, ...)
"""

from bo_engine.backend import BOBackend
from bo_engine.botorch_backend import BoTorchBackend

_backend: BOBackend | None = None


def get_backend() -> BOBackend:
    """Return the active BO backend (lazily initialized)."""
    global _backend  # noqa: PLW0603
    if _backend is None:
        _backend = BoTorchBackend()
    return _backend
