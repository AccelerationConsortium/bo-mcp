"""Backend provider for the MCP server.

Provides a single point of access to BO backends. Supports per-campaign
backend selection via the ``backend`` field on ``CampaignSpec``, with
``BO_BACKEND`` env var as the default.

Usage:
    from bo_mcp_server.backend import get_backend
    backend = get_backend()           # default (env var or "botorch")
    backend = get_backend("baybe")    # explicit backend name
"""

import logging
import os
from importlib.metadata import entry_points

from bo_engine.backend import BOBackend
from bo_engine.botorch_backend import BoTorchBackend

logger = logging.getLogger(__name__)

_backends: dict[str, BOBackend] = {}


def _load_backend(name: str) -> BOBackend:
    """Load a backend by name. Uses entry-point discovery for non-default backends."""
    if name == "botorch":
        return BoTorchBackend()

    eps = entry_points(group="bo_mcp.backends")
    for ep in eps:
        if ep.name == name:
            backend_class = ep.load()
            logger.info("Loaded backend '%s' via entry point: %s", name, ep.value)
            return backend_class()

    available = ["botorch"] + [ep.name for ep in eps]
    msg = f"Unknown backend '{name}'. Available: {available}"
    raise ValueError(msg)


def get_backend(name: str | None = None) -> BOBackend:
    """Return a BO backend by name (lazily initialized and cached).

    Args:
        name: Backend name. If None, uses the ``BO_BACKEND`` env var
              (default ``"botorch"``).
    """
    resolved = name or os.getenv("BO_BACKEND", "botorch")
    if resolved not in _backends:
        _backends[resolved] = _load_backend(resolved)
        logger.info("Backend initialized: %s", _backends[resolved].name)
    return _backends[resolved]
