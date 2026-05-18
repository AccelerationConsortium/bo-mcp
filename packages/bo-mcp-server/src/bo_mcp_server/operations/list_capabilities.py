"""List capabilities operation - protocol-neutral business logic."""

from typing import Any

from bo_mcp_server import __version__
from bo_mcp_server.backend import get_backend


def list_capabilities_operation() -> dict[str, Any]:
    """List the capabilities of the active BO backend.

    ``supported_features`` lists features the backend supports
    **unconditionally** — no spec-shape preconditions, so a campaign
    that uses any of these features will always be accepted by this
    backend. ``conditional_features`` lists features the backend can
    support only when the spec meets a stated precondition (e.g.
    BayBE's TRANSFER_LEARNING requires a TaskParameter); each entry
    maps the feature name to a short human-readable description of
    the precondition so LLM clients and humans can plan against the
    real runtime contract instead of an over-advertised static
    surface.

    Returns:
        Dictionary with backend name, unconditional and conditional
        feature surfaces, and server version.
    """
    backend = get_backend()
    conditional = {
        str(feature): reason
        for feature, reason in (getattr(backend, "conditional_features", {}) or {}).items()
    }
    return {
        "backend": backend.name,
        "supported_features": sorted(backend.supported_features),
        "conditional_features": conditional,
        "server_version": __version__,
    }
