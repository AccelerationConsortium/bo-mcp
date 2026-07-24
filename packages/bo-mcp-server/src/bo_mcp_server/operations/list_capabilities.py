"""List capabilities operation - protocol-neutral business logic."""

from typing import Any

from bo_mcp_server import __version__
from bo_mcp_server.backend import get_backend, list_available_backends
from bo_mcp_server.backend_context import campaign_backend_scope
from bo_mcp_server.response_formatter import attach_response_metadata
from bo_mcp_server.settings import get_default_backend_name


def list_capabilities_operation(backend_name: str | None = None) -> dict[str, Any]:
    """List the capabilities of a BO backend.

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

    Args:
        backend_name: Backend to report on. ``None`` reports the
            default backend. Unknown names raise ``ValueError`` (the
            same typed error ``get_backend`` uses everywhere).

    Returns:
        Dictionary with backend name, unconditional and conditional
        feature surfaces, the installed backend list, the default
        backend name, and server version.
    """
    backend = get_backend(backend_name)
    conditional = {
        str(feature): reason
        for feature, reason in (getattr(backend, "conditional_features", {}) or {}).items()
    }
    # Campaign-agnostic stamp: shield ``_metadata`` from a binding leaked
    # by a previous same-task operation — in-process callers may bypass
    # the transports' campaign_backend_scope (issue #82).
    with campaign_backend_scope():
        return attach_response_metadata(
            {
                "backend": backend.name,
                "supported_features": sorted(backend.supported_features),
                "conditional_features": conditional,
                "available_backends": list_available_backends(),
                "default_backend": get_default_backend_name(),
                "server_version": __version__,
            }
        )
