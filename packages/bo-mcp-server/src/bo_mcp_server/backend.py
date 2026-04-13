"""Backend provider for the MCP server.

Provides a single point of access to BO backends. Supports per-campaign
backend selection via the ``backend`` field on ``CampaignSpec``, with
``BO_BACKEND`` env var as the default.

Usage:
    from bo_mcp_server.backend import get_backend, resolve_backend_name
    backend = get_backend()           # default (env var or "botorch")
    backend = get_backend("baybe")    # explicit backend name
    name = resolve_backend_name("auto", spec_dict)  # auto-select
"""

import logging
import os
from importlib.metadata import entry_points
from typing import Any

from bo_engine.backend import BOBackend, Feature

logger = logging.getLogger(__name__)

_backends: dict[str, BOBackend] = {}

DEFAULT_BACKEND = "botorch"
_ENTRY_POINT_GROUP = "bo_mcp.backends"


def _load_backend(name: str) -> BOBackend:
    """Load a backend by name via entry-point discovery."""
    eps = entry_points(group=_ENTRY_POINT_GROUP)
    for ep in eps:
        if ep.name == name:
            backend_class = ep.load()
            logger.info("Loaded backend '%s' via entry point: %s", name, ep.value)
            return backend_class()

    available = [ep.name for ep in eps]
    msg = f"Unknown backend '{name}'. Available: {available}"
    raise ValueError(msg)


def _get_available_backend_names() -> list[str]:
    """Return names of all installed backends."""
    eps = entry_points(group=_ENTRY_POINT_GROUP)
    return [ep.name for ep in eps]


# Spec keys that directly map to a required feature when truthy.
_SPEC_KEY_TO_FEATURE: dict[str, Feature] = {
    "constraints": Feature.CONSTRAINTS,
    "outcome_constraints": Feature.OUTCOME_CONSTRAINTS,
    "use_cost_aware": Feature.COST_AWARE,
    "turbo_config": Feature.HIGH_DIMENSIONAL,
    "fidelity_parameter": Feature.MULTI_FIDELITY,
    "transfer_learning": Feature.TRANSFER_LEARNING,
    "use_input_warping": Feature.INPUT_WARPING,
}


def _detect_required_features(spec_dict: dict[str, Any]) -> frozenset[Feature]:
    """Detect which backend features a problem specification requires."""
    features: set[Feature] = set()

    if len(spec_dict.get("objectives", [])) > 1:
        features.add(Feature.MULTI_OBJECTIVE)

    # Parameter types
    param_types = {p.get("type", "") for p in spec_dict.get("parameters", [])}
    if "categorical" in param_types:
        features.add(Feature.CATEGORICAL)
    if "categorical" in param_types and param_types & {"continuous", "discrete"}:
        features.add(Feature.MIXED_SEARCH_SPACE)

    # Simple key-to-feature mapping
    for key, feature in _SPEC_KEY_TO_FEATURE.items():
        if spec_dict.get(key):
            features.add(feature)

    return frozenset(features)


def resolve_backend_name(name: str, spec_dict: dict[str, Any]) -> str:
    """Resolve a backend name, handling 'auto' by inspecting the spec.

    When name is 'auto', selects a backend whose supported_features cover
    all features the problem requires. Prefers the BO_BACKEND env var
    default when multiple backends qualify.

    Args:
        name: Backend name or 'auto'.
        spec_dict: Campaign spec as a dictionary.

    Returns:
        Concrete backend name (never 'auto').
    """
    if name != "auto":
        return name

    required = _detect_required_features(spec_dict)
    env_default = os.getenv("BO_BACKEND", DEFAULT_BACKEND)

    # Try env-var default first
    try:
        default_backend = get_backend(env_default)
        if required <= default_backend.supported_features:
            logger.info(
                "Auto-selected backend '%s' (env default, supports %s)",
                env_default,
                required,
            )
            return env_default
    except ValueError:
        pass

    # Try all available backends
    for backend_name in _get_available_backend_names():
        try:
            backend = get_backend(backend_name)
            if required <= backend.supported_features:
                logger.info(
                    "Auto-selected backend '%s' (supports %s)",
                    backend_name,
                    required,
                )
                return backend_name
        except ValueError:
            continue

    # Fallback — botorch supports everything
    logger.warning("No backend fully supports %s, falling back to '%s'", required, DEFAULT_BACKEND)
    return DEFAULT_BACKEND


def get_backend(name: str | None = None) -> BOBackend:
    """Return a BO backend by name (lazily initialized and cached).

    Args:
        name: Backend name. If None, uses the ``BO_BACKEND`` env var
              (default ``"botorch"``).
    """
    resolved = name or os.getenv("BO_BACKEND", DEFAULT_BACKEND)
    if resolved not in _backends:
        _backends[resolved] = _load_backend(resolved)
        logger.info("Backend initialized: %s", _backends[resolved].name)
    return _backends[resolved]
