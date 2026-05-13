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
from bo_engine.backend_base import required_features

from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import CampaignSpec

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


def _spec_dict_to_optimization_spec(spec_dict: dict[str, Any]):
    """Best-effort coercion of a raw spec dict to an OptimizationSpec.

    Used by :func:`resolve_backend_name` so the backend can answer
    ``validate_capabilities`` against the same typed problem the engine
    will receive. Falls back to the dict-only feature inference when
    coercion fails (e.g. partially-validated specs in legacy tests).
    """
    try:
        domain_spec = CampaignSpec.model_validate(spec_dict)
        return campaign_spec_to_optimization_spec(domain_spec)
    except (TypeError, ValueError):
        return None


def _detect_required_features(spec_dict: dict[str, Any]) -> frozenset[Feature]:
    """Detect which backend features a problem specification requires.

    Routes through :func:`bo_engine.backend_base.required_features` when
    the spec coerces cleanly so the spec → feature mapping stays in one
    place. Falls back to dict-only inference for the legacy auto-select
    callers that pass partially-validated payloads.
    """
    opt_spec = _spec_dict_to_optimization_spec(spec_dict)
    if opt_spec is not None:
        return required_features(opt_spec)

    features: set[Feature] = set()
    if len(spec_dict.get("objectives", [])) > 1:
        features.add(Feature.MULTI_OBJECTIVE)
    param_types = {p.get("type", "") for p in spec_dict.get("parameters", [])}
    if "categorical" in param_types:
        features.add(Feature.CATEGORICAL)
    if "categorical" in param_types and param_types & {"continuous", "discrete"}:
        features.add(Feature.MIXED_SEARCH_SPACE)
    for key, feature in _LEGACY_KEY_TO_FEATURE.items():
        if spec_dict.get(key):
            features.add(feature)
    return frozenset(features)


# Used only in the legacy fallback path of ``_detect_required_features``.
_LEGACY_KEY_TO_FEATURE: dict[str, Feature] = {
    "constraints": Feature.CONSTRAINTS,
    "outcome_constraints": Feature.OUTCOME_CONSTRAINTS,
    "use_cost_aware": Feature.COST_AWARE,
    "turbo_config": Feature.HIGH_DIMENSIONAL,
    "fidelity_parameter": Feature.MULTI_FIDELITY,
    "transfer_learning": Feature.TRANSFER_LEARNING,
    "use_input_warping": Feature.INPUT_WARPING,
}


def _backend_is_compatible(backend: BOBackend, spec_dict: dict[str, Any]) -> bool:
    """Spec-aware compatibility check used during auto-selection.

    Prefers :meth:`BOBackend.validate_capabilities` so backends can veto
    based on per-option detail (e.g. BayBE rejecting hybrid constraints
    even though ``Feature.CONSTRAINTS`` is in its broad set). Falls back
    to the historical ``required_features <= supported_features`` check
    when the spec dict cannot be coerced or the backend has not
    implemented the new method.
    """
    opt_spec = _spec_dict_to_optimization_spec(spec_dict)
    if opt_spec is not None:
        try:
            return backend.validate_capabilities(opt_spec).is_compatible
        except (AttributeError, NotImplementedError):
            pass
    required = _detect_required_features(spec_dict)
    return required <= backend.supported_features


def resolve_backend_name(name: str, spec_dict: dict[str, Any]) -> str:
    """Resolve a backend name, handling 'auto' by inspecting the spec.

    When name is 'auto', selects a backend whose
    :meth:`BOBackend.validate_capabilities` reports ``is_compatible`` for
    the concrete spec. Prefers the BO_BACKEND env var default when
    multiple backends qualify.

    Args:
        name: Backend name or 'auto'.
        spec_dict: Campaign spec as a dictionary.

    Returns:
        Concrete backend name (never 'auto').
    """
    if name != "auto":
        return name

    env_default = os.getenv("BO_BACKEND", DEFAULT_BACKEND)

    # Try env-var default first
    try:
        default_backend = get_backend(env_default)
        if _backend_is_compatible(default_backend, spec_dict):
            logger.info("Auto-selected backend '%s' (env default)", env_default)
            return env_default
    except ValueError:
        pass

    # Try all available backends
    for backend_name in _get_available_backend_names():
        try:
            backend = get_backend(backend_name)
            if _backend_is_compatible(backend, spec_dict):
                logger.info("Auto-selected backend '%s'", backend_name)
                return backend_name
        except ValueError:
            continue

    logger.warning("No backend fully supports spec, falling back to '%s'", DEFAULT_BACKEND)
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
