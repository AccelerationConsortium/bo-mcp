"""Backend provider for the MCP server.

Provides a single point of access to BO backends. Supports per-campaign
backend selection via the ``backend`` field on ``CampaignSpec``, with
``BO_BACKEND`` env var as the default.

Usage:
    from bo_mcp_server.backend import get_backend, resolve_backend_name
    backend = get_backend()           # default (env var or "botorch")
    backend = get_backend("baybe")    # explicit backend name
    name = resolve_backend_name("auto", spec_dict)  # auto-select

Entry-point discovery runs once at module import. The resulting list is
cached in :data:`_discovered_entry_points` so :func:`get_backend` and
:func:`resolve_backend_name` never repeat the scan. The module also
asserts at least one backend is installed — a missing entry-point
registration would otherwise surface as a confusing "backend not found"
error deep inside the first suggestion call.
"""

import logging
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from bo_engine.backend import BOBackend, Feature
from bo_engine.backend_base import required_features

from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import CampaignSpec
from bo_mcp_server.settings import get_default_backend_name

logger = logging.getLogger(__name__)

_backends: dict[str, BOBackend] = {}

DEFAULT_BACKEND = "botorch"
_ENTRY_POINT_GROUP = "bo_mcp.backends"


def _scan_entry_points() -> tuple[EntryPoint, ...]:
    """Scan installed packages for backend entry points (module load only)."""
    return tuple(entry_points(group=_ENTRY_POINT_GROUP))


# Cached entry-point discovery. Populated at module import so subsequent
# ``get_backend`` / ``resolve_backend_name`` calls do not re-scan the
# installed package metadata. Tests that monkey-patch ``entry_points``
# can call :func:`_refresh_discovered_backends` to invalidate.
_discovered_entry_points: tuple[EntryPoint, ...] = _scan_entry_points()


def _refresh_discovered_backends() -> None:
    """Re-run entry-point discovery; intended for tests that mock entry points."""
    global _discovered_entry_points  # noqa: PLW0603
    _discovered_entry_points = _scan_entry_points()


def _load_backend(name: str) -> BOBackend:
    """Load a backend by name via entry-point discovery (cached scan)."""
    for ep in _discovered_entry_points:
        if ep.name == name:
            backend_class = ep.load()
            logger.info("Loaded backend '%s' via entry point: %s", name, ep.value)
            return backend_class()

    available = [ep.name for ep in _discovered_entry_points]
    msg = f"Unknown backend '{name}'. Available: {available}"
    raise ValueError(msg)


def _get_available_backend_names() -> list[str]:
    """Return names of all installed backends (cached scan)."""
    return [ep.name for ep in _discovered_entry_points]


def list_available_backends() -> list[str]:
    """Public accessor for the discovered backend names.

    The list is built once at module import (see
    :data:`_discovered_entry_points`) so repeated calls are O(1) and
    safe to use inside health-check responses.
    """
    return _get_available_backend_names()


def get_backend_capabilities() -> dict[str, dict[str, Any]]:
    """Return capability metadata for every discovered backend.

    Used by the health endpoint to advertise which backends are actually
    available at runtime. Each entry exposes the backend's display name
    and the string values of its declared :class:`Feature` set; loading
    failures (missing optional dependency, broken entry point) are
    captured under ``error`` so the health response surfaces the
    misconfiguration instead of silently dropping the backend.
    """
    capabilities: dict[str, dict[str, Any]] = {}
    for name in _get_available_backend_names():
        try:
            backend = get_backend(name)
        except (ValueError, ImportError) as exc:
            capabilities[name] = {"loaded": False, "error": str(exc)}
            continue
        capabilities[name] = {
            "loaded": True,
            "name": backend.name,
            "features": sorted(f.value for f in backend.supported_features),
        }
    return capabilities


def _assert_backend_discovered() -> None:
    """Fail loudly when no backends are installed.

    Raised at module import so a broken install surfaces immediately at
    startup rather than deep inside the first suggestion call. The
    discovered names are also logged so operators can see which backends
    are wired up without inspecting the package metadata directly.
    """
    names = _get_available_backend_names()
    if not names:
        msg = (
            "No bo-mcp backends discovered. Ensure at least one package "
            f"registers an entry point under '{_ENTRY_POINT_GROUP}' "
            "(e.g. bo-engine for the BoTorch backend)."
        )
        raise RuntimeError(msg)
    logger.info("Discovered bo-mcp backends: %s", names)


_assert_backend_discovered()


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


class _CompatibilityTier:
    """Auto-selection preference tiers for ``backend="auto"``.

    A single boolean "compatible" signal conflates two separate
    questions: "can this backend run the spec at all" vs "would this
    backend honor every active option in the spec". ``IGNORED`` reports
    answer "yes to the first, no to the second" — fine for explicit
    backend selection ("I asked for BayBE, I accept the warning"), but
    misleading for ``backend="auto"`` because the selector would pick
    BayBE for ``use_input_warping=True`` even though BoTorch could
    actually honor it.

    The selector therefore ranks candidates: ``FULL`` (compatible AND
    no IGNORED reports) wins over ``DEGRADED`` (compatible WITH IGNORED
    reports). ``INCOMPATIBLE`` is filtered out entirely.
    """

    FULL = "full"
    DEGRADED = "degraded"
    INCOMPATIBLE = "incompatible"


def _backend_compatibility_tier(backend: BOBackend, spec_dict: dict[str, Any]) -> str:
    """Classify a backend's fit for the spec into one of three tiers.

    Prefers :meth:`BOBackend.validate_capabilities` so backends can veto
    based on per-option detail (e.g. BayBE rejecting hybrid constraints
    even though ``Feature.CONSTRAINTS`` is in its broad set), and so the
    auto-selector can spot ``IGNORED`` reports — which mean "I'll silently
    drop part of the spec" rather than "I fully support this".

    Falls back to the historical ``required_features <= supported_features``
    check (always FULL when satisfied) when the spec dict cannot be
    coerced or the backend has not implemented the new method.
    """
    opt_spec = _spec_dict_to_optimization_spec(spec_dict)
    if opt_spec is not None:
        try:
            result = backend.validate_capabilities(opt_spec)
        except (AttributeError, NotImplementedError):
            result = None
        if result is not None:
            if not result.is_compatible:
                return _CompatibilityTier.INCOMPATIBLE
            has_ignored = any(
                r.status.value == "ignored"
                for r in (*result.feature_reports, *result.option_reports)
            )
            return _CompatibilityTier.DEGRADED if has_ignored else _CompatibilityTier.FULL

    required = _detect_required_features(spec_dict)
    if required <= backend.supported_features:
        return _CompatibilityTier.FULL
    return _CompatibilityTier.INCOMPATIBLE


def _backend_is_compatible(backend: BOBackend, spec_dict: dict[str, Any]) -> bool:
    """Boolean compatibility — kept for callers that don't care about tiers."""
    return _backend_compatibility_tier(backend, spec_dict) != _CompatibilityTier.INCOMPATIBLE


def _candidate_backends(env_default: str) -> list[str]:
    """Yield candidate backends in selection order.

    Env default goes first (preserves the deploy-time preference), then
    every other installed backend in entry-point order. Duplicates are
    removed.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for name in (env_default, *_get_available_backend_names()):
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def resolve_backend_name(name: str, spec_dict: dict[str, Any]) -> str:
    """Resolve a backend name, handling 'auto' by inspecting the spec.

    When name is 'auto', the selector classifies each candidate backend
    into a tier:

    * ``FULL`` — :meth:`BackendValidationResult.is_compatible` is true
      AND there are no ``IGNORED`` reports. The backend will honor
      every active option in the spec.
    * ``DEGRADED`` — compatible, but the backend will silently ignore
      one or more BoTorch-only knobs (``use_input_warping``,
      ``turbo_config``, …).
    * ``INCOMPATIBLE`` — filtered out.

    ``FULL`` candidates win over ``DEGRADED`` candidates so an
    auto-selected backend never quietly drops options another installed
    backend could honor. Within each tier, the ``BO_BACKEND`` env var
    default wins ties.

    Args:
        name: Backend name or 'auto'.
        spec_dict: Campaign spec as a dictionary.

    Returns:
        Concrete backend name (never 'auto').
    """
    if name != "auto":
        return name

    env_default = get_default_backend_name()
    full: list[str] = []
    degraded: list[str] = []
    for backend_name in _candidate_backends(env_default):
        try:
            backend = get_backend(backend_name)
        except ValueError:
            continue
        tier = _backend_compatibility_tier(backend, spec_dict)
        if tier == _CompatibilityTier.FULL:
            full.append(backend_name)
        elif tier == _CompatibilityTier.DEGRADED:
            degraded.append(backend_name)

    if full:
        chosen = full[0]
        logger.info(
            "Auto-selected backend '%s' (full support)%s",
            chosen,
            " — env default" if chosen == env_default else "",
        )
        return chosen
    if degraded:
        chosen = degraded[0]
        logger.info(
            "Auto-selected backend '%s' (degraded; some options will be ignored)",
            chosen,
        )
        return chosen

    logger.warning("No backend fully supports spec, falling back to '%s'", DEFAULT_BACKEND)
    return DEFAULT_BACKEND


def get_backend(name: str | None = None) -> BOBackend:
    """Return a BO backend by name (lazily initialized and cached).

    Args:
        name: Backend name. If None, uses the ``BO_BACKEND`` env var
              (default ``"botorch"``).
    """
    resolved = name or get_default_backend_name()
    if resolved not in _backends:
        _backends[resolved] = _load_backend(resolved)
        logger.info("Backend initialized: %s", _backends[resolved].name)
    return _backends[resolved]
