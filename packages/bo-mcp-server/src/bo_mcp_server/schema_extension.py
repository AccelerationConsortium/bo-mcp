"""Splice typed per-backend ``parameter_options`` schema into intake schemas.

``ParameterSpec.parameter_options`` is an opaque ``dict`` keyed by backend
name. The neutral domain models cannot describe a backend's options without
importing that backend package — which would invert the dependency graph and
couple the backend-agnostic intake to BayBE. This module is the
backend-aware boundary that resolves the tension: it queries each discovered
backend's :meth:`bo_engine.backend.BOBackend.parameter_options_schema` hook
through the entry-point registry (never importing a backend package directly)
and splices the union into

* the **MCP** tool schemas (``bo_create_campaign`` / ``bo_validate_intake``)
  via :func:`intake_schema_with_parameter_options`, and
* the **REST** OpenAPI document via :func:`augment_parameter_options`,

so a client introspecting either transport discovers e.g. BayBE's
``role=substance`` molecular recipe (``substance_data`` + ``substance_encoding``)
without reading source. Both transports feed from the same once-built fragment
map, so they cannot drift.
"""

from __future__ import annotations

import copy
import logging
from functools import lru_cache
from typing import Any, cast

from bo_mcp_server.backend import get_backend, list_available_backends
from bo_mcp_server.domain.intake_models import INTAKE_INPUT_JSON_SCHEMA, inline_defs

logger = logging.getLogger(__name__)

_PARAMETER_OPTIONS_FIELD = "parameter_options"

_PARAMETER_OPTIONS_DESCRIPTION = (
    "Per-backend parameter options, keyed by backend name. Each backend reads "
    "only its own slot and silently ignores keys addressed to other backends. "
    "Documented backends appear under 'properties' with their typed shape "
    "(e.g. 'baybe' exposes 'role', for role='substance' the 'substance_data' "
    "SMILES map and 'substance_encoding', and for role='custom' the "
    "'custom_descriptors' per-label representation table and 'decorrelate'); "
    "unknown backends remain accepted via additionalProperties."
)


@lru_cache(maxsize=1)
def parameter_options_fragments() -> dict[str, dict[str, Any]]:
    """Return ``{backend_name: inlined JSON-schema fragment}`` for all backends.

    Each backend that declares typed per-parameter options (via the
    ``parameter_options_schema`` hook) contributes the schema for the value
    stored under ``parameter_options[<backend name>]``. Fragments are inlined
    (``$defs`` resolved) so they stay self-contained when spliced into a host
    schema. Backends that fail to load (missing optional dependency, broken
    entry point) or return ``None`` are skipped, so the surface degrades to
    whatever is actually installed rather than crashing schema generation.

    Cached: the registry is fixed after startup, so the fragment map is built
    once and shared by both transports.
    """
    fragments: dict[str, dict[str, Any]] = {}
    for name in list_available_backends():
        try:
            fragment = get_backend(name).parameter_options_schema()
        except (ValueError, ImportError, AttributeError) as exc:
            logger.warning(
                "Backend %r could not provide a parameter_options schema fragment: %s",
                name,
                exc,
            )
            continue
        if fragment is None:
            continue
        fragments[name] = inline_defs(fragment)
    return fragments


def _object_branches(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the object-typed subschema(s) of a ``parameter_options`` field.

    ``Optional[Mapping[...]]`` renders as ``anyOf: [object, null]``; a bare
    map renders as a single ``type: object``. Either way we return the object
    branch(es) so per-backend ``properties`` attach to the right place and the
    existing ``additionalProperties`` (unknown-backend passthrough) is kept.
    """
    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        return [m for m in any_of if isinstance(m, dict) and m.get("type") == "object"]
    if node.get("type") == "object":
        return [node]
    return []


def _augment_parameter_options_node(
    node: dict[str, Any],
    fragments: dict[str, dict[str, Any]],
) -> None:
    """Add typed per-backend ``properties`` to one ``parameter_options`` field schema."""
    if not fragments:
        return
    for obj in _object_branches(node):
        existing = obj.get("properties")
        merged: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        for name, fragment in fragments.items():
            # ``setdefault`` keeps any caller-provided override and makes the
            # injection idempotent (re-running OpenAPI generation is a no-op).
            merged.setdefault(name, copy.deepcopy(fragment))
        obj["properties"] = merged
    node.setdefault("description", _PARAMETER_OPTIONS_DESCRIPTION)


def _walk_and_augment(node: object, fragments: dict[str, dict[str, Any]]) -> None:
    """Recursively find every ``parameter_options`` field schema and augment it.

    Handles both the fully-inlined MCP schema (the field sits under
    ``parameters.items.properties``) and the ``$defs``/``components.schemas``
    form FastAPI emits (the field sits under ``InputParameter.properties``).
    ``parameter_options`` is a distinctive field name unique to the parameter
    model, so matching it directly avoids hardcoding a fragile deep path.
    """
    if isinstance(node, dict):
        node_dict = cast("dict[str, Any]", node)
        candidate = node_dict.get(_PARAMETER_OPTIONS_FIELD)
        if isinstance(candidate, dict):
            candidate_dict = cast("dict[str, Any]", candidate)
            if _object_branches(candidate_dict):
                _augment_parameter_options_node(candidate_dict, fragments)
        for value in node_dict.values():
            _walk_and_augment(value, fragments)
    elif isinstance(node, list):
        for item in cast("list[Any]", node):
            _walk_and_augment(item, fragments)


def augment_parameter_options(schema: dict[str, Any]) -> dict[str, Any]:
    """Inject the per-backend ``parameter_options`` properties into ``schema`` in place.

    Mutates and returns ``schema`` — intended for post-processing a generated
    OpenAPI document (``components/schemas/InputParameter``). Idempotent.
    """
    _walk_and_augment(schema, parameter_options_fragments())
    return schema


@lru_cache(maxsize=1)
def intake_schema_with_parameter_options() -> dict[str, Any]:
    """Return the MCP intake JSON schema enriched with per-backend options.

    A deep copy of :data:`INTAKE_INPUT_JSON_SCHEMA` (so the shared module
    global stays untouched) with the typed ``parameter_options`` properties
    spliced in. Consumed by the ``bo_create_campaign`` / ``bo_validate_intake``
    tool definitions so ``tools/list`` advertises the molecular recipe.
    """
    enriched = copy.deepcopy(INTAKE_INPUT_JSON_SCHEMA)
    _walk_and_augment(enriched, parameter_options_fragments())
    return enriched
