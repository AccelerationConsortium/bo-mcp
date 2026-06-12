"""REST success envelopes carry ``schema_version``.

The MCP envelope already advertises the top-level ``schema_version`` so
older clients can detect a backwards-incompatible contract change
without parsing the body. The REST surface mirrors the same contract:
every success-shape Pydantic model inherits from
:class:`api.schemas.common.ResponseEnvelope`, so the rendered JSON
carries the same integer the MCP path emits.

Reference: the single-integer-handshake pattern is documented at
https://stripe.com/docs/api/versioning. The REST and MCP transports
both bump in lockstep.
"""

from __future__ import annotations

import pytest

from bo_mcp_server.tools.create_campaign import create_campaign


async def _create_campaign_for_owner(owner_id: str, name: str = "Schema Version") -> str:
    result = await create_campaign(
        {
            "name": name,
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        },
        owner_id,
    )
    return result["campaign_id"]


@pytest.mark.asyncio
async def test_create_campaign_response_includes_schema_version(
    api_client, auth_headers, persisted_user
) -> None:
    """``POST /api/campaigns`` echoes the REST schema version."""
    from api.schemas.common import API_RESPONSE_SCHEMA_VERSION

    _ = persisted_user
    payload = {
        "intake": {
            "name": "Schema Version Create",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
    }
    response = await api_client.post("/api/campaigns", json=payload, headers=auth_headers)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["schema_version"] == API_RESPONSE_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_generate_suggestions_response_includes_schema_version(
    api_client, auth_headers, persisted_user
) -> None:
    """``POST /api/suggestions/{id}/generate`` envelope is versioned."""
    from api.schemas.common import API_RESPONSE_SCHEMA_VERSION

    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign_for_owner(owner_id, "Schema Version Generate")
    response = await api_client.post(
        f"/api/suggestions/{campaign_id}/generate",
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["schema_version"] == API_RESPONSE_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_submit_results_response_includes_schema_version(
    api_client, auth_headers, persisted_user
) -> None:
    """``POST /api/results/{id}`` envelope is versioned."""
    from api.schemas.common import API_RESPONSE_SCHEMA_VERSION

    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign_for_owner(owner_id, "Schema Version Submit")
    body = {
        "results": [
            {
                "parameter_values": {"x": 0.5},
                "objective_values": {"y": 0.7},
            }
        ],
        "source": "api",
    }
    response = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=auth_headers)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["schema_version"] == API_RESPONSE_SCHEMA_VERSION


def test_rest_envelope_tracks_mcp_envelope_version() -> None:
    """The REST and MCP envelopes always advertise the same integer.

    A divergent version on one transport would let a single bump on
    the other silently break clients that consume both. Pinning them
    in sync ensures the bump is a single edit on ``RESPONSE_SCHEMA_VERSION``.
    """
    from api.schemas.common import API_RESPONSE_SCHEMA_VERSION
    from bo_mcp_server.client import RESPONSE_SCHEMA_VERSION

    assert API_RESPONSE_SCHEMA_VERSION == RESPONSE_SCHEMA_VERSION


def _unwrap_container(model: type) -> tuple[type, ...]:
    """Return the inner Pydantic model types when ``model`` is a container.

    Routes can declare ``response_model=list[SuggestionResponse]`` or
    ``response_model=dict[str, ResultResponse]``; the schema-version
    contract applies to the element types, not the container shape.
    This helper unwraps a single level of ``list[...]`` / ``dict[..., X]``
    so the per-element rule can be applied uniformly.
    """
    import typing

    origin = typing.get_origin(model)
    if origin is None:
        return (model,)
    args = typing.get_args(model)
    # ``dict[K, V]`` carries the value type at index 1; everything
    # else (list, tuple, set, frozenset) keys on the first arg.
    if origin in (dict,):
        return tuple(arg for arg in args[1:] if isinstance(arg, type))
    return tuple(arg for arg in args if isinstance(arg, type))


def _collect_route_response_models() -> list[tuple[str, type]]:
    """Return ``(route_path, model_class)`` for every declared response model.

    Walks the live FastAPI app — the same set of routes a client sees —
    so the guard reflects what is actually wire-visible. Routes that
    do not declare a ``response_model`` (the diagnostics endpoint
    returns a raw dict, for example) are skipped: they have no
    statically-knowable shape and are out of the envelope contract by
    construction.
    """
    from pydantic import BaseModel

    from api.main import create_app

    app = create_app()
    pairs: list[tuple[str, type]] = []
    for route in app.routes:
        response_model = getattr(route, "response_model", None)
        if response_model is None:
            continue
        path = getattr(route, "path", "<unknown>")
        pairs.extend(
            (path, inner)
            for inner in _unwrap_container(response_model)
            if isinstance(inner, type) and issubclass(inner, BaseModel)
        )
    return pairs


def test_every_route_response_model_is_envelope_or_explicitly_exempt() -> None:
    """Every wire-visible response model carries ``schema_version`` or is allowlisted.

    Walks every ``response_model`` declared on the live FastAPI app
    (single types and ``list[...]`` / ``dict[..., X]`` containers
    alike). For each, asserts that the model either:

    * inherits from :class:`api.schemas.common.ResponseEnvelope`
      (carries ``schema_version`` by construction); or
    * is explicitly named in
      :data:`api.schemas.common.RESOURCE_VIEW_MODEL_NAMES` (the
      documented exemption for resource representations).

    A new ``FooResponse(BaseModel)`` wired into a route without
    inheriting :class:`ResponseEnvelope` and without an explicit
    exemption entry will fail this test with a pointer to the
    offending route — exactly the guard the friend's review asked
    for. The previous version of this test compared two hardcoded
    lists to each other and could not detect such a regression.
    """
    from api.schemas.common import RESOURCE_VIEW_MODEL_NAMES, ResponseEnvelope

    offenders: list[str] = []
    for path, model in _collect_route_response_models():
        if issubclass(model, ResponseEnvelope):
            continue
        if model.__name__ in RESOURCE_VIEW_MODEL_NAMES:
            continue
        offenders.append(
            f"  {path} -> {model.__name__}: "
            "neither inherits ResponseEnvelope nor is listed in "
            "RESOURCE_VIEW_MODEL_NAMES."
        )

    assert not offenders, (
        "REST response models must carry schema_version (via "
        "ResponseEnvelope) or be explicitly exempted in "
        "RESOURCE_VIEW_MODEL_NAMES:\n" + "\n".join(offenders)
    )


def test_resource_view_allowlist_matches_real_classes() -> None:
    """Every name in ``RESOURCE_VIEW_MODEL_NAMES`` resolves to an actual model.

    Stale allowlist entries (the named class moved or was renamed)
    would silently grant exemption to nothing real. Walking the
    schema modules surfaces the mismatch instead of letting it rot.
    """
    import inspect

    from pydantic import BaseModel

    from api.schemas import campaign as campaign_schemas
    from api.schemas import result as result_schemas
    from api.schemas import suggestion as suggestion_schemas
    from api.schemas.common import RESOURCE_VIEW_MODEL_NAMES

    all_models: dict[str, type] = {
        name: obj
        for module in (campaign_schemas, result_schemas, suggestion_schemas)
        for name, obj in inspect.getmembers(module)
        if inspect.isclass(obj) and issubclass(obj, BaseModel)
    }

    missing = sorted(name for name in RESOURCE_VIEW_MODEL_NAMES if name not in all_models)
    assert not missing, (
        "RESOURCE_VIEW_MODEL_NAMES contains entries that do not resolve "
        f"to a class in api.schemas.*: {missing}"
    )


def test_exempt_resource_models_do_not_inherit_envelope() -> None:
    """The allowlisted resource views must not accidentally carry ``schema_version``.

    An exempt model inheriting :class:`ResponseEnvelope` would
    contradict the documented contract: the exemption exists *because*
    these are not envelopes. The earlier static test asserted this
    against a hardcoded local dict; the dynamic check below pulls
    the exempt classes by name from the schema modules so it can't
    drift out of sync.
    """
    import inspect

    from pydantic import BaseModel

    from api.schemas import campaign as campaign_schemas
    from api.schemas import result as result_schemas
    from api.schemas import suggestion as suggestion_schemas
    from api.schemas.common import RESOURCE_VIEW_MODEL_NAMES, ResponseEnvelope

    name_to_model: dict[str, type] = {
        member_name: obj
        for module in (campaign_schemas, result_schemas, suggestion_schemas)
        for member_name, obj in inspect.getmembers(module)
        if inspect.isclass(obj) and issubclass(obj, BaseModel)
    }

    for name in RESOURCE_VIEW_MODEL_NAMES:
        model = name_to_model[name]
        # ``name_to_model`` was built by filtering ``inspect.getmembers``
        # for ``BaseModel`` subclasses; the explicit narrowing here
        # keeps ty happy with the ``model_fields`` access below
        # without weakening the runtime check.
        assert issubclass(model, BaseModel)
        assert not issubclass(model, ResponseEnvelope), (
            f"{name} inherits ResponseEnvelope despite being listed as a "
            "resource view; either remove the inheritance or drop the "
            "exemption."
        )
        assert "schema_version" not in model.model_fields, (
            f"{name} declares schema_version explicitly; the field is reserved for envelopes."
        )


@pytest.mark.asyncio
async def test_resource_get_responses_omit_schema_version(
    api_client, auth_headers, persisted_user
) -> None:
    """End-to-end pin: GET-by-id endpoints do not echo ``schema_version``.

    Confirms the documented exclusion on a live response body. If a
    future change accidentally wraps these in an envelope, the
    assertion fails with a clear diff.
    """
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign_for_owner(owner_id, "Resource Exempt")

    # Single-resource GET — resource representation, no envelope.
    response = await api_client.get(f"/api/v1/campaigns/{campaign_id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    assert "schema_version" not in response.json()
