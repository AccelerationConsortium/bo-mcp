"""OpenAPI inspection tools for the BO-MCP service."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Annotated, Literal
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import Field, validate_call
from pydantic_ai import ModelRetry

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}
OpenAPIVerbosity = Literal["default", "extended", "full"]

# Fixed output budget for both inspectors. Real campaign traces peak around
# 10k chars per call, so 25k leaves headroom while still bounding what a
# pathological spec can push into the agent context. Agents do not choose
# this value; truncated output tells them to narrow the request instead.
MAX_OUTPUT_CHARS = 25000


def _truncate_text(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text

    truncated_chars = len(text) - MAX_OUTPUT_CHARS
    return text[:MAX_OUTPUT_CHARS] + (f"\n\n[output truncated; {truncated_chars} chars omitted]")


def _default_openapi_url() -> str:
    explicit_url = os.getenv("BO_MCP_OPENAPI_URL")
    if explicit_url:
        return explicit_url

    api_url = (os.getenv("BO_MCP_API_URL") or os.getenv("BO_REST_URL") or "http://api:8000").rstrip(
        "/"
    )
    return f"{api_url}/openapi.json"


def _fetch_openapi(url: str) -> dict:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        message = "BO-MCP OpenAPI URL must be an absolute HTTP(S) URL."
        raise ValueError(message)

    req = Request(url, headers={"Accept": "application/json"})  # noqa: S310 - Scheme is validated above.
    with urlopen(req, timeout=30) as resp:  # noqa: S310 - Scheme is validated above.
        return json.loads(resp.read().decode("utf-8"))


def _schema_name_from_ref(ref: str) -> str:
    return ref.split("/")[-1] if isinstance(ref, str) else "UnknownRef"


def _type_repr(schema: dict, spec: dict) -> str:  # noqa: C901, PLR0911 - Mirrors OpenAPI schema variants.
    if not schema:
        return "Any"

    if "$ref" in schema:
        return _schema_name_from_ref(schema["$ref"])

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        parts = schema["anyOf"]
        non_null = [s for s in parts if s.get("type") != "null"]
        has_null = len(non_null) != len(parts)

        if len(non_null) == 1:
            inner = _type_repr(non_null[0], spec)
            return f"Optional[{inner}]" if has_null else inner

        union = " | ".join(_type_repr(part, spec) for part in non_null) or "Any"
        return f"Optional[{union}]" if has_null else union

    schema_type = schema.get("type")
    if schema_type == "string":
        if "enum" in schema and isinstance(schema["enum"], list):
            return "str  # enum=" + repr(schema["enum"])
        return "str"
    if schema_type == "integer":
        return "int"
    if schema_type == "number":
        return "float"
    if schema_type == "boolean":
        return "bool"
    if schema_type == "array":
        items = schema.get("items", {}) if isinstance(schema.get("items"), dict) else {}
        return f"list[{_type_repr(items, spec)}]"
    if schema_type == "object":
        additional_props = schema.get("additionalProperties")
        if isinstance(additional_props, dict):
            return f"dict[str, {_type_repr(additional_props, spec)}]"
        return "dict[str, Any]"

    return "Any"


def _describe_schema_brief(schema: dict, spec: dict) -> str:
    if not schema:
        return "-"
    if "$ref" in schema:
        return f"$ref({_schema_name_from_ref(schema['$ref'])})"
    if "anyOf" in schema:
        return _type_repr(schema, spec)

    schema_type = schema.get("type")
    schema_format = schema.get("format")
    return f"{schema_type}" + (f"({schema_format})" if schema_format else "")


def _iter_operations(spec: dict, substring_filter: str = "") -> Iterator[tuple[str, str, dict]]:
    filt = (substring_filter or "").lower().strip()
    for path, path_item in (spec.get("paths") or {}).items():
        if filt and filt not in path.lower():
            continue
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() in HTTP_METHODS and isinstance(operation, dict):
                yield path, method.upper(), operation


def _find_path_item_and_operation(
    spec: dict,
    path: str,
    method: str,
) -> tuple[dict, dict] | None:
    path_item = (spec.get("paths") or {}).get(path)
    if not isinstance(path_item, dict):
        return None

    operation = path_item.get(method.lower())
    return (path_item, operation) if isinstance(operation, dict) else None


def _find_operation(spec: dict, path: str, method: str) -> dict | None:
    found = _find_path_item_and_operation(spec, path, method)
    return found[1] if found else None


def _merge_parameters(
    path_parameters: list[dict],
    operation_parameters: list[dict],
) -> list[dict]:
    merged: list[dict] = []
    index_by_key: dict[tuple[str, str], int] = {}

    for param in path_parameters:
        if not isinstance(param, dict):
            continue
        key = (str(param.get("name", "")), str(param.get("in", "")))
        index_by_key[key] = len(merged)
        merged.append(param)

    for param in operation_parameters:
        if not isinstance(param, dict):
            continue
        key = (str(param.get("name", "")), str(param.get("in", "")))
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(merged)
            merged.append(param)
        else:
            merged[existing_index] = param

    return merged


def _format_client_contract(spec: dict) -> str:
    security_schemes = (spec.get("components") or {}).get("securitySchemes") or {}
    api_key_scheme = security_schemes.get("ApiKeyAuth") or {}
    api_key_header = api_key_scheme.get("name") or "X-API-Key"

    return "\n".join(
        [
            "=== BO-MCP REST CLIENT CONTRACT ===",
            "- Base URL: use BO_MCP_API_URL. OpenAPI is usually at "
            "`${BO_MCP_API_URL}/openapi.json`.",
            "- Use the versioned paths advertised by OpenAPI, usually `/api/v1/...`.",
            f"- Send `{api_key_header}` on authenticated endpoints.",
            "- For mutation endpoints that expose `Idempotency-Key`, generate one "
            "stable key per logical create/submit attempt and reuse that same key "
            "only for retries of the exact same payload.",
            "- Do not reuse an `Idempotency-Key` for a different payload; BO-MCP "
            "can return a conflict/in-progress response.",
            "- REST and MCP share the idempotency cache namespace, so a retry via "
            "the other transport can replay the same prior operation when the "
            "canonical payload matches.",
            '- Deliberate HTTP errors usually return `{"detail": ...}`.',
            "- Sanitized internal errors return a structured `success=false` "
            "error envelope with request-correlation details.",
            "- Some operation-level failures return HTTP 200 with `success=false`; "
            "client code must check the `success` field, not only `status_code`.",
            "- Treat `2xx` plus `success=false` as: request processed, operation rejected.",
        ]
    )


def _format_parameters(parameters: list[dict], spec: dict) -> list[str]:
    lines = ["  parameters:"]
    for param in parameters:
        schema = param.get("schema") or {}
        description = (param.get("description") or "").strip()
        lines.append(
            "    - "
            f"{param.get('name', '?')} in={param.get('in', '?')} "
            f"required={bool(param.get('required', False))} "
            f":: {_describe_schema_brief(schema, spec)}"
        )
        if description:
            lines.append(f"      desc: {description}")
    return lines


def _select_parameters(
    parameters: list[dict],
    verbosity: OpenAPIVerbosity,
) -> list[dict]:
    if verbosity != "default":
        return parameters

    selected: list[dict] = []
    for param in parameters:
        param_name = str(param.get("name", "")).lower()
        param_in = str(param.get("in", "")).lower()
        is_required = bool(param.get("required", False))
        if param_in == "header" and param_name == "x-api-key" and not is_required:
            continue
        selected.append(param)
    return selected


def _format_request_body(request_body: dict, spec: dict) -> list[str]:
    lines = ["  requestBody:"]
    content = request_body.get("content") or {}
    if not content:
        lines.append("    - (no content)")
        return lines

    for content_type, content_obj in content.items():
        schema = (content_obj or {}).get("schema") or {}
        lines.append(f"    - {content_type}: {_describe_schema_brief(schema, spec)}")
    return lines


def _select_responses(
    responses: dict,
    verbosity: OpenAPIVerbosity,
) -> list[tuple[str, dict]]:
    items = list((responses or {}).items())
    if verbosity != "default":
        return items

    success_items = [(code, response) for code, response in items if str(code).startswith("2")]
    return success_items or items[:1]


def _format_responses(
    responses: list[tuple[str, dict]],
    spec: dict,
    verbosity: OpenAPIVerbosity = "default",
) -> list[str]:
    lines = ["  responses:"]
    for code, response_obj in responses:
        response_obj = response_obj or {}
        description = (response_obj.get("description") or "").strip()
        content = response_obj.get("content") or {}
        if not content:
            lines.append(f"    - {code}: {description}")
            continue
        if verbosity in {"extended", "full"} and description:
            lines.append(f"    - {code}: {description}")
        for content_type, content_obj in content.items():
            schema = (content_obj or {}).get("schema") or {}
            lines.append(f"    - {code} {content_type}: {_describe_schema_brief(schema, spec)}")
            example = (content_obj or {}).get("example")
            if verbosity in {"extended", "full"} and example is not None:
                example_text = json.dumps(example, ensure_ascii=False)
                lines.append(f"      example: {example_text[:500]}")
    return lines


def _format_security(operation: dict, spec: dict) -> list[str]:
    operation_security = operation.get("security")
    if operation_security is None:
        operation_security = spec.get("security")
    if not operation_security:
        return []

    return ["  security: " + json.dumps(operation_security, ensure_ascii=False)]


def _format_endpoints(
    spec: dict,
    substring_filter: str = "",
    verbosity: OpenAPIVerbosity = "full",
) -> str:
    lines = ["=== PATHS / OPERATIONS ==="]

    for path, method, op in _iter_operations(spec, substring_filter):
        lines.append("")
        lines.append(f"{method:6s} {path}")
        if verbosity in {"extended", "full"}:
            lines.append(f"  operationId: {op.get('operationId', '-')}")
        if verbosity in {"extended", "full"} and op.get("tags"):
            lines.append(f"  tags: {', '.join(op['tags'])}")
        if verbosity in {"extended", "full"}:
            lines.extend(_format_security(op, spec))

        summary = op.get("summary") or op.get("description") or ""
        if summary:
            lines.append(f"  summary: {summary.strip().splitlines()[0][:200]}")

        params = _select_parameters(op.get("parameters") or [], verbosity)
        if params:
            lines.extend(_format_parameters(params, spec))

        request_body = op.get("requestBody")
        if request_body:
            lines.extend(_format_request_body(request_body, spec))

        responses = _select_responses(op.get("responses") or {}, verbosity)
        if responses:
            lines.extend(_format_responses(responses, spec, verbosity))

    return "\n".join(lines)


def _format_operation_details(
    spec: dict,
    path: str,
    method: str,
    operation: dict,
    parameters: list[dict] | None = None,
) -> str:
    lines = ["=== OPERATION ===", f"{method.upper():6s} {path}"]

    for key in ("operationId", "summary", "description"):
        value = operation.get(key)
        if value:
            lines.append(f"{key}: {str(value).strip()}")

    if operation.get("tags"):
        lines.append(f"tags: {', '.join(operation['tags'])}")

    lines.extend(_format_security(operation, spec))

    params = parameters if parameters is not None else operation.get("parameters") or []
    if params:
        lines.extend(_format_parameters(params, spec))

    request_body = operation.get("requestBody")
    if request_body:
        lines.extend(_format_request_body(request_body, spec))

    responses = list((operation.get("responses") or {}).items())
    if responses:
        lines.extend(_format_responses(responses, spec, "extended"))

    return "\n".join(lines)


def _format_schema_dataclass_like(name: str, schema: dict, spec: dict) -> list[str]:
    title = schema.get("title", name)
    description = schema.get("description")
    required = schema.get("required") or []
    properties = schema.get("properties") or {}

    lines = [f"class {title}:"]
    if description:
        escaped_description = description.replace('"""', r"\"\"\"")
        lines.append(f'    """{escaped_description}"""')

    if not isinstance(properties, dict) or not properties:
        schema_type = schema.get("type")
        suffix = f"; type={schema_type}" if schema_type else ""
        lines.append(f"    # schema has no explicit properties{suffix}")
        return lines

    def sort_key(item: tuple[str, dict]) -> tuple[int, str]:
        field, _ = item
        return (0 if field in required else 1, field)

    for field, field_schema in sorted(properties.items(), key=sort_key):
        is_required = field in required
        field_type = _type_repr(field_schema, spec)
        if not is_required and not field_type.startswith("Optional["):
            field_type = f"Optional[{field_type}]"

        default_value = "" if is_required else " = None"
        comment_bits: list[str] = []
        if field_schema.get("description"):
            comment_bits.append(f"desc={field_schema['description']!r}")
        if "default" in field_schema:
            comment_bits.append(f"default={field_schema['default']!r}")
        if "minimum" in field_schema:
            comment_bits.append(f"min={field_schema['minimum']!r}")
        if "maximum" in field_schema:
            comment_bits.append(f"max={field_schema['maximum']!r}")

        comment = ("  # " + ", ".join(comment_bits)) if comment_bits else ""
        lines.append(f"    {field}: {field_type}{default_value}{comment}")

    return lines


def _collect_schema_refs_from_schema(  # noqa: C901, PLR0912 - Visits the OpenAPI schema graph variants.
    schema: dict,
    spec: dict,
    seen: set[str],
) -> None:
    if not isinstance(schema, dict) or not schema:
        return

    ref = schema.get("$ref")
    if isinstance(ref, str):
        schema_name = _schema_name_from_ref(ref)
        if schema_name in seen:
            return
        seen.add(schema_name)
        target = ((spec.get("components") or {}).get("schemas") or {}).get(schema_name)
        if isinstance(target, dict):
            _collect_schema_refs_from_schema(target, spec, seen)
        return

    for key in ("items", "additionalProperties", "not", "if", "then", "else"):
        nested = schema.get(key)
        if isinstance(nested, dict):
            _collect_schema_refs_from_schema(nested, spec, seen)

    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        nested_items = schema.get(key)
        if isinstance(nested_items, list):
            for item in nested_items:
                if isinstance(item, dict):
                    _collect_schema_refs_from_schema(item, spec, seen)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for property_schema in properties.values():
            if isinstance(property_schema, dict):
                _collect_schema_refs_from_schema(property_schema, spec, seen)


def _collect_referenced_schema_names(
    spec: dict,
    substring_filter: str = "",
) -> set[str]:
    seen: set[str] = set()
    for _, _, operation in _iter_operations(spec, substring_filter):
        for param in operation.get("parameters") or []:
            _collect_schema_refs_from_schema(param.get("schema") or {}, spec, seen)

        request_body = operation.get("requestBody") or {}
        for content_obj in (request_body.get("content") or {}).values():
            _collect_schema_refs_from_schema(
                (content_obj or {}).get("schema") or {},
                spec,
                seen,
            )

        for response_obj in (operation.get("responses") or {}).values():
            response_obj = response_obj or {}
            for content_obj in (response_obj.get("content") or {}).values():
                _collect_schema_refs_from_schema(
                    (content_obj or {}).get("schema") or {},
                    spec,
                    seen,
                )

    return seen


def _collect_operation_schema_names(
    spec: dict,
    operation: dict,
    parameters: list[dict] | None = None,
) -> set[str]:
    seen: set[str] = set()

    params = parameters if parameters is not None else operation.get("parameters") or []
    for param in params:
        _collect_schema_refs_from_schema(param.get("schema") or {}, spec, seen)

    request_body = operation.get("requestBody") or {}
    for content_obj in (request_body.get("content") or {}).values():
        _collect_schema_refs_from_schema(
            (content_obj or {}).get("schema") or {},
            spec,
            seen,
        )

    for response_obj in (operation.get("responses") or {}).values():
        response_obj = response_obj or {}
        for content_obj in (response_obj.get("content") or {}).values():
            _collect_schema_refs_from_schema(
                (content_obj or {}).get("schema") or {},
                spec,
                seen,
            )

    return seen


def _format_components(
    spec: dict,
    *,
    schema_names: set[str] | None = None,
    include_non_schema_components: bool = True,
) -> str:
    components = spec.get("components") or {}
    if not components:
        return "\n=== COMPONENTS ===\n(no components section present)"

    schema_heading = "=== COMPONENTS (ALL) ==="
    schemas = components.get("schemas") or {}
    selected_schema_names = sorted(schemas)
    if schema_names is not None:
        schema_heading = "=== COMPONENTS (REFERENCED) ==="
        selected_schema_names = sorted(name for name in schema_names if name in schemas)

    lines = ["", schema_heading]
    schemas = components.get("schemas") or {}
    lines.append("")
    lines.append(f"-- components.schemas ({len(selected_schema_names)}) --")

    for name in selected_schema_names:
        lines.append("")
        lines.extend(_format_schema_dataclass_like(name, schemas[name], spec))

    if not include_non_schema_components:
        return "\n".join(lines)

    for key in sorted(k for k in components if k != "schemas"):
        value = components.get(key)
        item_count = len(value) if isinstance(value, dict) else 0
        lines.append("")
        lines.append(f"-- components.{key} ({item_count}) --")
        if isinstance(value, dict) and value:
            for item_name, item_obj in sorted(value.items()):
                lines.append("")
                lines.append(f"[{item_name}]")
                lines.append(json.dumps(item_obj, indent=2, ensure_ascii=False))
        else:
            lines.append("(empty or non-dict)")

    return "\n".join(lines)


@validate_call
def inspect_bo_mcp_openapi_overview(
    path_filter: Annotated[
        str,
        Field(
            description=(
                "Case-insensitive substring filter applied to endpoint paths. "
                "Leave empty to inspect the full BO OpenAPI spec."
            )
        ),
    ] = "",
    openapi_url: Annotated[
        str | None,
        Field(
            description=(
                "Optional OpenAPI JSON URL. Defaults to `BO_MCP_OPENAPI_URL`, or "
                "`BO_MCP_API_URL` with `/openapi.json` appended."
            )
        ),
    ] = None,
    verbosity: Annotated[
        OpenAPIVerbosity,
        Field(
            description=(
                "`default` returns a compact endpoint inventory, `extended` adds "
                "referenced schemas, and `full` includes the complete components "
                "section. Use `default` for quick overviews first."
            )
        ),
    ] = "default",
) -> str:
    """Inspect the BO-MCP OpenAPI schema and return a readable summary.

    Verbosity levels:
    - default: endpoint inventory with request/response refs; default
    - extended: detailed endpoints plus referenced schemas
    - full: current exhaustive dump including all components
    """
    target_url = (openapi_url or _default_openapi_url()).strip()

    try:
        spec = _fetch_openapi(target_url)
    except Exception as exc:  # pragma: no cover - exact urllib failure types vary
        message = f"Failed to fetch BO-MCP OpenAPI schema from {target_url}: {exc}"
        raise ModelRetry(message) from exc

    info = spec.get("info") or {}
    sections = [
        f"Source:  {target_url}",
        f"Title:   {info.get('title')}",
        f"Version: {info.get('version')}",
        f"OpenAPI: {spec.get('openapi')}",
        "",
        _format_client_contract(spec),
        "",
        _format_endpoints(spec, path_filter, verbosity),
    ]
    if verbosity == "extended":
        sections.append(
            _format_components(
                spec,
                schema_names=_collect_referenced_schema_names(spec, path_filter),
                include_non_schema_components=False,
            )
        )
    elif verbosity == "full":
        sections.append(_format_components(spec))

    return _truncate_text("\n".join(sections).strip())


@validate_call
def inspect_bo_mcp_openapi_operation(
    path: Annotated[
        str,
        Field(
            description=(
                "Exact OpenAPI path to inspect, e.g. `/api/v1/campaigns` or "
                "`/api/v1/results/{campaign_id}`."
            )
        ),
    ],
    method: Annotated[
        str,
        Field(
            description=("HTTP method for the operation, e.g. `get` or `post`. Case-insensitive.")
        ),
    ],
    openapi_url: Annotated[
        str | None,
        Field(
            description=(
                "Optional OpenAPI JSON URL. Defaults to `BO_MCP_OPENAPI_URL`, or "
                "`BO_MCP_API_URL` with `/openapi.json` appended."
            )
        ),
    ] = None,
) -> str:
    """Inspect one BO-MCP OpenAPI operation for agent-authored client code."""
    target_url = (openapi_url or _default_openapi_url()).strip()
    normalized_method = method.strip().lower()
    if normalized_method not in HTTP_METHODS:
        message = (
            f"Unsupported HTTP method {method!r}. Expected one of: "
            f"{', '.join(sorted(HTTP_METHODS))}."
        )
        raise ModelRetry(message)

    try:
        spec = _fetch_openapi(target_url)
    except Exception as exc:  # pragma: no cover - exact urllib failure types vary
        message = f"Failed to fetch BO-MCP OpenAPI schema from {target_url}: {exc}"
        raise ModelRetry(message) from exc

    found = _find_path_item_and_operation(spec, path, normalized_method)
    if found is None:
        matches = [
            f"{op_method} {op_path}" for op_path, op_method, _ in _iter_operations(spec, path)
        ]
        hint = "\nClosest path-filter matches:\n" + "\n".join(matches[:20]) if matches else ""
        message = f"Operation {normalized_method.upper()} {path} not found in {target_url}.{hint}"
        raise ModelRetry(message)

    path_item, operation = found
    parameters = _merge_parameters(
        path_item.get("parameters") or [],
        operation.get("parameters") or [],
    )
    sections = [
        f"Source:  {target_url}",
        _format_operation_details(spec, path, normalized_method, operation, parameters),
        _format_components(
            spec,
            schema_names=_collect_operation_schema_names(spec, operation, parameters),
            include_non_schema_components=False,
        ),
    ]

    return _truncate_text("\n".join(sections).strip())
