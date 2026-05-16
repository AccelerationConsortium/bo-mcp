"""Field-level error path formatting and accumulation helpers.

A flat ``errors: list[str]`` envelope tells an agent *what* went wrong but
not *which input field* tripped the validator. For batch submissions the
gap is acute -- a 20-row submit that fails because row 5 is missing an
objective forces the agent to either retry the whole batch or hand the
ambiguity back to the user. The ``field_errors`` map keyed by dotted
path (``results[5].objective_values.yield``) lets agents target the bad
field directly.

The path grammar mirrors how a human would address the offending field
in code:

* ``key`` for object attributes / dict keys (``parameters[0].name``).
* ``[i]`` for list / tuple positional access (``results[3]``).
* ``[k]`` for dict-key access when the key contains a dot or non-identifier
  character (``measurement_uncertainty['noisy.value']``).

The helpers here are intentionally transport-neutral: callers (operations
layer) build up the map and the response formatter forwards it verbatim
to the wire.
"""

from __future__ import annotations

import keyword
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import ValidationError

# Pydantic's loc tuple uses these synthetic keys for body-level errors
# (e.g. ``extra_forbidden`` reports ``loc=()``). Map them onto an empty
# dotted path so they collapse under a single ``""`` bucket.
_BODY_LEVEL_PATH = ""

_IDENTIFIER_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _is_safe_attr(segment: str) -> bool:
    """Return True iff ``segment`` reads cleanly as a dotted attribute."""
    if not segment or segment[0].isdigit() or keyword.iskeyword(segment):
        return False
    return all(ch in _IDENTIFIER_OK for ch in segment)


def _format_dict_key(segment: str) -> str:
    """Format a dict-key segment so paths round-trip through agent code.

    Plain identifier-shaped keys join with ``.`` (``a.b``); anything that
    would not parse as an attribute is bracketed with quoted-string
    syntax (``a['weird key']``) so the path stays unambiguous.
    """
    if _is_safe_attr(segment):
        return f".{segment}"
    escaped = segment.replace("\\", "\\\\").replace("'", "\\'")
    return f"['{escaped}']"


def format_loc_path(loc: Sequence[Any]) -> str:
    """Format a Pydantic-style ``loc`` tuple as a dotted/bracketed path.

    Examples:
        >>> format_loc_path(("parameters", 0, "name"))
        'parameters[0].name'
        >>> format_loc_path(("results", 5, "objective_values", "yield"))
        'results[5].objective_values.yield'
        >>> format_loc_path(())
        ''
    """
    if not loc:
        return _BODY_LEVEL_PATH
    parts: list[str] = []
    for index, segment in enumerate(loc):
        if isinstance(segment, int):
            parts.append(f"[{segment}]")
        elif index == 0:
            # The first textual segment becomes the root identifier; we
            # do not prefix it with a dot.
            parts.append(segment if _is_safe_attr(segment) else f"['{segment}']")
        else:
            parts.append(_format_dict_key(segment))
    return "".join(parts)


def validation_errors_to_field_errors(
    error: ValidationError,
) -> dict[str, list[str]]:
    """Convert a Pydantic ``ValidationError`` into a field-keyed error map.

    Multiple errors against the same path are preserved in input order.
    Body-level errors (``loc=()``) collapse onto the ``""`` key so the
    map shape stays uniform.
    """
    field_errors: dict[str, list[str]] = {}
    for entry in error.errors():
        path = format_loc_path(entry.get("loc", ()))
        message = str(entry.get("msg", ""))
        field_errors.setdefault(path, []).append(message)
    return field_errors


def _strip_index_prefix(path: str, index: int) -> str:
    prefix = f"[{index}]"
    if path == prefix:
        return ""
    if path.startswith(prefix):
        # Strip the leading dot so ``[0].x`` → ``x``.
        rest = path[len(prefix) :]
        if rest.startswith("."):
            return rest[1:]
        return rest
    return path


def add_row_field_error(
    field_errors: dict[str, list[str]],
    row_index: int,
    field_path: str,
    message: str,
) -> None:
    """Record an error against ``results[row_index].<field_path>``.

    ``field_path`` may itself be empty (the row as a whole) or a nested
    path like ``measurement_uncertainty['yield']``. The composed key uses
    the canonical bracket-then-dot grammar.
    """
    base = f"results[{row_index}]"
    if not field_path:
        composed = base
    elif field_path.startswith("["):
        composed = f"{base}{field_path}"
    else:
        composed = f"{base}.{field_path}"
    field_errors.setdefault(composed, []).append(message)


def merge_pydantic_row_errors(
    field_errors: dict[str, list[str]],
    row_index: int,
    error: ValidationError,
) -> None:
    """Splice a per-row Pydantic ``ValidationError`` under ``results[i]``.

    Each entry's ``loc`` is rewritten to live under the ``results[i]``
    namespace so the resulting paths are addressable from the
    submission-level envelope.
    """
    for entry in error.errors():
        sub_path = format_loc_path(entry.get("loc", ()))
        # Drop a leading ``[i]`` if Pydantic emitted it relative to the
        # outer list (we already prefix ``results[i]`` ourselves).
        sub_path = _strip_index_prefix(sub_path, row_index)
        add_row_field_error(field_errors, row_index, sub_path, str(entry.get("msg", "")))


def field_error_messages(field_errors: dict[str, list[str]]) -> Iterable[str]:
    """Flatten a field-error map into a list of ``path: message`` strings.

    Useful when the legacy ``errors: list[str]`` field still needs to
    carry the same information for older clients.
    """
    for path, messages in field_errors.items():
        for message in messages:
            if path:
                yield f"{path}: {message}"
            else:
                yield message


def validation_envelope(
    error: ValidationError,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a tool-boundary :class:`ValidationError` as a structured envelope.

    Used by MCP tool wrappers that accept loose payloads at the
    boundary and validate them explicitly so the failure path produces
    the same ``{success, error, errors, field_errors, ...}`` shape that
    operation-layer validation does. Without this wrapper, FastMCP's
    pre-call argument validator catches the exception and re-raises it
    as an opaque ``ToolError`` -- agents lose both the structured
    error envelope and the dotted-path ``field_errors`` map.

    ``extra`` is merged into the response dict so callers can pin
    operation-specific defaults (``result_ids: []``,
    ``campaign_id: None``, etc.) without re-implementing the envelope.
    """
    return _envelope_from_field_errors(validation_errors_to_field_errors(error), extra=extra)


def shape_envelope(
    field_path: str,
    message: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a synthesized outer-shape failure as a structured envelope.

    For shape errors the tool wrapper detects before any per-field
    Pydantic pass would run -- e.g. ``intake_data`` is not an object,
    ``results`` is not a list, ``results[i]`` is not a dict. Going
    through the same envelope keeps the failure contract identical to
    inner-field failures: agents address the bad input via
    ``field_errors[field_path]`` regardless of whether the breakage
    sits at the container level or inside a documented sub-field.
    """
    return _envelope_from_field_errors({field_path: [message]}, extra=extra)


def _envelope_from_field_errors(
    field_errors: dict[str, list[str]],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared rendering path for the two boundary-failure entry points."""
    from bo_mcp_server.errors import ErrorCode, make_error_response  # noqa: PLC0415

    flat_errors = list(field_error_messages(field_errors))
    response = make_error_response(
        ErrorCode.VALIDATION_FAILED,
        message="Tool payload failed validation",
        details={
            "validation_errors": flat_errors,
            "field_errors": field_errors,
        },
    )
    response["errors"] = flat_errors
    response["field_errors"] = field_errors
    if extra:
        for key, value in extra.items():
            response.setdefault(key, value)
    return response
