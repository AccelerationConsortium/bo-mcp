"""MCP ``completion/complete`` handler — enum-aware autocompletion.

Discovery surface hierarchy
---------------------------
The MCP spec exposes valid enum values to clients in two places:

1. **Tool JSON Schema** (primary). Tool parameters typed as
   ``Literal[...]`` / ``Enum`` produce ``"enum": [...]`` constraints in
   the per-tool ``inputSchema`` returned by ``tools/list``. Every
   compliant client reads these without an extra round-trip, so this is
   where we put authoritative coverage for ``status``,
   ``suggestion_status``, ``acquisition_method``, ``backend``, ``action``,
   and ``verbosity``. The audit in
   ``tests/unit/test_completion.py::TestToolSchemaEnumCoverage`` pins
   that contract.
2. **completion/complete** (secondary). The spec only wires completion
   into ``PromptReference`` / ``ResourceTemplateReference`` arguments,
   not tool arguments. This module covers the cases where future
   prompts / resource templates expose a variable whose ``argument.name``
   matches one of our enum surfaces -- the handler resolves the values
   without each new surface having to repeat the enum table.

The handler matches purely on ``argument.name`` so any
``PromptReference`` or ``ResourceTemplateReference`` that names a
variable ``status`` / ``acquisition_method`` / etc. picks up the right
list for free. Returning ``None`` for unknown arguments tells the
lowlevel server to emit an empty completion, which clients render as
"no suggestions" without surfacing an error.

* ``status`` (campaign status) — ``CampaignStatus`` enum.
* ``suggestion_status`` — ``SuggestionStatus`` plus ``"completed"`` and
  ``"expired"`` for read-only filters; for *transitions* the manual-only
  set is exposed instead.
* ``acquisition_method`` — ``AcquisitionMethod`` enum.
* ``backend`` — backend names accepted by ``CampaignIntakeInput.backend``
  (mirrors the Pydantic ``Literal`` constraint).
* ``action`` — campaign-lifecycle actions accepted by
  :func:`manage_campaign_lifecycle_operation`.
* ``verbosity`` — the three documented response-verbosity levels.

Values are filtered against the case-insensitive prefix in
``argument.value`` so the completion list shrinks as the user types --
aligned with the MCP completion spec
(https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/completion)
which says the server SHOULD filter values by the partial input. An
empty partial value returns the full list so clients without typed
input render it as a static dropdown.
"""

from __future__ import annotations

from collections.abc import Iterable

from mcp.server.fastmcp import FastMCP
from mcp.types import (
    Completion,
    CompletionArgument,
    CompletionContext,
    PromptReference,
    ResourceTemplateReference,
)

from bo_engine.types import AcquisitionMethod
from bo_mcp_server.domain import CampaignStatus, SuggestionStatus

# Hard cap mirroring the MCP spec: ``Completion.values`` MUST NOT exceed
# 100 items. Our enums are tiny so this is a defensive ceiling rather
# than a slicing target.
_MAX_COMPLETION_VALUES = 100

# Manual-only suggestion transitions accepted by
# ``bo_update_suggestion_status``. ``completed`` is excluded because it
# is set automatically by ``bo_submit_results`` -- exposing it as a
# completion would lure agents into invalid-state-transition errors.
_MANUAL_SUGGESTION_TRANSITIONS = ("accepted", "rejected", "expired")

# Lifecycle actions accepted by ``manage_campaign_lifecycle_operation``.
_LIFECYCLE_ACTIONS = ("pause", "resume", "terminate", "reopen")

# Verbosity levels. Keep in sync with
# ``bo_mcp_server.response_formatter.VerbosityLevel``.
_VERBOSITY_LEVELS = ("minimal", "standard", "detailed")

# Backend selectors honoured by ``CampaignIntakeInput.backend`` (see the
# ``pattern`` constraint there).
_BACKEND_NAMES = ("auto", "baybe", "botorch")


# Lookup table consumed by :func:`_enum_values_for`. Keyed by
# ``argument.name``; values are tuples so dispatch is a single dict
# lookup instead of a chain of ``if argument_name ==`` arms (which
# tripped PLR0911 once the enum surface grew past six entries).
#
# * ``status`` -- read filter for ``CampaignStatus``.
# * ``suggestion_status`` -- read filter for ``SuggestionStatus``
#   (includes ``completed``, which the mutation-side ``transition``
#   surface intentionally drops).
# * ``transition`` -- manual suggestion transitions accepted by
#   ``bo_update_suggestion_status``.
# * ``acquisition_method`` -- BO-engine acquisition selectors.
# * ``backend`` -- backend selectors honoured by
#   ``CampaignIntakeInput.backend``.
# * ``action`` -- campaign-lifecycle actions.
# * ``verbosity`` -- response verbosity levels.
_ENUM_VALUES: dict[str, tuple[str, ...]] = {
    "status": tuple(s.value for s in CampaignStatus),
    "suggestion_status": tuple(s.value for s in SuggestionStatus),
    "transition": _MANUAL_SUGGESTION_TRANSITIONS,
    "acquisition_method": tuple(m.value for m in AcquisitionMethod),
    "backend": _BACKEND_NAMES,
    "action": _LIFECYCLE_ACTIONS,
    "verbosity": _VERBOSITY_LEVELS,
}


def _enum_values_for(argument_name: str) -> tuple[str, ...] | None:
    """Map an MCP completion argument name to its enum value list.

    Returns ``None`` when the argument is not one of the documented
    enum surfaces; callers translate that into a no-op completion so
    clients keep accepting free-form input.
    """
    return _ENUM_VALUES.get(argument_name)


def _filter_by_prefix(values: Iterable[str], partial: str) -> list[str]:
    """Return enum values whose string starts with ``partial`` (case-insensitive).

    An empty ``partial`` returns the full list so clients without
    typed-input context (initial dropdown render) get the whole enum.
    Sort order is preserved from the source tuple so the value with
    canonical "first" semantics (``auto``, ``minimal``, …) stays at the
    top of the list.
    """
    if not partial:
        return list(values)
    needle = partial.lower()
    return [v for v in values if v.lower().startswith(needle)]


async def handle_completion(
    ref: PromptReference | ResourceTemplateReference,
    argument: CompletionArgument,
    context: CompletionContext | None,
) -> Completion | None:
    """Resolve completion values for known enum-typed arguments.

    ``ref`` is recorded but not used for matching today: the argument
    name alone identifies the enum surface, and the same names mean the
    same things across every prompt or resource template the server may
    add in the future. Returning ``None`` for unknown arguments
    instructs the lowlevel server to emit an empty completion list,
    which clients render as "no suggestions" without surfacing an
    error.
    """
    del ref, context  # Reserved for future per-template scoping.
    enum_values = _enum_values_for(argument.name)
    if enum_values is None:
        return None
    matches = _filter_by_prefix(enum_values, argument.value)
    return Completion(
        values=matches[:_MAX_COMPLETION_VALUES],
        total=len(matches),
        hasMore=len(matches) > _MAX_COMPLETION_VALUES,
    )


def register_completion_handler(mcp_instance: FastMCP) -> None:
    """Wire :func:`handle_completion` into the FastMCP instance.

    Implemented as a function so :mod:`bo_mcp_server.server` can call it
    from inside ``create_mcp_server()`` -- avoiding the import-time side
    effect of a module-level ``@mcp.completion()`` decorator (which
    would couple completion registration to module import order and
    make the dependency graph harder to test in isolation).
    """
    decorator = mcp_instance.completion()
    decorator(handle_completion)
