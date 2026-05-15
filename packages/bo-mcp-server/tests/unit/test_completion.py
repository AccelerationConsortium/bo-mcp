"""Unit tests for the MCP ``completion/complete`` handler.

References:
    - MCP completion spec
      https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/completion
      requires the server to filter values by the partial input
      (``argument.value``) and to advertise total / hasMore so paginated
      clients render the list correctly.
    - The Pydantic ``Literal`` typing pattern produces a JSON Schema
      ``enum`` constraint
      (https://docs.pydantic.dev/latest/concepts/types/#literal-types),
      which is the discoverability mechanism that pairs with this
      completion surface.
"""

import pytest
from bo_engine.types import AcquisitionMethod
from mcp.types import (
    CompletionArgument,
    PromptReference,
    ResourceTemplateReference,
)

from bo_mcp_server.completion import handle_completion
from bo_mcp_server.domain import CampaignStatus, SuggestionStatus

_REF = ResourceTemplateReference(type="ref/resource", uri="campaign://{status}")


@pytest.mark.asyncio
async def test_status_returns_full_campaign_status_enum_when_blank() -> None:
    completion = await handle_completion(
        _REF, CompletionArgument(name="status", value=""), context=None
    )
    assert completion is not None
    assert set(completion.values) == {s.value for s in CampaignStatus}
    # Total is the full match count -- not the truncated slice length.
    assert completion.total == len(CampaignStatus)


@pytest.mark.asyncio
async def test_status_filters_by_case_insensitive_prefix() -> None:
    completion = await handle_completion(
        _REF, CompletionArgument(name="status", value="Run"), context=None
    )
    assert completion is not None
    assert completion.values == ["running"]


@pytest.mark.asyncio
async def test_suggestion_status_includes_completed_for_filter_surface() -> None:
    """The read-side ``suggestion_status`` surface includes every value.

    The mutation-side surface lives under the ``transition`` argument
    name and intentionally excludes ``completed`` so agents do not try
    to drive the state machine into a status that
    ``bo_submit_results`` owns.
    """
    completion = await handle_completion(
        _REF,
        CompletionArgument(name="suggestion_status", value=""),
        context=None,
    )
    assert completion is not None
    assert "completed" in completion.values
    assert set(completion.values) == {s.value for s in SuggestionStatus}


@pytest.mark.asyncio
async def test_transition_excludes_completed() -> None:
    completion = await handle_completion(
        _REF, CompletionArgument(name="transition", value=""), context=None
    )
    assert completion is not None
    assert "completed" not in completion.values
    assert set(completion.values) == {"accepted", "rejected", "expired"}


@pytest.mark.asyncio
async def test_acquisition_method_returns_engine_enum() -> None:
    completion = await handle_completion(
        _REF,
        CompletionArgument(name="acquisition_method", value=""),
        context=None,
    )
    assert completion is not None
    assert set(completion.values) == {m.value for m in AcquisitionMethod}


@pytest.mark.asyncio
async def test_backend_enum_values() -> None:
    completion = await handle_completion(
        _REF, CompletionArgument(name="backend", value=""), context=None
    )
    assert completion is not None
    assert set(completion.values) == {"auto", "botorch", "baybe"}


@pytest.mark.asyncio
async def test_action_enum_values() -> None:
    completion = await handle_completion(
        _REF, CompletionArgument(name="action", value="re"), context=None
    )
    assert completion is not None
    assert completion.values == ["resume"]


@pytest.mark.asyncio
async def test_verbosity_enum_values() -> None:
    completion = await handle_completion(
        _REF, CompletionArgument(name="verbosity", value=""), context=None
    )
    assert completion is not None
    assert completion.values == ["minimal", "standard", "detailed"]


@pytest.mark.asyncio
async def test_unknown_argument_returns_none() -> None:
    """Unknown arguments fall back to free-form input with empty completions."""
    completion = await handle_completion(
        _REF,
        CompletionArgument(name="unknown_field", value="anything"),
        context=None,
    )
    assert completion is None


@pytest.mark.asyncio
async def test_prompt_reference_works_too() -> None:
    """Completion works for prompt arguments as well as resource templates.

    The MCP spec dispatches both shapes through the same handler;
    matching on ``argument.name`` keeps the implementation transport-
    shape-agnostic.
    """
    prompt_ref = PromptReference(type="ref/prompt", name="some_prompt")
    completion = await handle_completion(
        prompt_ref,
        CompletionArgument(name="status", value=""),
        context=None,
    )
    assert completion is not None
    assert set(completion.values) == {s.value for s in CampaignStatus}


@pytest.mark.asyncio
async def test_completion_handler_is_registered_on_mcp_server() -> None:
    """``create_mcp_server`` advertises the completion capability by registering a handler."""
    from mcp import types

    from bo_mcp_server.server import create_mcp_server

    mcp = create_mcp_server()
    assert types.CompleteRequest in mcp._mcp_server.request_handlers


@pytest.mark.asyncio
async def test_completion_round_trip_through_registered_handler() -> None:
    """End-to-end: a ``CompleteRequest`` produces the enum values via the wired handler.

    Exercising the request_handler that ``register_completion_handler``
    installs (rather than calling :func:`handle_completion` directly)
    proves that the FastMCP plumbing -- ref/argument/context decoding,
    response wrapping into ``CompleteResult`` -- is intact.
    """
    from mcp import types

    from bo_mcp_server.server import create_mcp_server

    mcp = create_mcp_server()
    handler = mcp._mcp_server.request_handlers[types.CompleteRequest]
    request = types.CompleteRequest(
        method="completion/complete",
        params=types.CompleteRequestParams(
            ref=types.ResourceTemplateReference(type="ref/resource", uri="campaign://{status}"),
            argument=types.CompletionArgument(name="status", value=""),
            context=None,
        ),
    )
    server_result = await handler(request)
    # ``ServerResult`` wraps the ``CompleteResult``; the ``root``
    # attribute is the concrete payload.
    payload = server_result.root
    assert isinstance(payload, types.CompleteResult)
    assert set(payload.completion.values) == {s.value for s in CampaignStatus}


class TestToolSchemaEnumCoverage:
    """Primary discovery surface: tool JSON schemas declare ``enum`` constraints.

    Audited explicitly because completion/complete only fires on
    prompt/resource-template references in the MCP spec, not on tool
    arguments. The schema is what every MCP client uses to know the
    valid values for a tool parameter; if a parameter drifts back to
    untyped ``str``, this test catches it.
    """

    @staticmethod
    async def _tool_param_schema(tool_name: str, param: str) -> dict:
        from bo_mcp_server.server import create_mcp_server

        mcp = create_mcp_server()
        tools = await mcp.list_tools()
        by_name = {t.name: t for t in tools}
        return by_name[tool_name].inputSchema["properties"][param]

    @staticmethod
    def _enum_values(prop_schema: dict) -> set[str]:
        """Pull the ``enum`` list out of a (possibly nullable) property schema."""
        if "enum" in prop_schema:
            return set(prop_schema["enum"])
        # Nullable Literal -> {"anyOf": [{"enum": [...], "type": "string"}, {"type": "null"}]}
        for branch in prop_schema.get("anyOf", []):
            if "enum" in branch:
                return set(branch["enum"])
        return set()

    @pytest.mark.asyncio
    async def test_list_campaigns_status_is_enum(self) -> None:
        schema = await self._tool_param_schema("bo_list_campaigns", "status")
        assert self._enum_values(schema) == {s.value for s in CampaignStatus}

    @pytest.mark.asyncio
    async def test_list_suggestions_status_filter_is_enum(self) -> None:
        schema = await self._tool_param_schema("bo_list_suggestions", "status_filter")
        assert self._enum_values(schema) == {s.value for s in SuggestionStatus}

    @pytest.mark.asyncio
    async def test_update_suggestion_status_status_is_enum(self) -> None:
        schema = await self._tool_param_schema("bo_update_suggestion_status", "status")
        # ``completed`` is intentionally omitted (set by bo_submit_results).
        assert self._enum_values(schema) == {"accepted", "rejected", "expired"}

    @pytest.mark.asyncio
    async def test_verbosity_is_enum_on_a_representative_tool(self) -> None:
        schema = await self._tool_param_schema("bo_list_campaigns", "verbosity")
        assert self._enum_values(schema) == {"minimal", "standard", "detailed"}
