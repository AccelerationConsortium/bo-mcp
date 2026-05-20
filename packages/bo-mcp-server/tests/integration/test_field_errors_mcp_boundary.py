"""MCP-boundary coverage for ``field_errors`` on mutating tools.

The direct-function tests in ``test_field_errors_e2e.py`` exercise the
operation layer, which is the path REST callers use. MCP clients hit
``ToolManager.call_tool``, which historically caught Pydantic
``ValidationError`` raised against the typed argument schema and re-
raised it as an opaque ``ToolError`` -- agents lost both the
structured envelope and the dotted-path ``field_errors`` map.

These tests dispatch payload-shape failures through
``mcp._tool_manager.call_tool`` exactly the way the FastMCP transport
does, so the assertions only pass when the tool boundary itself
returns the structured envelope.

Reference: MCP server-side validation
(https://modelcontextprotocol.io/specification/2025-06-18/server/tools#tool-arguments)
makes ``tools/call`` return either a successful result or an error
content block; the FastMCP SDK wraps validation failures as
``ToolError`` before the tool function runs. Accepting raw dicts at
the tool boundary and validating inside the operation layer keeps the
failure path on the same structured-envelope contract REST already
uses.
"""

from typing import Any
from uuid import uuid4

import pytest


async def _server() -> Any:
    from bo_mcp_server.server import create_mcp_server

    return create_mcp_server()


async def _call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call exactly the way the FastMCP transport does."""
    mcp = await _server()
    return await mcp._tool_manager.call_tool(tool, arguments=arguments)


@pytest.mark.usefixtures("setup_database")
class TestCreateCampaignBoundary:
    """MCP-boundary failures for ``bo_create_campaign``."""

    @pytest.mark.asyncio
    async def test_empty_name_returns_structured_field_errors(self) -> None:
        result = await _call(
            "bo_create_campaign",
            arguments={
                "intake_data": {
                    "name": "",
                    "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
                    "objectives": [{"name": "y", "direction": "minimize"}],
                },
                "owner_id": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        # The structured envelope must surface; the offending field
        # path is the row-level ``name`` attribute.
        assert "name" in result["field_errors"]
        # Sanity: the error envelope still carries the canonical code
        # and recovery hint so agents can route on it.
        assert result["error"]["code"] == "E005"

    @pytest.mark.asyncio
    async def test_unknown_intake_field_returns_field_errors(self) -> None:
        """``extra=forbid`` errors must also reach the structured envelope.

        Before the fix, ``intake_data`` was typed as
        ``CampaignIntakeInput`` with ``extra=forbid`` so FastMCP
        intercepted unknown keys at validation time and raised
        ``ToolError``. Now the rejection happens inside the operation
        and reads as a structured ``field_errors`` entry.
        """
        result = await _call(
            "bo_create_campaign",
            arguments={
                "intake_data": {
                    "name": "Stray Key",
                    "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
                    "objectives": [{"name": "y", "direction": "minimize"}],
                    "not_a_real_field": True,
                },
                "owner_id": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert any(
            "not_a_real_field" in path or "not_a_real_field" in msg
            for path, msgs in result["field_errors"].items()
            for msg in msgs
        )

    @pytest.mark.asyncio
    async def test_valid_intake_round_trips_through_call_tool(self) -> None:
        """The dict-boundary path still creates campaigns for valid payloads."""
        result = await _call(
            "bo_create_campaign",
            arguments={
                "intake_data": {
                    "name": "Valid MCP Boundary",
                    "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
                    "objectives": [{"name": "y", "direction": "minimize"}],
                },
                "owner_id": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["campaign_id"]


@pytest.mark.usefixtures("setup_database")
class TestSubmitResultsBoundary:
    """MCP-boundary failures for ``bo_submit_results``."""

    @pytest.mark.asyncio
    async def test_bad_uncertainty_type_pins_results_row_path(self) -> None:
        """A non-dict ``measurement_uncertainty`` pins ``results[i]``.

        The Pydantic ``loc`` of ``("measurement_uncertainty",)`` is
        rebased onto the outer ``results[i]`` index by the tool
        boundary so agents can target the bad row directly without
        re-parsing the legacy ``errors`` list.
        """
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": str(uuid4()),
                "results": [
                    {
                        "parameter_values": {"x": 0.5},
                        "objective_values": {"y": 1.0},
                        "measurement_uncertainty": "not_a_dict",
                    }
                ],
                "submitted_by": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "results[0].measurement_uncertainty" in result["field_errors"]
        # The envelope keeps the canonical fields populated so
        # downstream callers do not have to special-case the
        # boundary-failure path.
        assert result["result_ids"] == []
        assert result["error"]["code"] == "E005"

    @pytest.mark.asyncio
    async def test_unknown_result_key_pins_results_row_path(self) -> None:
        """``ResultSubmissionInput`` rejects unknown keys with row-rooted paths."""
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": str(uuid4()),
                "results": [
                    {
                        "parameter_values": {"x": 0.5},
                        "objective_values": {"y": 1.0},
                        "not_a_real_key": 42,
                    }
                ],
                "submitted_by": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        # The unknown-key error path is rooted at the offending row.
        assert any(path.startswith("results[0]") for path in result["field_errors"])

    @pytest.mark.asyncio
    async def test_second_row_bad_pins_correct_index(self) -> None:
        """A failing later row reports its own index, not row 0's.

        This is the regression that motivated the field-error work:
        an agent that gets ``results[5].objective_values`` can edit
        row 5 directly instead of retrying the whole batch.
        """
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": str(uuid4()),
                "results": [
                    {
                        "parameter_values": {"x": 0.1},
                        "objective_values": {"y": 0.1},
                    },
                    {
                        "parameter_values": {"x": 0.2},
                        "objective_values": {"y": 0.2},
                        "measurement_uncertainty": "not_a_dict",
                    },
                ],
                "submitted_by": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        keys = list(result["field_errors"])
        assert any("results[1]" in k for k in keys), keys
        assert not any("results[0]" in k for k in keys), keys


@pytest.mark.usefixtures("setup_database")
class TestOuterShapeBoundary:
    """Outer-shape failures must also produce structured envelopes.

    Per-field Pydantic errors are only one half of the boundary
    contract: callers can also send the wrong outer shape entirely
    (a string instead of an intake object, a dict instead of a list
    of results, a string at ``results[0]``). With a typed ``dict`` /
    ``list[dict]`` boundary FastMCP would intercept these and raise
    ``ToolError``; the wrapper widens the boundary to ``Any`` and
    handles the shape check itself, so the failure path stays on the
    same ``field_errors`` envelope.
    """

    @pytest.mark.asyncio
    async def test_intake_data_not_an_object_returns_envelope(self) -> None:
        result = await _call(
            "bo_create_campaign",
            arguments={
                "intake_data": "not-an-object",
                "owner_id": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "intake_data" in result["field_errors"]
        # The envelope carries the canonical defaults so downstream
        # consumers do not have to special-case shape failures.
        assert result["campaign_id"] is None

    @pytest.mark.asyncio
    async def test_intake_data_as_list_returns_envelope(self) -> None:
        """A list is not an object either — distinct shape, same envelope."""
        result = await _call(
            "bo_create_campaign",
            arguments={
                "intake_data": [{"name": "wrong-shape"}],
                "owner_id": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "intake_data" in result["field_errors"]

    @pytest.mark.asyncio
    async def test_results_not_a_list_returns_envelope(self) -> None:
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": str(uuid4()),
                "results": "not-a-list",
                "submitted_by": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "results" in result["field_errors"]
        assert result["result_ids"] == []

    @pytest.mark.asyncio
    async def test_results_as_dict_returns_envelope(self) -> None:
        """A dict-shaped payload that should be a list is still a shape failure.

        Pinning the dict shape separately from the string shape (above)
        guards against a check that only catches scalars.
        """
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": str(uuid4()),
                "results": {"parameter_values": {"x": 0.5}},
                "submitted_by": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "results" in result["field_errors"]

    @pytest.mark.asyncio
    async def test_results_item_not_an_object_returns_envelope(self) -> None:
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": str(uuid4()),
                "results": ["not-a-dict"],
                "submitted_by": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "results[0]" in result["field_errors"]

    @pytest.mark.asyncio
    async def test_second_row_not_an_object_pins_correct_index(self) -> None:
        """A shape-bad row reports its own index, not row 0's."""
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": str(uuid4()),
                "results": [
                    {
                        "parameter_values": {"x": 0.1},
                        "objective_values": {"y": 0.1},
                    },
                    "still-not-a-dict",
                ],
                "submitted_by": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "results[1]" in result["field_errors"]
        assert "results[0]" not in result["field_errors"]


@pytest.mark.usefixtures("setup_database")
class TestScalarBoundary:
    """Missing or wrongly-typed scalar arguments produce envelopes too.

    Widening ``intake_data`` / ``results`` to ``Any`` covered outer
    container shape, but FastMCP still validates the remaining scalar
    args (``owner_id``, ``campaign_id``, ``submitted_by``, boolean
    flags) against the function signature and raises ``ToolError``
    before the tool body runs. The boundary wrapper catches that
    ToolError when its ``__cause__`` is a Pydantic ``ValidationError``
    and renders the same ``field_errors`` envelope. These tests pin
    that contract.
    """

    @pytest.mark.asyncio
    async def test_missing_intake_data_returns_envelope(self) -> None:
        result = await _call(
            "bo_create_campaign",
            arguments={"owner_id": str(uuid4())},
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "intake_data" in result["field_errors"]
        # The wrapper merges the per-tool defaults so the envelope
        # keeps the keys downstream consumers expect.
        assert result["campaign_id"] is None

    @pytest.mark.asyncio
    async def test_missing_results_returns_envelope(self) -> None:
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": str(uuid4()),
                "submitted_by": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "results" in result["field_errors"]
        assert result["result_ids"] == []

    @pytest.mark.asyncio
    async def test_non_string_owner_id_returns_envelope(self) -> None:
        result = await _call(
            "bo_create_campaign",
            arguments={
                "intake_data": {
                    "name": "x",
                    "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
                    "objectives": [{"name": "y", "direction": "minimize"}],
                },
                "owner_id": 12345,
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "owner_id" in result["field_errors"]

    @pytest.mark.asyncio
    async def test_non_string_campaign_id_returns_envelope(self) -> None:
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": 12345,
                "results": [
                    {
                        "parameter_values": {"x": 0.5},
                        "objective_values": {"y": 1.0},
                    }
                ],
                "submitted_by": str(uuid4()),
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "campaign_id" in result["field_errors"]

    @pytest.mark.asyncio
    async def test_non_string_submitted_by_returns_envelope(self) -> None:
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": str(uuid4()),
                "results": [
                    {
                        "parameter_values": {"x": 0.5},
                        "objective_values": {"y": 1.0},
                    }
                ],
                "submitted_by": 12345,
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "submitted_by" in result["field_errors"]

    @pytest.mark.asyncio
    async def test_invalid_bool_atomic_returns_envelope(self) -> None:
        result = await _call(
            "bo_submit_results",
            arguments={
                "campaign_id": str(uuid4()),
                "results": [
                    {
                        "parameter_values": {"x": 0.5},
                        "objective_values": {"y": 1.0},
                    }
                ],
                "submitted_by": str(uuid4()),
                "atomic": "not-a-bool",
            },
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "atomic" in result["field_errors"]


@pytest.mark.usefixtures("setup_database")
class TestValidateIntakeBoundary:
    """``bo_validate_intake`` shares the validation-localization workflow.

    Agents use it as a dry-run before ``bo_create_campaign`` -- the
    failure-path contract must match. The wrapper renders boundary
    failures with the same ``valid=False`` plus ``field_errors`` shape
    the operation layer produces for inner-field failures.
    """

    @pytest.mark.asyncio
    async def test_missing_intake_data_returns_envelope(self) -> None:
        result = await _call("bo_validate_intake", arguments={})
        assert isinstance(result, dict)
        # The validate-intake response carries ``valid=False`` instead
        # of the generic ``success=False`` flag used elsewhere; the
        # wrapper merges the per-tool defaults so this stays consistent.
        assert result["valid"] is False
        assert "intake_data" in result["field_errors"]
        assert result["spec"] is None

    @pytest.mark.asyncio
    async def test_string_intake_data_pins_single_field(self) -> None:
        """Outer-shape failure surfaces a clean ``intake_data`` path.

        Pre-fix the ``CampaignIntakeInput | dict[str, Any]`` union
        type forced Pydantic to walk both arms and emit internal ``loc``
        markers (``function-after[...]``, ``dict[str,any]``). Widening
        to ``Any`` and doing the shape check inline keeps the path
        addressable.
        """
        result = await _call(
            "bo_validate_intake",
            arguments={"intake_data": "not-an-object"},
        )
        assert isinstance(result, dict)
        assert result["valid"] is False
        assert list(result["field_errors"].keys()) == ["intake_data"]

    @pytest.mark.asyncio
    async def test_list_intake_data_returns_envelope(self) -> None:
        result = await _call(
            "bo_validate_intake",
            arguments={"intake_data": [{"name": "wrong-shape"}]},
        )
        assert isinstance(result, dict)
        assert result["valid"] is False
        assert "intake_data" in result["field_errors"]

    @pytest.mark.asyncio
    async def test_non_string_verbosity_returns_envelope(self) -> None:
        """Scalar-type failures are converted by the FastMCP-boundary wrapper.

        Verbosity is not in the wrapper's ``extra`` defaults map -- it
        is just a scalar arg validated by FastMCP. The wrapper still
        catches the resulting ``ToolError`` because its ``__cause__``
        is a ``ValidationError`` and ``bo_validate_intake`` is in
        ``_TOOL_ENVELOPE_DEFAULTS``.
        """
        result = await _call(
            "bo_validate_intake",
            arguments={"intake_data": {}, "verbosity": 12345},
        )
        assert isinstance(result, dict)
        assert result["valid"] is False
        assert "verbosity" in result["field_errors"]

    @pytest.mark.asyncio
    async def test_inner_field_failure_still_uses_operation_layer(self) -> None:
        """Inner-field failures stay on the operation-layer code path.

        The operation already produces per-field ``field_errors``;
        adding the wrapper must not regress that path. This test
        guards the contract by exercising the same fixture the
        operation tests use.
        """
        result = await _call(
            "bo_validate_intake",
            arguments={
                "intake_data": {
                    "parameters": [],
                    "objectives": [],
                }
            },
        )
        assert isinstance(result, dict)
        assert result["valid"] is False
        # ``name`` is a required intake field -- the operation-layer
        # validation pins it with the ``loc`` path the formatter
        # forwards verbatim.
        assert "name" in result["field_errors"]


@pytest.mark.usefixtures("setup_database")
class TestNonValidationToolErrorsPassthrough:
    """Non-validation ``ToolError`` (unknown tool, body raised) is not converted.

    The wrapper is opt-in per tool; it converts only ``ToolError``
    whose ``__cause__`` is a Pydantic ``ValidationError``. This test
    pins that contract so we do not accidentally swallow real
    transport-level errors.
    """

    @pytest.mark.asyncio
    async def test_unknown_tool_still_raises(self) -> None:
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            await _call("bo_definitely_not_a_tool", arguments={})


class TestSchemaDiscoverabilityPreserved:
    """Schema-level discoverability did not regress when we widened to dict.

    Accepting raw ``dict`` / ``list[dict]`` at the boundary risks
    flattening the per-field schema agents introspect on ``tools/list``.
    The tool definitions splice the original Pydantic JSON schema back
    into ``json_schema_extra`` so the structural hints stay visible.
    """

    @pytest.mark.asyncio
    async def test_create_campaign_intake_schema_still_advertises_required_fields(
        self,
    ) -> None:
        mcp = await _server()
        tools = await mcp.list_tools()
        schema = next(t for t in tools if t.name == "bo_create_campaign").inputSchema
        intake = schema["properties"]["intake_data"]
        # ``name``, ``parameters``, ``objectives`` are required intake
        # keys; the spliced schema must keep them in the advertised
        # ``required`` list so agents do not guess.
        assert {"name", "parameters", "objectives"}.issubset(set(intake.get("required", [])))
        # Nested ``parameters`` keeps its own schema (per-parameter
        # ``type`` / ``bounds`` etc.) -- not a bare ``object``.
        assert "items" in intake["properties"]["parameters"]

    @pytest.mark.asyncio
    async def test_submit_results_items_schema_documents_per_row_fields(
        self,
    ) -> None:
        mcp = await _server()
        tools = await mcp.list_tools()
        schema = next(t for t in tools if t.name == "bo_submit_results").inputSchema
        results = schema["properties"]["results"]
        # ``json_schema_extra`` splices an ``items`` schema describing
        # the ``ResultSubmissionInput`` shape; agents discover
        # ``parameter_values`` / ``objective_values`` /
        # ``measurement_uncertainty`` / ``metadata`` without round-
        # tripping a failing request.
        item_props = results["items"]["properties"]
        assert "parameter_values" in item_props
        assert "objective_values" in item_props
        assert "measurement_uncertainty" in item_props
