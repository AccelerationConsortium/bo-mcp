"""Tests for transport-neutral operations shared by MCP and HTTP."""

from uuid import UUID, uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.operations.batch_status import batch_get_status_operation
from bo_mcp_server.operations.campaign_lifecycle import manage_campaign_lifecycle_operation
from bo_mcp_server.operations.compare_campaigns import compare_campaigns_operation
from bo_mcp_server.operations.create_campaign import create_campaign_operation
from bo_mcp_server.operations.export_campaign import export_campaign_operation
from bo_mcp_server.operations.generate_suggestions import generate_suggestions_operation
from bo_mcp_server.operations.list_campaigns import list_campaigns_operation
from bo_mcp_server.operations.list_results import list_results_operation
from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
from bo_mcp_server.operations.submit_results import submit_results_operation
from bo_mcp_server.operations.suggestion_explanation import (
    get_suggestion_explanation_operation,
)
from bo_mcp_server.operations.transfer_candidates import (
    discover_transfer_candidates_operation,
)
from bo_mcp_server.operations.update_suggestion_status import (
    update_suggestion_status_operation,
)
from bo_mcp_server.tools.batch_operations import batch_get_status
from bo_mcp_server.tools.campaign_lifecycle import pause_campaign
from bo_mcp_server.tools.compare_campaigns import compare_campaigns
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.discover_transfer_candidates import discover_transfer_candidates
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_suggestion_explanation import get_suggestion_explanation
from bo_mcp_server.tools.list_campaigns import list_campaigns as list_campaigns_tool
from bo_mcp_server.tools.list_results import export_campaign as export_campaign_tool
from bo_mcp_server.tools.list_results import list_results as list_results_tool
from bo_mcp_server.tools.list_suggestions import list_suggestions as list_suggestions_tool
from bo_mcp_server.tools.submit_results import submit_results
from bo_mcp_server.tools.update_suggestion_status import (
    update_suggestion_status as update_suggestion_status_tool,
)


def _to_result_inputs(rows: list[dict]) -> list[ResultSubmissionInput]:
    return [
        ResultSubmissionInput(
            parameter_values=row["parameter_values"],
            objective_values=row["objective_values"],
            suggestion_id=row.get("suggestion_id"),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


async def _create_single_objective_campaign(owner_id: str, name: str) -> str:
    result = await create_campaign(
        {
            "name": name,
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        },
        owner_id,
    )
    return result["campaign_id"]


@pytest.mark.usefixtures("setup_database")
class TestSharedOperations:
    @pytest.mark.asyncio
    async def test_lifecycle_operation_matches_tool(self):
        owner_id = str(uuid4())
        operation_campaign_id = await _create_single_objective_campaign(
            owner_id,
            "Lifecycle Shared Op Test",
        )
        tool_campaign_id = await _create_single_objective_campaign(
            owner_id,
            "Lifecycle Tool Test",
        )
        await generate_suggestions(operation_campaign_id)
        await generate_suggestions(tool_campaign_id)

        operation_result = await manage_campaign_lifecycle_operation(
            operation_campaign_id,
            "pause",
        )
        tool_result = await pause_campaign(tool_campaign_id)

        assert operation_result["success"] is True
        assert tool_result["success"] is True
        assert tool_result["status"] == "paused"
        assert operation_result["status"] == "paused"
        assert tool_result["previous_status"] == operation_result["previous_status"] == "running"

    @pytest.mark.asyncio
    async def test_suggestion_explanation_operation_matches_tool(self):
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(
            owner_id,
            "Explanation Shared Op Test",
        )
        generated = await generate_suggestions(campaign_id)
        suggestion_id = generated["suggestions"][0]["id"]

        operation_result = await get_suggestion_explanation_operation(suggestion_id)
        tool_result = await get_suggestion_explanation(suggestion_id)

        assert operation_result == tool_result

    @pytest.mark.asyncio
    async def test_batch_status_operation_matches_tool(self):
        owner_id = str(uuid4())
        campaign_ids = [
            await _create_single_objective_campaign(owner_id, "Batch Shared Op Test A"),
            await _create_single_objective_campaign(owner_id, "Batch Shared Op Test B"),
        ]

        operation_result = await batch_get_status_operation(campaign_ids, verbosity="standard")
        tool_result = await batch_get_status(campaign_ids, verbosity="standard")

        assert operation_result == tool_result

    @pytest.mark.asyncio
    async def test_compare_operation_matches_tool(self):
        owner_id = str(uuid4())
        campaign_a = await _create_single_objective_campaign(owner_id, "Compare Shared Op Test A")
        campaign_b = await _create_single_objective_campaign(owner_id, "Compare Shared Op Test B")

        await generate_suggestions(campaign_a)
        await generate_suggestions(campaign_b)
        await submit_results(
            campaign_a,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )
        await submit_results(
            campaign_b,
            _to_result_inputs([{"parameter_values": {"x": 0.3}, "objective_values": {"y": 0.8}}]),
            owner_id,
        )

        operation_result = await compare_campaigns_operation(
            [campaign_a, campaign_b],
            verbosity="standard",
        )
        tool_result = await compare_campaigns([campaign_a, campaign_b], verbosity="standard")

        assert operation_result == tool_result

    @pytest.mark.asyncio
    async def test_transfer_candidates_operation_matches_tool(self):
        owner_id = str(uuid4())
        source = await create_campaign(
            {
                "name": "Shared Op Source",
                "parameters": [
                    {"name": "temperature", "type": "continuous", "bounds": [20.0, 100.0]},
                    {"name": "pressure", "type": "continuous", "bounds": [1.0, 10.0]},
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            },
            owner_id,
        )
        target = await create_campaign(
            {
                "name": "Shared Op Target",
                "parameters": [
                    {"name": "temperature", "type": "continuous", "bounds": [30.0, 90.0]},
                    {"name": "pressure", "type": "continuous", "bounds": [2.0, 8.0]},
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            },
            owner_id,
        )

        await generate_suggestions(source["campaign_id"])
        await submit_results(
            source["campaign_id"],
            _to_result_inputs(
                [
                    {
                        "parameter_values": {"temperature": 50.0, "pressure": 5.0},
                        "objective_values": {"yield": 0.8},
                    },
                    {
                        "parameter_values": {"temperature": 60.0, "pressure": 6.0},
                        "objective_values": {"yield": 0.85},
                    },
                    {
                        "parameter_values": {"temperature": 70.0, "pressure": 7.0},
                        "objective_values": {"yield": 0.9},
                    },
                ]
            ),
            owner_id,
        )

        operation_result = await discover_transfer_candidates_operation(
            target["campaign_id"],
            similarity_threshold=0.3,
            verbosity="standard",
        )
        tool_result = await discover_transfer_candidates(
            target["campaign_id"],
            similarity_threshold=0.3,
            verbosity="standard",
        )

        assert operation_result == tool_result

    @pytest.mark.asyncio
    async def test_list_campaigns_operation_matches_tool(self):
        owner_id = str(uuid4())
        await _create_single_objective_campaign(owner_id, "List Op Test A")
        await _create_single_objective_campaign(owner_id, "List Op Test B")

        operation_result = await list_campaigns_operation(
            owner_id=UUID(owner_id),
            verbosity="standard",
        )
        tool_result = await list_campaigns_tool(
            owner_id=owner_id,
            verbosity="standard",
        )

        assert operation_result == tool_result
        assert operation_result["total_count"] == 2

    @pytest.mark.asyncio
    async def test_list_results_operation_matches_tool(self):
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(owner_id, "ListResults Op Test")
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )

        operation_result = await list_results_operation(campaign_id, verbosity="standard")
        tool_result = await list_results_tool(campaign_id, verbosity="standard")

        assert operation_result == tool_result
        assert operation_result["total_count"] == 1

    @pytest.mark.asyncio
    async def test_list_suggestions_operation_matches_tool(self):
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(owner_id, "ListSugg Op Test")
        await generate_suggestions(campaign_id)

        operation_result = await list_suggestions_operation(campaign_id, verbosity="standard")
        tool_result = await list_suggestions_tool(campaign_id, verbosity="standard")

        assert operation_result == tool_result
        assert operation_result["total_count"] > 0

    @pytest.mark.asyncio
    async def test_export_campaign_operation_matches_tool(self):
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(owner_id, "Export Op Test")
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )

        operation_result = await export_campaign_operation(campaign_id)
        tool_result = await export_campaign_tool(campaign_id)

        assert operation_result == tool_result
        assert operation_result["n_results"] == 1
        assert "param_x" in operation_result["content"]

    @pytest.mark.asyncio
    async def test_update_suggestion_status_operation_matches_tool(self):
        owner_id = str(uuid4())

        # Create two campaigns with suggestions for independent testing
        campaign_a = await _create_single_objective_campaign(owner_id, "Status Op Test A")
        campaign_b = await _create_single_objective_campaign(owner_id, "Status Op Test B")
        gen_a = await generate_suggestions(campaign_a)
        gen_b = await generate_suggestions(campaign_b)
        suggestion_a = gen_a["suggestions"][0]["id"]
        suggestion_b = gen_b["suggestions"][0]["id"]

        operation_result = await update_suggestion_status_operation(suggestion_a, "accepted")
        tool_result = await update_suggestion_status_tool(suggestion_b, "accepted")

        assert operation_result["success"] is True
        assert tool_result["success"] is True
        assert operation_result["status"] == tool_result["status"] == "accepted"
        assert operation_result["previous_status"] == tool_result["previous_status"] == "pending"

    @pytest.mark.asyncio
    async def test_update_suggestion_status_transition_matrix(self):
        """Test all valid and invalid status transitions."""
        owner_id = str(uuid4())

        async def _fresh_suggestion() -> str:
            cid = await _create_single_objective_campaign(owner_id, f"Trans {uuid4().hex[:6]}")
            gen = await generate_suggestions(cid)
            return gen["suggestions"][0]["id"]

        # Valid: pending -> accepted
        sid = await _fresh_suggestion()
        result = await update_suggestion_status_operation(sid, "accepted")
        assert result["success"] is True

        # Valid: accepted -> rejected
        result = await update_suggestion_status_operation(sid, "rejected")
        assert result["success"] is True

        # Invalid: rejected -> accepted (no path back)
        result = await update_suggestion_status_operation(sid, "accepted")
        assert result["success"] is False

        # Valid: pending -> expired
        sid = await _fresh_suggestion()
        result = await update_suggestion_status_operation(sid, "expired")
        assert result["success"] is True

        # Invalid: pending -> completed (only set by submit_results)
        sid = await _fresh_suggestion()
        result = await update_suggestion_status_operation(sid, "completed")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_lifecycle_dry_run_returns_preview_without_mutating(self):
        """``dry_run`` validates the transition and returns a preview only.

        Reference: the dry-run / planning idiom mirrors ``terraform plan`` and
        the AWS ``--dry-run`` flag, both of which validate authorization and
        target state without committing the change so an automated caller can
        confirm an irreversible action.
        """
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(owner_id, "Dry Run Lifecycle")
        await generate_suggestions(campaign_id)

        preview = await manage_campaign_lifecycle_operation(
            campaign_id,
            "pause",
            dry_run=True,
        )
        assert preview["success"] is True
        assert preview["dry_run"] is True
        assert preview["status"] == "running"
        assert preview["previous_status"] == "running"
        assert preview["preview"]["from_status"] == "running"
        assert preview["preview"]["to_status"] == "paused"

        # Confirm no mutation took place — a real pause now must still succeed.
        committed = await manage_campaign_lifecycle_operation(campaign_id, "pause")
        assert committed["success"] is True
        assert committed["status"] == "paused"
        assert committed["previous_status"] == "running"

    @pytest.mark.asyncio
    async def test_update_suggestion_status_dry_run_preserves_pending(self):
        """Dry-run keeps the suggestion at ``pending`` so the real call still applies."""
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(owner_id, "Dry Run Status")
        generated = await generate_suggestions(campaign_id)
        suggestion_id = generated["suggestions"][0]["id"]

        preview = await update_suggestion_status_operation(
            suggestion_id,
            "accepted",
            dry_run=True,
        )
        assert preview["success"] is True
        assert preview["dry_run"] is True
        assert preview["preview"]["from_status"] == "pending"
        assert preview["preview"]["to_status"] == "accepted"

        # A subsequent real call must still see the suggestion in ``pending``.
        committed = await update_suggestion_status_operation(suggestion_id, "accepted")
        assert committed["success"] is True
        assert committed["previous_status"] == "pending"
        assert committed["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_create_campaign_dry_run_persists_nothing(self):
        """``dry_run=True`` validates the intake and emits a preview without writing."""
        owner_id = str(uuid4())
        intake = {
            "name": "Dry Run Create",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        preview = await create_campaign_operation(intake, owner_id, dry_run=True)
        assert preview["success"] is True
        assert preview["dry_run"] is True
        assert preview["campaign_id"] is None
        assert preview["spec_id"] is None
        assert preview["preview"]["n_parameters"] == 1
        assert preview["preview"]["n_objectives"] == 1

        # Confirm no campaign was created.
        listed = await list_campaigns_operation(owner_id=UUID(owner_id))
        assert listed["total_count"] == 0

    @pytest.mark.asyncio
    async def test_generate_suggestions_dry_run_does_not_advance_state(self):
        """Dry-run reports next iteration but leaves campaign + suggestions untouched."""
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(owner_id, "Dry Run Generate")

        preview = await generate_suggestions_operation(campaign_id=campaign_id, dry_run=True)
        assert preview["success"] is True
        assert preview["dry_run"] is True
        assert preview["suggestions"] == []
        assert preview["preview"]["next_iteration"] == 1
        assert preview["preview"]["planned_batch_size"] >= 1
        # The preview now includes the read-only preflight signals.
        assert preview["preview"]["n_pending"] == 0
        assert preview["preview"]["batch_clamped_by_budget"] is False

        # Real call still produces iteration 1 — dry-run did not advance state.
        first_real = await generate_suggestions(campaign_id)
        assert first_real["iteration"] == 1
        assert len(first_real["suggestions"]) >= 1

    @pytest.mark.asyncio
    async def test_generate_suggestions_dry_run_surfaces_pending_count(self):
        """Dry-run reports actionable-pending count instead of pretending none exist.

        Previously the dry-run preview ignored ``X_pending`` and the
        ``max_observations`` budget — a caller looking at the preview
        could not tell that the real call would clamp the batch.
        """
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(owner_id, "Dry Run Pending")
        # First real call seeds an actionable PENDING suggestion.
        await generate_suggestions(campaign_id)

        preview = await generate_suggestions_operation(campaign_id=campaign_id, dry_run=True)
        assert preview["success"] is True
        assert preview["dry_run"] is True
        # One actionable suggestion is now in flight from the earlier call.
        assert preview["preview"]["n_pending"] == 1
        assert preview["preview"]["actionable_breakdown"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_generate_suggestions_dry_run_routes_budget_stop(self):
        """Dry-run surfaces the same stopping envelope as a real call would.

        Reference: ``terraform plan`` mirrors apply-time refusals; an
        agentic workflow that consults the dry-run before committing
        must see the same budget-exhaustion verdict the real call
        would emit so it can route on
        ``next_action_recommendation == "terminate_campaign"`` without
        triggering the (otherwise wasted) BO algorithm.
        """
        owner_id = str(uuid4())
        # max_observations=1 + one submitted result -> budget exhausted.
        intake = {
            "name": "Dry Run Budget Stop",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "max_observations": 1,
        }
        create_result = await create_campaign(intake, owner_id)
        campaign_id = create_result["campaign_id"]
        await submit_results(
            campaign_id,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )

        preview = await generate_suggestions_operation(campaign_id=campaign_id, dry_run=True)
        # Stopping envelope is forwarded with ``dry_run=True`` so callers
        # can distinguish a planned no-op from a real generation.
        assert preview["dry_run"] is True
        assert preview["success"] is False
        error = preview.get("error") or {}
        details = error.get("details") or {}
        assert details.get("next_action_recommendation") == "terminate_campaign"

    @pytest.mark.asyncio
    async def test_wrapper_validation_envelope_echoes_trace_id(self):
        """Tool-wrapper validation envelopes also carry the bound trace id.

        ``bo_create_campaign`` runs a shape check before delegating to
        the operation. That early-return path bypasses
        ``with_response_metadata``, so without the wrapper-level
        ``attach_response_metadata`` it would emit a validation
        envelope with no ``_metadata`` block — meaning the only tool
        returns that drop trace metadata would be exactly the ones a
        debugging operator needs it on.
        """
        from bo_mcp_server.tools.create_campaign import (
            create_campaign as create_campaign_tool,
        )
        from bo_mcp_server.trace_context import bind_trace_id

        # Pass a non-object intake to force the shape envelope. The
        # error path predates ``with_response_metadata``; trace echo
        # has to come from the wrapper.
        with bind_trace_id("trace-bad-intake"):
            response = await create_campaign_tool(
                "not an object",  # type: ignore[arg-type]
                str(uuid4()),
            )

        assert response["success"] is False
        assert response.get("_metadata", {}).get("trace_id") == "trace-bad-intake"

    @pytest.mark.asyncio
    async def test_lifecycle_response_echoes_trace_id_metadata(self):
        """Raw-dict lifecycle responses still echo trace_id in ``_metadata``.

        Before the operation-level decorator was added, only
        formatter-routed responses (diagnostics, suggestions, …)
        carried ``_metadata.trace_id``. Lifecycle responses returned a
        plain dict and dropped the field on the floor, so the cookbook
        promise that "every formatted response echoes trace_id" was
        false for pause / resume / terminate. The
        ``with_response_metadata`` wrapper closes that gap.
        """
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(owner_id, "Lifecycle Trace")
        await generate_suggestions(campaign_id)

        from bo_mcp_server.trace_context import bind_trace_id

        with bind_trace_id("trace-lifecycle-echo"):
            response = await manage_campaign_lifecycle_operation(campaign_id, "pause")

        assert response["success"] is True
        assert response["_metadata"]["trace_id"] == "trace-lifecycle-echo"

    @pytest.mark.asyncio
    async def test_update_suggestion_status_tool_trace_id_reaches_audit(self):
        """The MCP tool's ``trace_id`` arg lands on the audit event input_summary.

        Drives the real :func:`bo_mcp_server.tools.update_suggestion_status`
        tool (not ``log_tool_call``) because that operation writes its
        ``LIFECYCLE`` ``Event`` row directly via ``EventRepository.save``.
        With the trace-id splice centralized at the repository level,
        the persisted event picks up the bound trace id without any
        extra wiring in the operation.

        Reference: W3C trace-context recommends one trace id per
        logical workflow — https://www.w3.org/TR/trace-context/#trace-id.
        """
        from bo_mcp_server.storage import EventRepository, get_session
        from bo_mcp_server.tools.update_suggestion_status import (
            update_suggestion_status as update_suggestion_status_real_tool,
        )

        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(owner_id, "Trace Audit")
        generated = await generate_suggestions(campaign_id)
        suggestion_id = generated["suggestions"][0]["id"]

        result = await update_suggestion_status_real_tool(
            suggestion_id,
            "accepted",
            trace_id="workflow-audit",
        )
        assert result["success"] is True

        async with get_session() as session:
            events = await EventRepository(session).list_by_campaign(UUID(campaign_id))
        tagged = [e for e in events if (e.input_summary or {}).get("trace_id") == "workflow-audit"]
        assert tagged, (
            "operation-level Event row must carry the bound trace id under "
            "input_summary (centralized in EventRepository.save)"
        )

    @pytest.mark.asyncio
    async def test_submit_results_dry_run_does_not_mutate_suggestion(self):
        """``dry_run=True`` must leave the linked suggestion in ``pending``.

        Reference: the dry-run / planning idiom commits no state changes —
        the same semantics as ``terraform plan`` and AWS ``--dry-run`` flags.
        Phase 2 of submit_results historically transitioned the suggestion
        to ``completed`` before the response was assembled, defeating
        the preview contract.
        """
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(owner_id, "Dry Run Submit")
        generated = await generate_suggestions(campaign_id)
        suggestion = generated["suggestions"][0]
        suggestion_id = suggestion["id"]
        param_value = suggestion["parameter_values"]["x"]

        preview = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "parameter_values": {"x": param_value},
                        "objective_values": {"y": 0.42},
                        "suggestion_id": suggestion_id,
                    }
                ]
            ),
            submitted_by=owner_id,
            dry_run=True,
        )
        assert preview["success"] is True
        assert preview["dry_run"] is True
        assert preview["result_ids"] == []
        assert preview["preview"]["rows_would_persist"] == 1

        # Suggestion must still be actionable for a real submission.
        listed = await list_suggestions_operation(campaign_id=campaign_id)
        statuses = {s["suggestion_id"]: s["status"] for s in listed["suggestions"]}
        assert statuses[suggestion_id] == "pending"

        committed = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "parameter_values": {"x": param_value},
                        "objective_values": {"y": 0.42},
                        "suggestion_id": suggestion_id,
                    }
                ]
            ),
            submitted_by=owner_id,
        )
        assert committed["success"] is True
        listed_after = await list_suggestions_operation(campaign_id=campaign_id)
        statuses_after = {s["suggestion_id"]: s["status"] for s in listed_after["suggestions"]}
        assert statuses_after[suggestion_id] == "completed"

    @pytest.mark.asyncio
    async def test_list_campaigns_pagination(self):
        """Test that pagination works correctly."""
        owner_id = str(uuid4())
        for i in range(5):
            await _create_single_objective_campaign(owner_id, f"Page Test {i}")

        page1 = await list_campaigns_operation(owner_id=UUID(owner_id), limit=2, offset=0)
        page2 = await list_campaigns_operation(owner_id=UUID(owner_id), limit=2, offset=2)
        page3 = await list_campaigns_operation(owner_id=UUID(owner_id), limit=2, offset=4)

        assert page1["total_count"] == 5
        assert len(page1["campaigns"]) == 2
        assert len(page2["campaigns"]) == 2
        assert len(page3["campaigns"]) == 1

        # No overlap between pages
        ids_1 = {c["campaign_id"] for c in page1["campaigns"]}
        ids_2 = {c["campaign_id"] for c in page2["campaigns"]}
        ids_3 = {c["campaign_id"] for c in page3["campaigns"]}
        assert not ids_1 & ids_2
        assert not ids_2 & ids_3
