"""Contract tests for the suggestion identity key across shared operations.

``suggestion_id`` is the canonical identity key for a suggestion: the
generate and list surfaces both emit it and result submission consumes
it, so its value can be copied into a submission without renaming.
``id`` is a deprecated alias that the generate surface keeps emitting
for existing clients; removing it is a deliberate contract change that
must fail these tests first.
"""

from uuid import uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.operations.idempotency_wrapper import (
    canonical_generate_suggestions_payload,
)
from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
from bo_mcp_server.operations.submit_results import submit_results_operation
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.response_models import GenerateSuggestionsResponse
from tests.factories import seed_owner

UNLINKED_WARNING_MARKER = "without suggestion_id"


async def _create_campaign(owner_id: str, name: str) -> str:
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
class TestSuggestionIdentityKey:
    @pytest.mark.asyncio
    async def test_generate_emits_canonical_key_and_deprecated_alias(self):
        owner_id = await seed_owner()
        campaign_id = await _create_campaign(owner_id, "Identity Key Generate")

        generated = await generate_suggestions(campaign_id)

        assert generated["success"] is True
        assert generated["suggestions"]
        for suggestion in generated["suggestions"]:
            assert suggestion["suggestion_id"] == suggestion["id"]

    @pytest.mark.asyncio
    async def test_generate_and_list_agree_on_identity(self):
        owner_id = await seed_owner()
        campaign_id = await _create_campaign(owner_id, "Identity Key Cross Surface")

        generated = await generate_suggestions(campaign_id)
        listed = await list_suggestions_operation(campaign_id, verbosity="standard")

        generated_ids = {s["suggestion_id"] for s in generated["suggestions"]}
        listed_ids = {s["suggestion_id"] for s in listed["suggestions"]}
        assert generated_ids == listed_ids

    @pytest.mark.asyncio
    async def test_all_list_verbosities_carry_suggestion_id(self):
        owner_id = await seed_owner()
        campaign_id = await _create_campaign(owner_id, "Identity Key Verbosity")
        await generate_suggestions(campaign_id)

        for verbosity in ("minimal", "standard", "detailed"):
            listed = await list_suggestions_operation(campaign_id, verbosity=verbosity)
            assert listed["success"] is True
            assert listed["suggestions"], verbosity
            for suggestion in listed["suggestions"]:
                assert suggestion["suggestion_id"], verbosity

    @pytest.mark.asyncio
    async def test_minimal_generate_projection_resolves_ids(self):
        owner_id = await seed_owner()
        campaign_id = await _create_campaign(owner_id, "Identity Key Minimal Generate")

        generated = await generate_suggestions(campaign_id, verbosity="minimal")

        assert generated["success"] is True
        assert generated["suggestion_ids"]
        assert all(sid is not None for sid in generated["suggestion_ids"])

    @pytest.mark.asyncio
    async def test_generate_payload_round_trips_into_result_submission(self):
        """The key read from a generate payload must submit verbatim.

        This pins the failure mode the canonical key exists to prevent:
        a caller copying ``suggestion_id`` from a generated suggestion
        into a result submission without renaming must link the result
        and complete the suggestion.
        """
        owner_id = await seed_owner()
        campaign_id = await _create_campaign(owner_id, "Identity Key Round Trip")

        generated = await generate_suggestions(campaign_id)
        suggestion = generated["suggestions"][0]

        submitted = await submit_results_operation(
            campaign_id,
            [
                ResultSubmissionInput(
                    parameter_values=suggestion["parameter_values"],
                    objective_values={"y": 0.5},
                    suggestion_id=suggestion["suggestion_id"],
                )
            ],
            submitted_by=owner_id,
        )

        assert submitted["success"] is True, submitted["errors"]
        assert not any(UNLINKED_WARNING_MARKER in w for w in submitted["warnings"])

        completed = await list_suggestions_operation(campaign_id, status_filter="completed")
        completed_ids = {s["suggestion_id"] for s in completed["suggestions"]}
        assert suggestion["suggestion_id"] in completed_ids

    @pytest.mark.asyncio
    async def test_unlinked_api_submission_with_open_suggestions_warns(self):
        """Unlinked rows stay legal but must be visible at submit time.

        A programmatic (``api``) submission without ``suggestion_id``
        while actionable suggestions are open is the signature of a
        caller that dropped the key: the result is accepted but the
        originating suggestion silently stays pending, so the response
        must warn.
        """
        owner_id = await seed_owner()
        campaign_id = await _create_campaign(owner_id, "Identity Key Unlinked Warn")
        await generate_suggestions(campaign_id)

        submitted = await submit_results_operation(
            campaign_id,
            [
                ResultSubmissionInput(
                    parameter_values={"x": 0.123},
                    objective_values={"y": 1.0},
                )
            ],
            submitted_by=owner_id,
            source="api",
        )

        assert submitted["success"] is True, submitted["errors"]
        assert any(UNLINKED_WARNING_MARKER in w for w in submitted["warnings"])

    @pytest.mark.asyncio
    async def test_seed_submission_before_generation_does_not_warn(self):
        """Seeding observations before any generation is intentional.

        With no actionable suggestion to link to, an unlinked row
        cannot be a dropped-key mistake, so no warning fires.
        """
        owner_id = await seed_owner()
        campaign_id = await _create_campaign(owner_id, "Identity Key Seed Silent")

        submitted = await submit_results_operation(
            campaign_id,
            [
                ResultSubmissionInput(
                    parameter_values={"x": 0.25},
                    objective_values={"y": 2.0},
                )
            ],
            submitted_by=owner_id,
            source="api",
        )

        assert submitted["success"] is True, submitted["errors"]
        assert not any(UNLINKED_WARNING_MARKER in w for w in submitted["warnings"])

    @pytest.mark.asyncio
    async def test_unlinked_gui_submission_does_not_warn(self):
        """Manual GUI entries are legitimately unlinked — no warning."""
        owner_id = await seed_owner()
        campaign_id = await _create_campaign(owner_id, "Identity Key GUI Silent")
        await generate_suggestions(campaign_id)

        submitted = await submit_results_operation(
            campaign_id,
            [
                ResultSubmissionInput(
                    parameter_values={"x": 0.75},
                    objective_values={"y": 3.0},
                )
            ],
            submitted_by=owner_id,
            source="gui",
        )

        assert submitted["success"] is True, submitted["errors"]
        assert not any(UNLINKED_WARNING_MARKER in w for w in submitted["warnings"])

    @pytest.mark.asyncio
    async def test_legacy_idempotency_replay_gains_suggestion_id(self):
        """Cache entries persisted before ``suggestion_id`` are normalized.

        The idempotency cache is DB-backed and outlives a deployment,
        so for the cache lifetime a retry can replay a response whose
        suggestions only carry ``id``. The MCP tool must backfill
        ``suggestion_id`` on the replay.
        """
        campaign_id = str(uuid4())
        idempotency_key = str(uuid4())
        legacy_response = {
            "success": True,
            "suggestions": [
                {
                    "id": str(uuid4()),
                    "parameter_values": {"x": 0.5},
                    "provenance": {
                        "iteration": 1,
                        "batch_index": 0,
                        "generation_method": "sobol",
                    },
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ],
            "iteration": 1,
            "errors": [],
            "warnings": [],
            "method_selection": {},
        }

        async def seed_legacy(_session):
            return legacy_response

        seeded = await apply_idempotency(
            tool_name="bo_generate_suggestions",
            idempotency_key=idempotency_key,
            request_payload=canonical_generate_suggestions_payload(
                campaign_id=campaign_id,
                batch_size=None,
                verbosity="standard",
            ),
            executor=seed_legacy,
        )
        assert seeded["success"] is True

        replayed = await generate_suggestions(campaign_id, idempotency_key=idempotency_key)

        assert replayed["idempotency_replay"] is True
        assert replayed["suggestions"]
        for suggestion in replayed["suggestions"]:
            assert suggestion["suggestion_id"] == suggestion["id"]


class TestGenerateSuggestionsOutputSchema:
    def test_schema_declares_canonical_identity_key(self):
        """Schema-driven MCP clients must be able to discover the key.

        The generated-suggestion item must declare ``suggestion_id``
        as required and keep the deprecated ``id`` alias visible.
        ``id`` is required too: the compatibility contract promises it
        on every generated suggestion until its scheduled removal, so
        dropping it early must fail here first.
        """
        schema = GenerateSuggestionsResponse.model_json_schema()
        item_schema = schema["$defs"]["GeneratedSuggestionItem"]

        assert "suggestion_id" in item_schema["properties"]
        assert "id" in item_schema["properties"]
        assert "suggestion_id" in item_schema["required"]
        assert "id" in item_schema["required"]
        assert item_schema["properties"]["id"].get("deprecated") is True
