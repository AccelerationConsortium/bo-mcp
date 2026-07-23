"""Contract tests for the suggestion identity key across shared operations.

``suggestion_id`` is the only identity key for a suggestion: the
generate and list surfaces both emit it and result submission consumes
it, so its value can be copied into a submission without renaming.
The retired ``id`` alias must not reappear on any surface —
reintroducing it is a deliberate contract change that must fail these
tests first.
"""

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
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
    async def test_generate_emits_only_the_canonical_key(self):
        owner_id = await seed_owner()
        campaign_id = await _create_campaign(owner_id, "Identity Key Generate")

        generated = await generate_suggestions(campaign_id)

        assert generated["success"] is True
        assert generated["suggestions"]
        for suggestion in generated["suggestions"]:
            assert suggestion["suggestion_id"]
            assert "id" not in suggestion

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


class TestGenerateSuggestionsOutputSchema:
    def test_schema_declares_only_the_canonical_identity_key(self):
        """Schema-driven MCP clients must be able to discover the key.

        The generated-suggestion item must declare ``suggestion_id``
        as required, and the retired ``id`` alias must not reappear in
        the schema.
        """
        schema = GenerateSuggestionsResponse.model_json_schema()
        item_schema = schema["$defs"]["GeneratedSuggestionItem"]

        assert "suggestion_id" in item_schema["properties"]
        assert "suggestion_id" in item_schema["required"]
        assert "id" not in item_schema["properties"]
