"""Pre-persistence backend-state size backstop (all backends).

PostgreSQL rejects any single wire-protocol message over 1 GiB by
dropping the connection — the failure mode that turned an oversized
BayBE campaign state into an untyped E199 with a dead DB connection
(see the PostgreSQL message-format docs,
https://www.postgresql.org/docs/current/protocol-message-formats.html).
These tests pin the server-side defense in depth: an oversized
``backend_state`` from *any* backend raises the typed
:class:`BackendStateTooLargeError` before the database write, mapping to
the non-retryable ``BACKEND_STATE_TOO_LARGE`` (E109) envelope.
"""

from __future__ import annotations

import pytest

from bo_engine.backend import SuggestionBatch
from bo_mcp_server.errors import (
    ERROR_CODE_RETRY_HINTS,
    ERROR_CODE_TO_HTTP_STATUS,
    ErrorCode,
)
from bo_mcp_server.operations.backend_output import (
    BackendOutputError,
    BackendStateTooLargeError,
    validate_backend_batch,
)
from tests.factories import seed_owner

_PROVENANCE = {
    "iteration": 1,
    "batch_index": 0,
    "generation_method": "bo",
    "random_seed": 1,
}


def _batch(state: dict | None) -> SuggestionBatch:
    return SuggestionBatch(
        suggestions=[{"parameter_values": {"x": 0.5}, "provenance": dict(_PROVENANCE)}],
        method_info={},
        backend_state=state,
        warnings=[],
    )


class TestSizeBackstop:
    def test_oversized_state_raises_typed_error(self) -> None:
        state = {"blob": "x" * 2048}
        with pytest.raises(BackendStateTooLargeError) as excinfo:
            validate_backend_batch(_batch(state), max_backend_state_bytes=1024)
        assert excinfo.value.limit_bytes == 1024
        assert excinfo.value.state_bytes > 1024

    def test_size_error_is_a_backend_output_error(self) -> None:
        """The generation loop's existing except branch must catch it."""
        assert issubclass(BackendStateTooLargeError, BackendOutputError)

    def test_state_within_limit_passes(self) -> None:
        batch = _batch({"blob": "x" * 10})
        assert validate_backend_batch(batch, max_backend_state_bytes=1024) is batch

    def test_zero_limit_disables_the_check(self) -> None:
        batch = _batch({"blob": "x" * 2048})
        assert validate_backend_batch(batch, max_backend_state_bytes=0) is batch

    def test_none_state_is_exempt(self) -> None:
        batch = _batch(None)
        assert validate_backend_batch(batch, max_backend_state_bytes=1) is batch

    def test_non_serializable_state_still_rejected(self) -> None:
        """The single-dump refactor keeps the serializability contract."""
        batch = _batch({"tensor": object()})
        with pytest.raises(BackendOutputError, match="not JSON-serializable"):
            validate_backend_batch(batch)


class TestEnvelopeWiring:
    def test_error_code_maps(self) -> None:
        assert ErrorCode.BACKEND_STATE_TOO_LARGE.value == "E109"
        assert ERROR_CODE_TO_HTTP_STATUS[ErrorCode.BACKEND_STATE_TOO_LARGE] == 422
        retryable, _backoff = ERROR_CODE_RETRY_HINTS[ErrorCode.BACKEND_STATE_TOO_LARGE]
        assert retryable is False

    def test_settings_default_is_below_postgres_protocol_limit(self) -> None:
        from bo_mcp_server.settings import get_max_backend_state_bytes

        one_gib = 1024**3
        assert 0 < get_max_backend_state_bytes() < one_gib


@pytest.mark.usefixtures("setup_database")
class TestEndToEndBackstopWiring:
    """The settings→operation threading of the limit, exercised end to end.

    The validator and the failure handler are unit-tested above in
    isolation; this test drives ``generate_suggestions_operation`` against
    the in-memory database with a stub backend emitting an oversized
    state, so removing the single ``validate_backend_batch(...,
    max_backend_state_bytes=...)`` call-site wiring (or its settings
    threading) fails a test instead of silently disabling the backstop.
    """

    @pytest.mark.asyncio
    async def test_oversized_state_yields_e109_and_no_writes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from uuid import UUID

        from bo_mcp_server.operations import generate_suggestions as gen_mod
        from bo_mcp_server.storage import (
            CampaignRepository,
            SuggestionRepository,
            get_session,
        )
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = await seed_owner()
        created = await create_campaign(
            {
                "name": "Oversized State Backstop",
                "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
            owner_id,
        )
        campaign_id = created["campaign_id"]
        campaign_uuid = UUID(campaign_id)

        async with get_session() as session:
            campaign_before = await CampaignRepository(session).get(campaign_uuid)
        assert campaign_before is not None

        class _OversizedStateBackend:
            """Minimal BOBackend stand-in returning a too-large state blob."""

            name = "stub-oversized"

            def generate_suggestions(
                self,
                spec,
                observations,
                batch_size,
                iteration,
                backend_state=None,
                pending_points=None,
                progress_callback=None,
                initial_design_history=None,
            ) -> SuggestionBatch:
                _ = (spec, observations, batch_size, backend_state, pending_points)
                _ = (progress_callback, initial_design_history)
                provenance = dict(_PROVENANCE)
                provenance["iteration"] = iteration
                return SuggestionBatch(
                    suggestions=[{"parameter_values": {"x": 0.5}, "provenance": provenance}],
                    method_info={},
                    backend_state={"blob": "x" * 4096},
                    warnings=[],
                )

        async def _stub_backend(_name: str) -> _OversizedStateBackend:
            return _OversizedStateBackend()

        monkeypatch.setattr(gen_mod, "get_backend_async", _stub_backend)
        monkeypatch.setattr(gen_mod, "get_max_backend_state_bytes", lambda: 1024)

        response = await gen_mod.generate_suggestions_operation(campaign_id)

        envelope = response["error"]
        assert response["success"] is False
        assert envelope["code"] == ErrorCode.BACKEND_STATE_TOO_LARGE.value
        assert envelope["retryable"] is False
        assert envelope["details"]["limit_bytes"] == 1024
        assert envelope["details"]["state_bytes"] > 1024
        assert response["suggestions"] == []

        # Atomicity: the failure fired before the persist phase, so the
        # campaign row is untouched and no suggestion rows exist.
        async with get_session() as session:
            campaign_after = await CampaignRepository(session).get(campaign_uuid)
            assert campaign_after is not None
            assert campaign_after.version == campaign_before.version
            assert campaign_after.iteration == campaign_before.iteration
            assert campaign_after.backend_state == campaign_before.backend_state
            suggestions = await SuggestionRepository(session).list_by_campaign(campaign_uuid)
            assert suggestions == []


class TestGenerationFailureHandling:
    @pytest.mark.asyncio
    async def test_handler_emits_e109_envelope_with_sizes(self) -> None:
        """The generation loop maps the size error to the E109 envelope.

        Structural atomicity note: the error is raised in the compute
        phase of the three-phase generation split, *before* the persist
        phase opens its write transaction — so no campaign version bump
        and no orphaned suggestion rows can exist by construction.
        """
        from bo_mcp_server.operations.generate_suggestions import (
            _handle_generation_failure,
        )

        err = BackendStateTooLargeError(state_bytes=2_000_000, limit_bytes=1_000_000)
        response = await _handle_generation_failure(
            err, "00000000-0000-0000-0000-000000000000", None
        )
        envelope = response["error"]
        assert envelope["code"] == ErrorCode.BACKEND_STATE_TOO_LARGE.value
        assert envelope["retryable"] is False
        assert envelope["details"]["state_bytes"] == 2_000_000
        assert envelope["details"]["limit_bytes"] == 1_000_000
