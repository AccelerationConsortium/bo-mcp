"""BayBE state reconciliation via stable identity tests.

Replaces the previous count-prefix reconciliation
(``len(campaign.measurements)`` vs ``len(observations)``) with an
identity index stored in the backend-state payload. Reordered or
shuffled observations must produce the same set of new measurements,
and a backfilled observation inserted before the prior prefix must be
detected as new.
"""

from __future__ import annotations

from baybe import Campaign
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

from bo_engine_baybe.backend import BayBEBackend


def _spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


def _measurement_count(state: dict | None) -> int:
    """Round-trip the BayBE campaign JSON and count its measurements.

    The restored ``Campaign.measurements`` is a pandas DataFrame with
    one row per stored measurement — the cleanest source of truth for
    the test, since BayBE's serialized JSON shape is an internal detail.
    """
    assert state is not None
    payload = state["payload"]
    campaign = Campaign.from_json(payload["campaign_json"])
    return len(campaign.measurements)


def _stored_identity_count(state: dict | None) -> int:
    assert state is not None
    return len(state["payload"].get("observation_identity", []))


class TestIdentityReconciliation:
    def test_reordered_observations_do_not_double_add(self) -> None:
        """Shuffling the observation list across calls is a no-op for BayBE."""
        backend = BayBEBackend()
        spec = _spec()
        obs1 = [
            ObservationData(parameter_values={"x1": 0.1, "x2": 0.9}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.4, "x2": 0.6}, objective_values={"y": 0.5}),
            ObservationData(parameter_values={"x1": 0.7, "x2": 0.3}, objective_values={"y": 0.3}),
        ]
        batch1 = backend.generate_suggestions(spec, obs1, batch_size=1, iteration=1)
        assert _measurement_count(batch1.backend_state) == 3

        # Re-run with the same observations in a different order.
        obs2 = [obs1[2], obs1[0], obs1[1]]
        batch2 = backend.generate_suggestions(
            spec,
            obs2,
            batch_size=1,
            iteration=2,
            backend_state=batch1.backend_state,
        )
        assert _measurement_count(batch2.backend_state) == 3

    def test_backfilled_observation_is_added(self) -> None:
        """A new observation inserted before the prior prefix is detected as new."""
        backend = BayBEBackend()
        spec = _spec()
        obs1 = [
            ObservationData(parameter_values={"x1": 0.3, "x2": 0.7}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.8, "x2": 0.2}, objective_values={"y": 0.5}),
        ]
        batch1 = backend.generate_suggestions(spec, obs1, batch_size=1, iteration=1)
        assert _measurement_count(batch1.backend_state) == 2

        # Backfill: a third observation that should have been logged earlier
        # is inserted before the existing prefix.
        obs2 = [
            ObservationData(parameter_values={"x1": 0.1, "x2": 0.1}, objective_values={"y": 0.2}),
            obs1[0],
            obs1[1],
        ]
        batch2 = backend.generate_suggestions(
            spec,
            obs2,
            batch_size=1,
            iteration=2,
            backend_state=batch1.backend_state,
        )
        assert _measurement_count(batch2.backend_state) == 3

    def test_rebuild_when_stored_identity_misses(self) -> None:
        """Stored identities that vanish from storage trigger a rebuild."""
        backend = BayBEBackend()
        spec = _spec()
        obs1 = [
            ObservationData(parameter_values={"x1": 0.2, "x2": 0.2}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.7}),
            ObservationData(parameter_values={"x1": 0.8, "x2": 0.8}, objective_values={"y": 0.3}),
        ]
        batch1 = backend.generate_suggestions(spec, obs1, batch_size=1, iteration=1)

        # Caller drops one result: storage no longer contains the third row.
        obs2 = obs1[:2]
        batch2 = backend.generate_suggestions(
            spec,
            obs2,
            batch_size=1,
            iteration=2,
            backend_state=batch1.backend_state,
        )
        assert _measurement_count(batch2.backend_state) == 2


class TestLegacyStateFallback:
    def test_legacy_state_without_identity_still_loads(self) -> None:
        """v1 payloads (no identity index) must still produce valid suggestions."""
        backend = BayBEBackend()
        spec = _spec()
        obs = [
            ObservationData(parameter_values={"x1": 0.3, "x2": 0.7}, objective_values={"y": 1.0}),
        ]
        batch1 = backend.generate_suggestions(spec, obs, batch_size=1, iteration=1)
        assert batch1.backend_state is not None

        # Strip the identity index to mimic a legacy payload — only campaign_json survives.
        legacy_state = {
            "backend": "baybe",
            "schema_version": 1,
            "payload": {"campaign_json": batch1.backend_state["payload"]["campaign_json"]},
        }
        batch2 = backend.generate_suggestions(
            spec,
            obs,
            batch_size=1,
            iteration=2,
            backend_state=legacy_state,
        )
        assert len(batch2.suggestions) == 1
        # The identity reconciliation treats the legacy state as "no prior identities";
        # all current observations look new and get re-added to the rebuilt campaign.
        assert _measurement_count(batch2.backend_state) == 1
