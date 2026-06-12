"""Per-row discriminator for BayBE observation identity.

The original ``_observation_fingerprint`` hashed only
``(parameter_values, objective_values)`` so two replicate rows
collapsed onto the same identity slot and could only be reconciled by
a positional ``Counter`` pass. The audit asked for a stable cross-system
discriminator so a specific replicate can be addressed across reorder
/ restart cycles.

This module pins:

1. Identical replicate rows produce **distinct** identities when a
   per-row ``result_id`` is threaded through.
2. The legacy direct-engine path (``result_id=None``) still produces
   the same fingerprint for identical replicates — the multiset
   reconciler covers that case and behaviour is preserved.
3. v1 payload restore emits the structured WARNING the operator
   dashboards key off, with ``migration=v1_rebuild`` in
   ``LogRecord.__dict__`` so log filters can target it.
4. The discriminator survives a serialize → restore → re-serialize
   round-trip: the persisted identity index references the same
   ``result_id`` values.

References
==========

* Python ``hashlib.blake2b``: https://docs.python.org/3/library/hashlib.html#hashlib.blake2b
"""

from __future__ import annotations

import json
import logging

import pytest
from baybe import Campaign

from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.backend import (
    BayBEBackend,
    _build_observation_identity,
    _observation_fingerprint,
)


def _spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


# ---------------------------------------------------------------------------
# Fingerprint-level contract
# ---------------------------------------------------------------------------


class TestFingerprintDiscriminator:
    def test_identical_replicates_with_distinct_result_ids_diverge(self) -> None:
        """Two byte-identical rows with different result_ids hash differently.

        Without the discriminator the audit-flagged regression bites:
        external callers cannot address one replicate vs the other, and
        reconciliation falls back to the positional Counter pass.
        """
        obs_a = ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.5},
            objective_values={"y": 1.0},
            result_id="row-1",
        )
        obs_b = ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.5},
            objective_values={"y": 1.0},
            result_id="row-2",
        )
        fp_a = _observation_fingerprint(obs_a, ["x1", "x2"], ["y"])
        fp_b = _observation_fingerprint(obs_b, ["x1", "x2"], ["y"])
        assert fp_a != fp_b

    def test_identical_replicates_without_result_ids_collide(self) -> None:
        """Legacy direct-engine path (no result_id) keeps the multiset semantics.

        Direct-engine callers that do not thread an ID through still get
        the same fingerprint for identical rows; the multiset reconciler
        in ``_reconcile_measurements`` handles those replicate slots.
        """
        obs_a = ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.5},
            objective_values={"y": 1.0},
        )
        obs_b = ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.5},
            objective_values={"y": 1.0},
        )
        fp_a = _observation_fingerprint(obs_a, ["x1", "x2"], ["y"])
        fp_b = _observation_fingerprint(obs_b, ["x1", "x2"], ["y"])
        assert fp_a == fp_b

    def test_reorder_preserves_per_row_identity(self) -> None:
        """Identity is keyed on payload + result_id, not on list position."""
        obs_a = ObservationData(
            parameter_values={"x1": 0.1, "x2": 0.9},
            objective_values={"y": 1.0},
            result_id="row-A",
        )
        obs_b = ObservationData(
            parameter_values={"x1": 0.4, "x2": 0.6},
            objective_values={"y": 0.5},
            result_id="row-B",
        )
        spec = _spec()
        forward = _build_observation_identity(spec, [obs_a, obs_b])
        reverse = _build_observation_identity(spec, [obs_b, obs_a])
        assert forward == [reverse[1], reverse[0]]


# ---------------------------------------------------------------------------
# Round-trip contract through the full backend
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_replicates_remain_distinct_across_serialize_restore(self) -> None:
        """Restore → re-serialize keeps the discriminator-augmented identity.

        A v2 payload that includes the discriminator must survive a
        full round-trip without collapsing replicate slots. We assert
        the BayBE campaign carries one row per replicate and the
        identity index keeps as many distinct entries as observations.
        """
        backend = BayBEBackend()
        spec = _spec()
        obs = [
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5},
                objective_values={"y": 1.0},
                result_id="r-1",
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5},
                objective_values={"y": 1.0},
                result_id="r-2",
            ),
        ]
        first = backend.generate_suggestions(spec, obs, batch_size=1, iteration=1)
        assert first.backend_state is not None
        identity = first.backend_state["payload"]["observation_identity"]
        assert len(set(identity)) == 2

        second = backend.generate_suggestions(
            spec,
            obs,
            batch_size=1,
            iteration=2,
            backend_state=first.backend_state,
        )
        assert second.backend_state is not None
        identity2 = second.backend_state["payload"]["observation_identity"]
        assert identity2 == identity

        restored = Campaign.from_json(second.backend_state["payload"]["campaign_json"])
        assert len(restored.measurements) == 2


# ---------------------------------------------------------------------------
# v1 → v2 migration WARNING shape
# ---------------------------------------------------------------------------


class TestV1RebuildWarning:
    def test_v1_rebuild_emits_structured_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """v1 payload restore surfaces a WARNING with ``migration=v1_rebuild``.

        The audit flagged the legacy INFO-level message as unfilterable;
        dashboards need a stable structured field so cached-state
        rebuilds during a deploy show up in monitoring.
        """
        backend = BayBEBackend()
        spec = _spec()
        obs = [
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.7},
                objective_values={"y": 1.0},
                result_id="r-A",
            ),
        ]
        first = backend.generate_suggestions(spec, obs, batch_size=1, iteration=1)
        assert first.backend_state is not None
        legacy_state = {
            "backend": "baybe",
            "schema_version": 1,
            "payload": {
                "campaign_json": first.backend_state["payload"]["campaign_json"],
            },
        }
        with caplog.at_level(logging.WARNING, logger="bo_engine_baybe.backend"):
            _ = backend.generate_suggestions(
                spec,
                obs,
                batch_size=1,
                iteration=2,
                backend_state=legacy_state,
            )
        rebuild_records = [
            record
            for record in caplog.records
            if getattr(record, "migration", None) == "v1_rebuild"
        ]
        assert rebuild_records, (
            "Expected a WARNING with migration=v1_rebuild on v1 payload restore. "
            f"Saw: {[r.message for r in caplog.records]}"
        )
        record = rebuild_records[0]
        assert record.levelno == logging.WARNING
        # ``extra={...}`` attaches keys onto the LogRecord at emit time; the
        # type checker only knows the static LogRecord attributes, so we
        # read the structured fields off the record's ``__dict__``.
        assert record.__dict__["schema_version_seen"] == 1
        assert record.__dict__["schema_version_target"] == 2
        assert record.__dict__["n_observations"] == 1


# ---------------------------------------------------------------------------
# Reconciler picks the right delta when discriminator is present
# ---------------------------------------------------------------------------


class TestReconcilerWithDiscriminator:
    def test_new_replicate_is_detected_as_unseen(self) -> None:
        """A new replicate of an existing row adds a *second* measurement.

        With the discriminator threaded through, the second replicate's
        identity does not collide with the first, so the reconciler
        recognises it as a new row and adds it to the BayBE campaign.
        """
        backend = BayBEBackend()
        spec = _spec()
        obs_first = [
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5},
                objective_values={"y": 1.0},
                result_id="r-1",
            ),
        ]
        batch1 = backend.generate_suggestions(spec, obs_first, batch_size=1, iteration=1)
        assert batch1.backend_state is not None
        assert len(batch1.backend_state["payload"]["observation_identity"]) == 1

        obs_with_replicate = [
            *obs_first,
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5},
                objective_values={"y": 1.0},
                result_id="r-2",
            ),
        ]
        batch2 = backend.generate_suggestions(
            spec,
            obs_with_replicate,
            batch_size=1,
            iteration=2,
            backend_state=batch1.backend_state,
        )
        assert batch2.backend_state is not None
        identity = batch2.backend_state["payload"]["observation_identity"]
        assert len(set(identity)) == 2
        restored = Campaign.from_json(batch2.backend_state["payload"]["campaign_json"])
        assert len(restored.measurements) == 2

    def test_legacy_direct_engine_path_falls_back_to_counter(self) -> None:
        """Backwards-compat: identical rows without ids still both stay.

        The audit explicitly notes that direct ``bo-engine`` callers
        may not have an external ID. The multiset Counter pass keeps
        both rows in the campaign in that case.
        """
        backend = BayBEBackend()
        spec = _spec()
        obs = [
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.7},
                objective_values={"y": 1.0},
            ),
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.7},
                objective_values={"y": 1.0},
            ),
        ]
        first = backend.generate_suggestions(spec, obs, batch_size=1, iteration=1)
        assert first.backend_state is not None
        restored = Campaign.from_json(first.backend_state["payload"]["campaign_json"])
        assert len(restored.measurements) == 2

    def test_identity_payload_is_json_serializable(self) -> None:
        """Identity index round-trips through ``json.dumps`` for storage."""
        backend = BayBEBackend()
        spec = _spec()
        obs = [
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5},
                objective_values={"y": 0.7},
                result_id="r-only",
            ),
        ]
        batch = backend.generate_suggestions(spec, obs, batch_size=1, iteration=1)
        assert batch.backend_state is not None
        payload = batch.backend_state["payload"]["observation_identity"]
        # Round-trip through json to confirm the index is portable.
        decoded = json.loads(json.dumps(payload))
        assert decoded == payload
