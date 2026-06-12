"""BayBE Campaign construction, serialization, and measurement reconciliation.

Split from :mod:`bo_engine_baybe.backend` to give the campaign
state-envelope helpers and the stable-identity measurement reconciler
their own module. The :class:`~bo_engine_baybe.backend.BayBEBackend`
class composes the helpers below to restore campaigns from persisted
state, add new measurements by stable identity, and write
the next state envelope back to storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from typing import Any

from baybe import Campaign
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpaceType

from bo_engine.types import ObservationData, OptimizationSpec
from bo_engine_baybe.converters import (
    observations_to_dataframe,
    spec_to_acquisition_function,
    spec_to_objective,
    spec_to_searchspace,
)
from bo_engine_baybe.options import extract_baybe_backend_options

logger = logging.getLogger(__name__)


# Random → BO switch point used when neither the BayBE-native
# ``backend_options['baybe'].recommender.switch_after`` nor the neutral
# ``spec.initial_design_size`` is set: BayBE moves to the GP-based
# recommender after the first measurement.
_DEFAULT_SWITCH_AFTER = 1


# Backend-state schema versions. v1 = bare {campaign_json}; v2 adds the
# stable observation identity index.
_STATE_SCHEMA_VERSION_LEGACY = 1
_STATE_SCHEMA_VERSION_IDENTITY = 2


# BayBE-specific exceptions that must be caught alongside standard ones.
# IncompatibilityError is raised when the recommender phase (random vs BO)
# doesn't provide the requested API (e.g. posterior_stats during random phase).
_BAYBE_SAFE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
)

try:
    from baybe.exceptions import IncompatibilityError

    _BAYBE_SAFE_EXCEPTIONS = (*_BAYBE_SAFE_EXCEPTIONS, IncompatibilityError)
except ImportError:
    pass  # Older BayBE versions may not expose this


def _resolve_switch_after(spec: OptimizationSpec) -> int:
    """Resolve the random → GP switch point for the TwoPhaseMetaRecommender.

    Precedence: the explicit BayBE-native
    ``backend_options['baybe'].recommender.switch_after`` wins over the
    neutral ``spec.initial_design_size``, which in turn wins over the
    legacy default of switching after the first measurement. Bridging the
    neutral knob keeps ``backend="auto"`` comparisons like-for-like — a
    requested warmup of N random points means N on both backends.
    """
    options = extract_baybe_backend_options(spec.backend_options)
    if options.recommender:
        return options.recommender.switch_after
    if spec.initial_design_size:
        return spec.initial_design_size
    return _DEFAULT_SWITCH_AFTER


def _build_campaign(spec: OptimizationSpec) -> Campaign:
    """Create a fresh BayBE Campaign from an OptimizationSpec.

    The BO-phase recommender honors ``spec.acquisition_method`` via
    :func:`spec_to_acquisition_function`; ``None`` (AUTO or a method BayBE
    cannot express) keeps BayBE's own default acquisition function.
    """
    searchspace = spec_to_searchspace(spec)
    is_purely_discrete = searchspace.type == SearchSpaceType.DISCRETE
    options = extract_baybe_backend_options(spec.backend_options)

    kwargs: dict[str, object] = {
        "searchspace": searchspace,
        "objective": spec_to_objective(spec),
        "recommender": TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                acquisition_function=spec_to_acquisition_function(spec),
            ),
            switch_after=_resolve_switch_after(spec),
        ),
    }

    if is_purely_discrete:
        kwargs["allow_recommending_already_measured"] = False
        kwargs["allow_recommending_already_recommended"] = False
        kwargs["allow_recommending_pending_experiments"] = (
            options.allow_recommending_pending_experiments
        )

    return Campaign(**kwargs)  # ty: ignore[invalid-argument-type]


def _restore_or_build_campaign(
    spec: OptimizationSpec,
    backend_state: dict[str, Any] | None,
) -> Campaign:
    """Restore a Campaign from serialized state, or build a fresh one."""
    if backend_state and "campaign_json" in backend_state:
        try:
            return Campaign.from_json(backend_state["campaign_json"])
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            # WARNING, not DEBUG: a failed restore drops the campaign's
            # entire measurement history and the identity reconciliation
            # must repopulate it from storage. Operators need a
            # dashboard-filterable signal when this happens (e.g. a BayBE
            # upgrade changing the serialization schema).
            logger.warning(
                "Failed to restore BayBE campaign from stored state; rebuilding fresh: %s",
                e,
                extra={
                    "outcome": "campaign_restore_failed",
                    "error_class": type(e).__name__,
                },
            )
    return _build_campaign(spec)


def _observation_fingerprint(
    obs: ObservationData,
    param_names: list[str],
    obj_names: list[str],
) -> str:
    """Compute a stable identity hash for an observation.

    Uses sorted parameter and objective columns so reordering the input
    list (or shuffling the underlying DB query) produces the same hash.
    When the caller threads a durable cross-system ID through
    ``ObservationData.result_id`` it is folded into the
    payload as the per-row discriminator so otherwise-identical
    replicate rows produce *distinct* identities and can be addressed
    individually. The hash is intentionally short (16 hex chars) —
    enough to make collisions astronomically unlikely for any realistic
    campaign and cheap to compare during reconciliation.
    """
    payload: dict[str, Any] = {
        "params": [(name, obs.parameter_values.get(name)) for name in param_names],
        "objectives": [(name, obs.objective_values.get(name)) for name in obj_names],
    }
    if obs.result_id is not None:
        payload["result_id"] = obs.result_id
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()


def _build_observation_identity(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> list[str]:
    """Return one fingerprint per observation in input order."""
    param_names = [p.name for p in spec.parameters]
    obj_names = [o.name for o in spec.objectives]
    return [_observation_fingerprint(o, param_names, obj_names) for o in observations]


def _serialize_campaign(
    campaign: Campaign,
    observation_identity: list[str],
) -> dict[str, Any]:
    """Serialize a Campaign and identity index for storage as backend_state.

    The identity index is the list of fingerprints for the measurements
    BayBE believes it has. On restore, the next call compares its incoming
    observation fingerprints to this list and adds only truly unseen rows
    Schema version bumps to ``_STATE_SCHEMA_VERSION_IDENTITY``.
    """
    return {
        "schema_version": _STATE_SCHEMA_VERSION_IDENTITY,
        "campaign_json": campaign.to_json(),
        "observation_identity": list(observation_identity),
    }


def _reconcile_measurements(
    campaign: Campaign,
    spec: OptimizationSpec,
    observations: list[ObservationData],
    backend_state: dict[str, Any] | None,
) -> tuple[Campaign, list[str]]:
    """Add new measurements to a restored campaign via stable identity.

    Returns ``(campaign, identity_index)`` — the campaign may be a freshly
    built replacement when the stored identity index references rows that
    have since been deleted from storage. Reconciliation rules:

    * Compute a fingerprint for each observation in the current
      ``observations`` list. When ``ObservationData.result_id`` is set,
      it is folded into the fingerprint so otherwise-identical
      replicate rows produce distinct identities.
    * Fingerprints are reconciled as a **multiset**, not a set, so two
      observations with identical parameter/objective values that also
      lack a ``result_id`` (the legacy direct-engine path) both stay in
      the campaign via the multiset Counter pass.
    * Stored identities that no longer appear in storage trigger a
      rebuild — the user deleted or rewrote rows, so the BayBE campaign
      must not keep training on data the server no longer owns.
    * A restored campaign whose measurement count disagrees with the
      stored identity index also triggers a rebuild. The typical cause
      is a failed ``Campaign.from_json`` restore that fell back to an
      empty campaign while the index still lists every prior
      observation — consuming the index against the empty campaign
      would add nothing, persist the empty campaign together with the
      full index, and silently degrade the campaign to random sampling
      on every subsequent call.
    * Storage emptying out (``observations=[]``) is also a rebuild
      trigger: the restored campaign has measurements the source of
      truth no longer has.
    * Legacy payloads (no identity index) cannot be reconciled safely —
      the previous count-prefix logic was the bug 1.63 set out to fix.
      The migration path is therefore to rebuild from current
      observations and start tracking identities from this call onwards.
      The rebuild emits a structured WARNING with ``migration=v1_rebuild``
      so dashboards alarm on cached v1 state being reset on deploy.
    """
    incoming_ids = _build_observation_identity(spec, observations)
    incoming_counts: Counter[str] = Counter(incoming_ids)
    stored_ids = list(_extract_stored_identity(backend_state))
    stored_counts: Counter[str] = Counter(stored_ids)
    has_identity_field = _has_stored_identity_field(backend_state)

    if not has_identity_field and _campaign_has_measurements(campaign):
        # Promoted from INFO to WARNING with structured fields
        # so operators running on cached v1 payloads during a deploy that
        # drops v1 support get a dashboard-filterable signal — the state
        # reset would otherwise be invisible to monitoring.
        logger.warning(
            "BayBE state migration: v1 payload rebuild",
            extra={
                "migration": "v1_rebuild",
                "schema_version_seen": _STATE_SCHEMA_VERSION_LEGACY,
                "schema_version_target": _STATE_SCHEMA_VERSION_IDENTITY,
                "n_observations": len(observations),
            },
        )
        return _rebuild_from_observations(spec, observations), incoming_ids

    # The identity index is only trustworthy when the restored campaign
    # actually carries the measurements the index records. A disagreement
    # means the restore fell back to a fresh campaign (or the payload was
    # tampered with) — consuming the index here would skip every incoming
    # observation and leave the campaign permanently empty.
    restored_count = _campaign_measurement_count(campaign)
    if has_identity_field and restored_count != len(stored_ids):
        logger.warning(
            "Restored BayBE campaign carries %s measurement(s) but the stored "
            "identity index records %d; rebuilding from current observations.",
            "an unreadable number of" if restored_count is None else restored_count,
            len(stored_ids),
        )
        return _rebuild_from_observations(spec, observations), incoming_ids

    # A stored multiplicity that exceeds the incoming one means a measurement
    # the campaign believes it has is no longer in storage — rebuild.
    if any(stored_counts[k] > incoming_counts[k] for k in stored_counts):
        missing = sum(max(stored_counts[k] - incoming_counts[k], 0) for k in stored_counts)
        logger.warning(
            "Restored BayBE campaign references %d measurement(s) no longer present "
            "in storage; rebuilding from current observations.",
            missing,
        )
        return _rebuild_from_observations(spec, observations), incoming_ids

    # Add the multiplicity delta per fingerprint, preserving observation order
    # so each replicate produces a distinct BayBE row.
    remaining: Counter[str] = stored_counts.copy()
    new_observations: list[ObservationData] = []
    for fingerprint, obs in zip(incoming_ids, observations, strict=True):
        if remaining[fingerprint] > 0:
            remaining[fingerprint] -= 1
            continue
        new_observations.append(obs)

    if new_observations:
        obs_df = observations_to_dataframe(new_observations, spec)
        campaign.add_measurements(obs_df)

    return campaign, incoming_ids


def _has_stored_identity_field(backend_state: dict[str, Any] | None) -> bool:
    """Return True when the payload includes the v2 ``observation_identity`` key."""
    if not backend_state:
        return False
    return "observation_identity" in backend_state


def _campaign_measurement_count(campaign: Campaign) -> int | None:
    """Measurement count of a restored campaign, or ``None`` when unreadable.

    ``None`` deliberately compares unequal to every valid index length so
    an unreadable campaign is treated as desynchronized rather than intact.
    """
    try:
        return len(campaign.measurements)
    except (AttributeError, TypeError):
        return None


def _campaign_has_measurements(campaign: Campaign) -> bool:
    """Defensive check for whether the restored campaign carries any data."""
    return bool(_campaign_measurement_count(campaign))


def _rebuild_from_observations(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> Campaign:
    """Build a fresh campaign and add all current observations.

    Used as the safe fallback whenever identity reconciliation can no
    longer trust the restored campaign (legacy payload, missing rows).
    """
    fresh = _build_campaign(spec)
    if observations:
        obs_df = observations_to_dataframe(observations, spec)
        fresh.add_measurements(obs_df)
    return fresh


def _extract_stored_identity(backend_state: dict[str, Any] | None) -> list[str]:
    """Pull the identity index out of a stored payload.

    Supports both the new schema (``observation_identity`` field, v2) and
    the legacy schema (no identity field, v1). For v1 payloads the
    reconciliation falls back to ``len(campaign.measurements)`` interpretation
    via an empty identity list (treated as "no prior identities recorded").
    """
    if not backend_state:
        return []
    raw = backend_state.get("observation_identity")
    if not isinstance(raw, list):
        return []
    return [str(v) for v in raw]
