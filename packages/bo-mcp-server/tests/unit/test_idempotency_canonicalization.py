"""Non-semantic field stripping for idempotency hashing.

Clients regenerate transport / telemetry fields on every retry —
``request_id``, ``trace_id``, ``created_at``. Hashing those would
turn the idempotency-key contract into a footgun: the same logical
call would hash differently on retry, surface as
:class:`ErrorCode.IDEMPOTENCY_CONFLICT`, and the user would see the
exact symptom the mechanism is supposed to prevent.

This suite pins the canonical set of stripped fields and exercises
the end-to-end replay path through :func:`apply_idempotency`.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.idempotency import (
    NON_SEMANTIC_PAYLOAD_KEYS,
    apply_idempotency,
    canonical_request_hash,
    strip_non_semantic_fields,
)

pytestmark = pytest.mark.usefixtures("setup_database")


def test_non_semantic_payload_keys_documented_set() -> None:
    """The stripped set is a fixed allowlist — pin it so changes are reviewed.

    Widening this set silently could turn two distinct requests into a
    hash collision; narrowing it could re-introduce the spurious-
    conflict bug. Treat the set as a public contract.
    """
    assert "created_at" in NON_SEMANTIC_PAYLOAD_KEYS
    assert "request_id" in NON_SEMANTIC_PAYLOAD_KEYS
    assert "trace_id" in NON_SEMANTIC_PAYLOAD_KEYS
    assert "idempotency_key" in NON_SEMANTIC_PAYLOAD_KEYS
    # A non-stripped key from a real payload — guards against accidentally
    # stripping a semantic field.
    assert "campaign_id" not in NON_SEMANTIC_PAYLOAD_KEYS


def test_strip_drops_non_semantic_keys() -> None:
    """Top-level transport keys are removed; semantic keys survive."""
    payload = {
        "campaign_id": "c-1",
        "owner_id": "u-1",
        "request_id": "req-aaa",
        "created_at": "2026-05-18T10:00:00Z",
        "trace_id": "trace-1",
    }
    stripped = strip_non_semantic_fields(payload)
    assert stripped == {"campaign_id": "c-1", "owner_id": "u-1"}


def test_strip_only_at_top_level_preserves_nested_metadata() -> None:
    """Nested ``_metadata`` is treated as semantic data; only envelope keys are dropped.

    Top-level-only stripping: the
    domain layer permits ``parameter_values["_metadata"]`` as a
    user-defined key (``min_length=1`` on parameter names plus
    free-form ``parameter_values``). A recursive ``_metadata`` strip
    would silently collapse a legitimate
    ``parameter_values = {"_metadata": 5, ...}`` vs
    ``{"_metadata": 10, ...}`` across distinct submissions onto the
    same hash bucket. The contract is 'strip only what the transport
    put at the top of the payload'.
    """
    payload = {
        "intake_data": {
            "parameters": [
                {"name": "x", "trace_id": "trace-row", "value": 1},
            ],
            "_metadata": {"request_id": "req-bbb"},
            "objectives": [{"name": "y"}],
        },
        "created_at": "2026-05-18T10:00:00Z",
    }
    stripped = strip_non_semantic_fields(payload)
    assert stripped == {
        "intake_data": {
            "parameters": [{"name": "x", "trace_id": "trace-row", "value": 1}],
            # Nested ``_metadata`` survives — only the top-level
            # ``created_at`` is removed by the envelope-only strip.
            "_metadata": {"request_id": "req-bbb"},
            "objectives": [{"name": "y"}],
        }
    }


def test_strip_preserves_parameter_named_metadata() -> None:
    """A campaign with ``parameter_values["_metadata"]`` is not silently collapsed.

    Regression for the recursive ``_metadata`` bug: the original Phase G implementation removed
    ``_metadata`` at every level under the assumption it was always
    transport-owned. But ``Parameter.name`` only requires
    ``min_length=1`` and ``ResultSubmissionInput.parameter_values``
    accepts arbitrary keys, so a real campaign can legitimately use
    that label. Distinct measurements differing only on the
    ``_metadata`` parameter must produce distinct idempotency hashes
    so the cache cannot replay one as the other.
    """
    a = {
        "campaign_id": "c-1",
        "results": [
            {
                "parameter_values": {"_metadata": 5, "temperature": 300},
                "objective_values": {"yield": 0.8},
            },
        ],
    }
    b = {
        "campaign_id": "c-1",
        "results": [
            {
                "parameter_values": {"_metadata": 10, "temperature": 300},
                "objective_values": {"yield": 0.8},
            },
        ],
    }
    assert canonical_request_hash(a) != canonical_request_hash(b)


def test_strip_preserves_semantic_keys_colliding_with_allowlist() -> None:
    """A user-defined parameter named ``timestamp`` survives the strip.

    Regression guard for the recursive-strip bug: when the canonical
    payload was walked deeply by key name, a real ``parameter_values =
    {"temperature": 300, "timestamp": 12345}`` lost its ``timestamp``
    field at hash time and two genuinely-distinct submissions
    (different ``timestamp`` values, same ``temperature``) collapsed
    into one cache bucket. The path-aware strip preserves nested
    fields verbatim so this cannot happen.
    """
    a = {
        "campaign_id": "c-1",
        "results": [
            {
                "parameter_values": {"temperature": 300, "timestamp": 11111},
                "objective_values": {"yield": 0.8},
            },
        ],
        "created_at": "telemetry-ignored",
    }
    b = {
        "campaign_id": "c-1",
        "results": [
            {
                "parameter_values": {"temperature": 300, "timestamp": 99999},
                "objective_values": {"yield": 0.8},
            },
        ],
        "created_at": "telemetry-ignored",
    }
    assert canonical_request_hash(a) != canonical_request_hash(b)


def test_strip_preserves_semantic_objective_named_created_at() -> None:
    """An objective literally named ``created_at`` is not collapsed.

    Mirrors the parameter-name regression on the objective side —
    different ``objective_values["created_at"]`` measurements must
    produce different hashes.
    """
    a = {
        "campaign_id": "c-1",
        "objectives": {"yield": 0.5, "created_at": 12345.0},
    }
    b = {
        "campaign_id": "c-1",
        "objectives": {"yield": 0.5, "created_at": 99999.0},
    }
    assert canonical_request_hash(a) != canonical_request_hash(b)


def test_hash_invariant_under_created_at_drift() -> None:
    """Two calls differing only in ``created_at`` hash equal."""
    base = {"campaign_id": "c-1", "owner_id": "u-1"}
    first = {**base, "created_at": "2026-05-18T10:00:00Z"}
    second = {**base, "created_at": "2026-05-18T10:00:05Z"}
    assert canonical_request_hash(first) == canonical_request_hash(second)


def test_hash_invariant_under_request_id_and_trace_id_drift() -> None:
    """Per-request transport ids must not perturb the canonical hash."""
    base = {"results": [{"value": 1}, {"value": 2}]}
    first = {**base, "request_id": "req-1", "trace_id": "trace-1"}
    second = {**base, "request_id": "req-2", "trace_id": "trace-2"}
    assert canonical_request_hash(first) == canonical_request_hash(second)


def test_hash_still_distinguishes_real_payload_differences() -> None:
    """Stripping must not collapse genuinely different payloads.

    Regression guard: a buggy strip that walked too far could erase
    list ordering or nested values, causing distinct submissions to
    silently share a hash and replay each other's responses.
    """
    a = {"results": [{"value": 1}], "created_at": "t1"}
    b = {"results": [{"value": 2}], "created_at": "t1"}
    assert canonical_request_hash(a) != canonical_request_hash(b)


@pytest.mark.asyncio
async def test_replay_works_when_only_created_at_differs() -> None:
    """End-to-end: two retries with drifting telemetry replay the cached response.

    This is the user-visible payoff: a retry path that
    re-stamps ``created_at`` no longer surfaces ``IDEMPOTENCY_CONFLICT``
    against the original call's hash.
    """
    n_calls = 0

    async def run(_session: AsyncSession) -> dict[str, Any]:
        nonlocal n_calls
        n_calls += 1
        return {"success": True, "n": n_calls}

    first = await apply_idempotency(
        tool_name="canon_tool",
        idempotency_key="canon-key",
        request_payload={
            "campaign_id": "c-1",
            "owner_id": "u-1",
            "created_at": "2026-05-18T10:00:00Z",
            "request_id": "req-original",
        },
        executor=run,
    )
    replay = await apply_idempotency(
        tool_name="canon_tool",
        idempotency_key="canon-key",
        request_payload={
            "campaign_id": "c-1",
            "owner_id": "u-1",
            "created_at": "2026-05-18T10:00:42Z",
            "request_id": "req-retry",
        },
        executor=run,
    )
    assert n_calls == 1, "the retry must not re-execute the operation"
    assert first["n"] == 1
    assert replay["n"] == 1
    assert replay["idempotency_replay"] is True
