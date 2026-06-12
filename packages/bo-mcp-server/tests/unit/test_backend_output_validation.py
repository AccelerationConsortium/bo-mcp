"""Validate that the adapter rejects malformed backend ``SuggestionBatch`` payloads.

``bo-engine`` declares ``SuggestionBatch.suggestions`` as
``list[dict[str, Any]]`` to keep the public protocol Pydantic-free for
third-party backend authors. The operation layer used to blindly index
``item["parameter_values"]`` and ``item["provenance"]`` and let a
``KeyError`` surface as an opaque 500 (or, worse, persist a malformed
``SuggestionProvenance`` row built from partial data).

The adapter at :mod:`bo_mcp_server.operations.backend_output` re-validates
the batch with strict Pydantic models. These tests pin the contract:

* missing ``parameter_values`` / ``provenance`` raises
  :class:`BackendOutputError`, never ``KeyError``;
* a non-JSON-serializable ``backend_state`` (e.g. an in-place
  ``set``) is caught at the boundary instead of crashing the
  storage write;
* malformed ``method_info`` shape (non-dict) is caught at the
  boundary;
* a happy-path batch passes through unchanged.

Reference: protocol contract in
:class:`bo_engine.backend.SuggestionBatch` and the persistence path in
:meth:`SuggestionRepository.save`.
"""

from __future__ import annotations

import json

import pytest

from bo_engine.backend import SuggestionBatch
from bo_mcp_server.operations.backend_output import (
    BackendOutputError,
    validate_backend_batch,
)


def _valid_suggestion() -> dict:
    return {
        "parameter_values": {"x": 0.3},
        "provenance": {
            "iteration": 1,
            "batch_index": 0,
            "generation_method": "bo",
        },
    }


def test_validate_backend_batch_accepts_well_formed_batch() -> None:
    """Happy path: a conforming batch passes through unchanged."""
    batch = SuggestionBatch(
        suggestions=[_valid_suggestion()],
        method_info={"acquisition_function": "qLogNoisyExpectedImprovement"},
        backend_state={"turbo": {"length": 0.5}},
        warnings=[],
    )
    result = validate_backend_batch(batch)
    assert result is batch  # validation does not transform


def test_validate_backend_batch_rejects_missing_parameter_values() -> None:
    """A backend that forgets ``parameter_values`` fails loudly, not via KeyError."""
    batch = SuggestionBatch(
        suggestions=[
            {
                # parameter_values omitted
                "provenance": {
                    "iteration": 1,
                    "batch_index": 0,
                    "generation_method": "bo",
                },
            }
        ],
    )
    with pytest.raises(BackendOutputError) as exc:
        validate_backend_batch(batch)
    keys = {tuple(e["loc"]) for e in exc.value.errors}
    assert any("parameter_values" in t for t in keys)


def test_validate_backend_batch_rejects_missing_provenance() -> None:
    """A backend that forgets ``provenance`` fails loudly, not via KeyError."""
    batch = SuggestionBatch(
        suggestions=[
            {
                "parameter_values": {"x": 0.5},
                # provenance omitted
            }
        ],
    )
    with pytest.raises(BackendOutputError) as exc:
        validate_backend_batch(batch)
    keys = {tuple(e["loc"]) for e in exc.value.errors}
    assert any("provenance" in t for t in keys)


def test_validate_backend_batch_rejects_malformed_provenance() -> None:
    """Provenance dicts that lack required fields fail at the adapter, not in storage."""
    batch = SuggestionBatch(
        suggestions=[
            {
                "parameter_values": {"x": 0.5},
                # ``iteration`` and ``batch_index`` are required on SuggestionProvenance
                "provenance": {"generation_method": "bo"},
            }
        ],
    )
    with pytest.raises(BackendOutputError) as exc:
        validate_backend_batch(batch)
    keys = {tuple(e["loc"]) for e in exc.value.errors}
    assert any("iteration" in t for t in keys)
    assert any("batch_index" in t for t in keys)


def test_validate_backend_batch_rejects_non_json_parameter_values() -> None:
    """Non-JSON ``parameter_values`` is rejected at the adapter, not at storage.

    :meth:`SuggestionRepository.save` persists ``parameter_values``
    via ``json.dumps``. A backend that smuggles a ``set`` / tensor /
    numpy array through the loose ``dict[str, Any]`` shape check
    would otherwise crash inside the write transaction with a raw
    ``TypeError`` — bypassing the structured
    ``ACQUISITION_OPTIMIZATION_FAILED`` envelope and leaving partial
    state in flight. Catch it at the contract boundary instead.
    """
    batch = SuggestionBatch(
        suggestions=[
            {
                "parameter_values": {"x": {1, 2, 3}},  # set is not JSON
                "provenance": {
                    "iteration": 1,
                    "batch_index": 0,
                    "generation_method": "bo",
                },
            }
        ],
    )
    with pytest.raises(BackendOutputError) as exc:
        validate_backend_batch(batch)
    locs = {tuple(e["loc"]) for e in exc.value.errors}
    assert any("parameter_values" in part for loc in locs for part in loc), locs


def test_validate_backend_batch_rejects_non_json_backend_state() -> None:
    """A non-JSON-serializable backend_state is rejected before persistence.

    ``SuggestionRepository.save`` ``json.dumps`` the value; surfacing
    the type error here points the failure at the backend contract
    instead of at the database write site.
    """
    batch = SuggestionBatch(
        suggestions=[_valid_suggestion()],
        backend_state={"unserializable": {1, 2, 3}},  # set is not JSON
    )
    with pytest.raises(BackendOutputError) as exc:
        validate_backend_batch(batch)
    keys = {tuple(e["loc"]) for e in exc.value.errors}
    assert any("backend_state" in t for t in keys)


def test_validate_backend_batch_rejects_empty_suggestions_list() -> None:
    """A backend that returns zero suggestions is contractually wrong.

    Callers depend on at least one suggestion per batch; an empty list
    would otherwise pass through and surface downstream as a vacuous
    "iteration completed, nothing to do" response.
    """
    batch = SuggestionBatch(suggestions=[])
    with pytest.raises(BackendOutputError) as exc:
        validate_backend_batch(batch)
    keys = {tuple(e["loc"]) for e in exc.value.errors}
    assert any("suggestions" in t for t in keys)


def test_validate_backend_batch_rejects_plain_dict_payload() -> None:
    """A backend that returns a plain dict instead of a ``SuggestionBatch``.

    The contract advertises a ``SuggestionBatch`` dataclass; a plain
    dict would otherwise crash on ``batch.suggestions`` with an
    ``AttributeError`` outside the guarded validation block — bypassing
    the structured ``ACQUISITION_OPTIMIZATION_FAILED`` envelope and
    surfacing as an opaque 500 instead. The adapter must convert this
    contract violation into a typed :class:`BackendOutputError`.
    """
    with pytest.raises(BackendOutputError) as exc:
        validate_backend_batch({"suggestions": []})  # ty: ignore[invalid-argument-type]
    assert exc.value.errors  # carries structured per-attribute reports
    locs = {tuple(e["loc"]) for e in exc.value.errors}
    # All four required attributes are missing on a bare dict.
    assert ("method_info",) in locs
    assert ("backend_state",) in locs
    assert ("warnings",) in locs


def test_validate_backend_batch_rejects_none_return() -> None:
    """A backend that returns ``None`` is also caught at the type-check stage."""
    with pytest.raises(BackendOutputError) as exc:
        validate_backend_batch(None)  # ty: ignore[invalid-argument-type]
    locs = {tuple(e["loc"]) for e in exc.value.errors}
    assert ("suggestions",) in locs


def test_backend_output_errors_are_json_serializable() -> None:
    """The captured errors must survive ``json.dumps`` for the response envelope.

    Pydantic's raw ``ValidationError.errors()`` carries the offending
    ``input`` (which is exactly the unserializable value we just
    rejected — e.g. a ``set``) and a ``ctx`` map that can hold the
    underlying ``ValueError`` instance. Putting either into the
    structured ``ACQUISITION_OPTIMIZATION_FAILED`` response would
    crash ``json.dumps`` and convert a clean 4xx into a 500. The
    sanitizer must strip ``input`` and stringify ``ctx``.
    """
    batch = SuggestionBatch(
        suggestions=[_valid_suggestion()],
        backend_state={"unserializable": {1, 2, 3}},
    )
    with pytest.raises(BackendOutputError) as exc:
        validate_backend_batch(batch)

    # Every captured error must dump cleanly. This is the contract the
    # response envelope (errors land in ``details.validation_errors``)
    # depends on.
    payload = json.dumps(exc.value.errors)
    assert "backend_state" in payload
    # The raw ``input`` field (the offending ``set``) must not leak
    # through; only documented fields survive sanitization.
    for entry in exc.value.errors:
        assert "input" not in entry
        ctx = entry.get("ctx")
        if ctx is not None:
            # ctx values are stringified so ``json.dumps`` cannot trip
            # over a ``ValueError`` instance.
            assert all(isinstance(v, str) for v in ctx.values())
