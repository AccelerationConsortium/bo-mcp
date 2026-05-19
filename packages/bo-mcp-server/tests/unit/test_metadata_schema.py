"""``ResultMetadata`` schema enforcement (TODO 1.12).

Background: ``Result.metadata`` was previously a free-form ``dict[str,
Any]`` — agents could not introspect the expected keys, and a typo
silently survived to storage. The new :class:`ResultMetadata` schema
validates the keys at the intake boundary while leaving the in-memory
``Result.metadata`` typed as a dict for backward-compat with already-
persisted rows.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bo_mcp_server.domain import (
    ExternalRef,
    ResultMetadata,
    ResultSubmissionInput,
)


def test_external_ref_round_trip() -> None:
    ref = ExternalRef(system="lims", id="EXP-2024-0042", url="https://lab.example/EXP-2024-0042")
    assert ref.system == "lims"
    assert ref.id == "EXP-2024-0042"
    assert ref.url == "https://lab.example/EXP-2024-0042"


def test_external_ref_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ExternalRef(system="lims", id="x", surprise="boom")  # ty: ignore[unknown-argument]


def test_result_metadata_accepts_documented_keys() -> None:
    meta = ResultMetadata(
        external_ref=ExternalRef(system="lims", id="EXP-1"),
        conditions={"ambient_temp": 22.1, "operator": "WG"},
        cost=1.5,
        experiment_id="exp-1",
        operator="WG",
        batch_ref="batch-1",
        notes="ran with new reagent batch",
        source_row=3,
        source_file="results.csv",
    )
    dumped = meta.model_dump(exclude_unset=True)
    # Every key the caller set survives the round trip.
    assert dumped["cost"] == 1.5
    assert dumped["experiment_id"] == "exp-1"
    assert dumped["source_row"] == 3
    assert dumped["source_file"] == "results.csv"


def test_result_metadata_rejects_unknown_keys() -> None:
    """Typos like ``operater`` (instead of ``operator``) fail at intake."""
    with pytest.raises(ValidationError):
        ResultMetadata(operater="WG")  # ty: ignore[unknown-argument]


def test_result_metadata_rejects_negative_cost() -> None:
    with pytest.raises(ValidationError):
        ResultMetadata(cost=-1.0)


def test_result_submission_validates_metadata() -> None:
    """ResultSubmissionInput round-trips through ResultMetadata."""
    payload = ResultSubmissionInput(
        parameter_values={"x": 0.5},
        objective_values={"y": 1.0},
        metadata={"experiment_id": "exp-1", "cost": 2.0},
    )
    assert payload.metadata == {"experiment_id": "exp-1", "cost": 2.0}


def test_result_submission_rejects_unknown_metadata_key() -> None:
    """Unknown metadata key on the input surfaces as a validation error."""
    with pytest.raises(ValidationError):
        ResultSubmissionInput(
            parameter_values={"x": 0.5},
            objective_values={"y": 1.0},
            metadata={"experiment_id": "exp-1", "surprise": "boom"},
        )


def test_result_submission_empty_metadata_passes_through() -> None:
    payload = ResultSubmissionInput(
        parameter_values={"x": 0.5},
        objective_values={"y": 1.0},
    )
    assert payload.metadata == {}


def test_result_submission_round_trips_source_file_metadata() -> None:
    """REST file-upload route emits ``source_file`` — it must validate.

    Regression test for a Pydantic ``extra_forbidden`` 500 that fired
    on every REST CSV/XLSX upload before ``source_file`` was added to
    the documented metadata set.
    """
    payload = ResultSubmissionInput(
        parameter_values={"x": 0.5},
        objective_values={"y": 1.0},
        metadata={"source_file": "experiments_q1.csv", "source_row": 7},
    )
    assert payload.metadata == {"source_file": "experiments_q1.csv", "source_row": 7}


def test_result_metadata_rejects_empty_source_file() -> None:
    """An empty filename is a programming bug, not a valid provenance value."""
    with pytest.raises(ValidationError):
        ResultMetadata(source_file="")
