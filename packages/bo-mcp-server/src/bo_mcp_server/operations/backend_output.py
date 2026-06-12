"""Adapter-layer validation for :class:`bo_engine.backend.SuggestionBatch`.

``bo-engine`` keeps the public :class:`SuggestionBatch` Pydantic-free
so third-party backends can implement the protocol without taking
:mod:`pydantic` as a hard dependency. That means the server can be
handed a batch whose ``suggestions`` list is shaped any-which-way —
the operation used to blindly index ``item["parameter_values"]`` and
``item["provenance"]`` and surface a ``KeyError`` (or persist malformed
provenance) when a misbehaving backend returned a partial dict.

This module re-validates the batch with strict Pydantic models at the
operation layer. The engine stays untouched; the validation is a thin
adapter at the boundary where the engine output enters the server.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from bo_engine.backend import SuggestionBatch
from bo_mcp_server.domain import SuggestionProvenance

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class BackendOutputError(ValueError):
    """Raised when a backend's :class:`SuggestionBatch` fails adapter validation.

    Carries the underlying Pydantic ``ValidationError`` for structured
    error reporting (the operation translates this to an
    ``ACQUISITION_OPTIMIZATION_FAILED`` envelope so retries are clearly
    not the answer — the backend itself is malformed).
    """

    def __init__(
        self,
        message: str,
        errors: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Record the human-readable message and structured per-field errors."""
        super().__init__(message)
        self.errors: list[dict[str, Any]] = list(errors) if errors else []


class _ValidatedSuggestion(BaseModel):
    """Single suggestion entry returned by a backend.

    Mirrors the ``{"parameter_values": ..., "provenance": ...}`` shape
    produced by :class:`bo_engine.botorch_backend.BoTorchBackend` and
    :class:`bo_engine_baybe.backend.BayBEBackend`. ``provenance`` is
    validated against the domain :class:`SuggestionProvenance` so the
    persisted row carries the documented fields and types.

    ``parameter_values`` is typed as ``dict[str, Any]`` because the
    semantic value-type set varies by parameter (continuous floats,
    discrete ints, categorical strings, one-hot vectors). The
    serializability check below catches backends that smuggle through
    a non-JSON value (``torch.Tensor``, ``numpy.ndarray``, ``set``,
    custom objects) — those would otherwise pass the loose dict check
    and crash later in :meth:`SuggestionRepository.save` when the
    persisted row hits ``json.dumps``.
    """

    model_config = ConfigDict(extra="forbid")

    parameter_values: dict[str, Any] = Field(..., min_length=1)
    provenance: SuggestionProvenance

    @field_validator("parameter_values")
    @classmethod
    def _parameter_values_are_json_serializable(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Refuse a parameter_values dict that the storage layer cannot persist."""
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            msg = f"parameter_values is not JSON-serializable: {e}"
            raise ValueError(msg) from e
        return value


class _ValidatedSuggestionBatch(BaseModel):
    """Whole batch as returned by a backend.

    Keeps ``method_info``/``backend_state``/``warnings`` typed as the
    permissive shapes the protocol advertises (``dict[str, Any]`` /
    ``dict[str, Any] | None`` / ``list[str]``). The strict part is the
    ``suggestions`` list — that is the surface every persister and
    response formatter downstream indexes into.
    """

    model_config = ConfigDict(extra="forbid")

    suggestions: list[_ValidatedSuggestion] = Field(..., min_length=1)
    method_info: dict[str, Any] = Field(default_factory=dict)
    backend_state: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("backend_state")
    @classmethod
    def _backend_state_is_json_serializable(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Refuse a backend_state that the storage layer cannot persist.

        The server persists ``backend_state`` via ``json.dumps`` /
        ``json.loads``. A backend that returns a torch tensor or numpy
        array passes the dict shape check but crashes at persist time;
        catching it here surfaces the contract violation against the
        backend rather than against the database write.
        """
        if value is None:
            return value
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            msg = f"backend_state is not JSON-serializable: {e}"
            raise ValueError(msg) from e
        return value


_REQUIRED_BATCH_ATTRS: tuple[str, ...] = (
    "suggestions",
    "method_info",
    "backend_state",
    "warnings",
)


def validate_backend_batch(batch: SuggestionBatch) -> SuggestionBatch:
    """Validate a backend-returned :class:`SuggestionBatch` at the adapter boundary.

    Returns the original ``batch`` unchanged when validation passes —
    the validator is a contract check, not a transformation. Raises
    :class:`BackendOutputError` (subclass of :class:`ValueError`) on
    any shape mismatch so the operation layer can convert to a
    structured error envelope instead of letting a downstream
    ``KeyError`` / ``AttributeError`` surface as an opaque 500.

    The first check is a *type* check: a third-party backend that
    returns a plain dict, ``None``, or anything else that lacks the
    documented attribute surface would otherwise crash on the bare
    attribute reads below. Catch that here so the operation layer
    receives a single typed failure mode for any contract violation.
    """
    missing = [attr for attr in _REQUIRED_BATCH_ATTRS if not hasattr(batch, attr)]
    if missing:
        msg = (
            f"Backend returned an object of type {type(batch).__name__!r} "
            "that does not match the SuggestionBatch contract; expected "
            "an instance with attributes "
            f"{list(_REQUIRED_BATCH_ATTRS)}, missing {missing}."
        )
        raise BackendOutputError(
            msg,
            errors=[
                {
                    "type": "missing_attribute",
                    "loc": [attr],
                    "msg": f"backend output is missing required attribute '{attr}'",
                }
                for attr in missing
            ],
        )

    payload = {
        "suggestions": batch.suggestions,
        "method_info": batch.method_info,
        "backend_state": batch.backend_state,
        "warnings": batch.warnings,
    }
    try:
        _ValidatedSuggestionBatch.model_validate(payload)
    except ValidationError as e:
        msg = (
            "Backend returned a SuggestionBatch that does not conform "
            "to the documented shape; this is a bug in the backend "
            "implementation, not in the request."
        )
        raise BackendOutputError(msg, errors=_sanitize_errors(e.errors())) from e
    return batch


def _sanitize_errors(errors: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project Pydantic error dicts into a JSON-safe shape.

    Pydantic's :meth:`ValidationError.errors` returns dicts that can
    carry the original ``input`` (which may itself be non-JSON, e.g.
    a ``set`` or a tensor) and a ``ctx`` map that may contain the
    underlying ``ValueError`` instance. Putting either into the
    response envelope risks crashing ``json.dumps`` at serialization
    time — which would turn the structured error we just built into a
    500. Keep only the fields agents actually use (``type``, ``loc``,
    ``msg``, ``url``) and stringify everything in ``ctx``.
    """
    sanitized: list[dict[str, Any]] = []
    for err in errors:
        entry: dict[str, Any] = {
            "type": err.get("type"),
            "loc": [str(part) for part in err.get("loc", ())],
            "msg": err.get("msg"),
        }
        if "url" in err:
            entry["url"] = err["url"]
        ctx = err.get("ctx")
        if isinstance(ctx, dict) and ctx:
            entry["ctx"] = {k: str(v) for k, v in ctx.items()}
        sanitized.append(entry)
    return sanitized
