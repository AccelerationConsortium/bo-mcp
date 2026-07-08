"""Transport-neutral idempotency wrapper for mutating operations.

The MCP tool layer and the REST API both need the same "run an
operation at most once per ``(operation_name, idempotency_key)``"
semantics so retries are safe across either transport. Keeping the
wrapper in :mod:`bo_mcp_server.operations` (rather than the MCP-
specific tool modules) lets the REST routes inherit it via the
:mod:`bo_mcp_server.client` facade without depending on FastMCP
internals.

This module is a thin layer over :func:`apply_idempotency`; it
exists so REST handlers can adopt idempotency without re-importing
the MCP server's idempotency module and so a future replacement of
the cache backend only has to touch one re-export.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.domain import CampaignIntakeInput, ResultSubmissionInput
from bo_mcp_server.idempotency import apply_idempotency

OperationExecutor = Callable[[AsyncSession], Awaitable[dict[str, Any]]]
"""Same executor signature as :data:`bo_mcp_server.idempotency.ToolExecutor`.

Re-exported here so transport layers can import a single name from
the operations package instead of reaching into the MCP-specific
``idempotency`` module.
"""


async def run_idempotent_operation(
    operation_name: str,
    idempotency_key: str | None,
    request_payload: dict[str, Any],
    executor: OperationExecutor,
    *,
    ttl_seconds: int | None = None,
    reservation_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    """Run ``executor`` at most once per ``(operation_name, idempotency_key)``.

    Identical contract to :func:`apply_idempotency`: ``None`` key
    always executes (no caching); a matching key + payload replays
    the cached response with ``idempotency_replay: True``; a key
    reused with a different payload returns the structured
    :class:`~bo_mcp_server.errors.ErrorCode.IDEMPOTENCY_CONFLICT`
    envelope; an in-flight retry returns
    :class:`~bo_mcp_server.errors.ErrorCode.IDEMPOTENCY_IN_PROGRESS`.

    The ``operation_name`` is used as the cache-key namespace: REST
    and MCP that hit the *same* logical operation should pass the
    same ``operation_name`` (e.g. ``"create_campaign"``) so a retry
    from either transport replays the cached response from the
    other. The MCP tool wrappers historically passed
    ``"bo_create_campaign"`` etc., so REST routes that want to share
    the cache should pass the matching prefix.
    """
    return await apply_idempotency(
        tool_name=operation_name,
        idempotency_key=idempotency_key,
        request_payload=request_payload,
        executor=executor,
        ttl_seconds=ttl_seconds,
        reservation_ttl_seconds=reservation_ttl_seconds,
    )


# ---------------------------------------------------------------------------
# Canonical request-payload builders
# ---------------------------------------------------------------------------
#
# The idempotency cache hashes ``request_payload`` to decide whether two
# calls are "the same". To make a retry on REST replay an MCP call (and
# vice versa) the two transports MUST hash the same canonical shape for
# semantically identical inputs — otherwise the cache namespace is
# shared in name only and the audit promise of "same cache namespace as
# the matching MCP tool" is hollow. Pre-fix REST and MCP built the
# payload independently: REST hashed the validated Pydantic model dump
# (defaults filled in), MCP hashed the raw boundary dict (defaults
# omitted), so identical user intents hashed differently.
#
# These helpers route both transports through the validated domain
# model so the dump (and therefore the hash) is identical regardless
# of how the caller supplied the data. Validation failures fall back to
# the raw dict so the retry of a malformed payload still hashes stably
# (the operation will surface the same field-error envelope, which IS
# safe to cache).


def canonical_create_campaign_payload(
    intake_data: dict[str, Any] | CampaignIntakeInput,
    owner_id: str,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Build the canonical ``bo_create_campaign`` idempotency payload.

    Both MCP tool wrappers and REST routes call this helper so the
    request-hash is invariant across transports for semantically
    identical inputs. Raw-dict inputs are normalized through
    :class:`CampaignIntakeInput` so defaults and field aliases land
    in the same shape REST already emits.

    A :class:`pydantic.ValidationError` is intentionally swallowed —
    the operation layer will surface the field-error envelope, and we
    still want a stable hash for the malformed payload so a retry
    replays the same response (caching a deterministic validation
    failure is safe).
    """
    if isinstance(intake_data, BaseModel):
        intake_dict = intake_data.model_dump()
    else:
        try:
            intake_dict = CampaignIntakeInput.model_validate(intake_data).model_dump()
        except ValidationError:
            intake_dict = dict(intake_data)
    return {
        "intake_data": intake_dict,
        "owner_id": owner_id,
        "verbosity": verbosity,
    }


def canonical_generate_suggestions_payload(
    campaign_id: str,
    batch_size: int | None,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Build the canonical ``bo_generate_suggestions`` idempotency payload.

    Mirrors :func:`canonical_create_campaign_payload`: MCP tool
    wrappers and REST routes both call this helper so the request
    hash — and therefore the cache row — is shared across transports
    for semantically identical generation requests. The inputs are
    already plain scalars, so canonicalization is just pinning the
    key set and defaults in one place.
    """
    return {
        "campaign_id": campaign_id,
        "batch_size": batch_size,
        "verbosity": verbosity,
    }


def canonical_submit_results_payload(
    campaign_id: str,
    results: Sequence[dict[str, Any] | ResultSubmissionInput],
    submitted_by: str,
    *,
    source: str = "api",
    force: bool = False,
    atomic: bool = True,
    continue_on_error: bool = False,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Build the canonical ``bo_submit_results`` idempotency payload.

    Mirrors :func:`canonical_create_campaign_payload`: every result
    row passes through :class:`ResultSubmissionInput.model_dump` so
    the hash is invariant across transports. Rows that fail
    validation fall back to a plain ``dict`` so the malformed
    payload still hashes stably.
    """
    canonical_rows: list[dict[str, Any]] = []
    for row in results:
        if isinstance(row, BaseModel):
            canonical_rows.append(row.model_dump())
            continue
        try:
            canonical_rows.append(ResultSubmissionInput.model_validate(row).model_dump())
        except ValidationError:
            canonical_rows.append(dict(row))
    return {
        "campaign_id": campaign_id,
        "results": canonical_rows,
        "submitted_by": submitted_by,
        "source": source,
        "force": force,
        "atomic": atomic,
        "continue_on_error": continue_on_error,
        "verbosity": verbosity,
    }
