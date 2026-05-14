"""Submit results tool wrapper for MCP."""

from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.operations.submit_results import submit_results_operation
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import NON_IDEMPOTENT_MUTATION


def _payload_for_result(result: ResultSubmissionInput | dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe dict representation of a single result payload."""
    if isinstance(result, BaseModel):
        return result.model_dump()
    return dict(result)


@mcp.tool(name="bo_submit_results", annotations=NON_IDEMPOTENT_MUTATION)
async def submit_results(
    campaign_id: str,
    results: list[ResultSubmissionInput],
    submitted_by: str,
    source: str = "api",
    force: bool = False,
    atomic: bool = True,
    continue_on_error: bool = False,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Submit experimental results for a campaign.

    Workflow: Call after running experiments from bo_generate_suggestions.
    Follow up with bo_get_diagnostics to check progress and convergence.

    Args:
        campaign_id: UUID of the campaign.
        results: List of result payloads with parameter_values and
            objective_values.
        submitted_by: UUID of the user submitting results.
        source: Result source (gui, file_upload, or api).
        force: If True, skip duplicate detection.
        atomic: If True (default), the batch is pre-validated and rejected
            on the first error — no result row or suggestion-status update
            is persisted. If False, invalid rows are skipped and valid rows
            are saved individually.
        continue_on_error: If True, continue after errors (non-atomic).
        verbosity: Response verbosity level (minimal, standard, detailed).
        idempotency_key: Optional client-supplied key (recommended: UUIDv7
            per logical batch). If supplied and the same key was used in
            the last 24 hours with an identical payload, the prior
            response is replayed with ``idempotency_replay: True`` instead
            of re-executing the submission. Re-using a key with a
            different payload returns a ``VALIDATION_FAILED`` envelope
            (``details.idempotency_conflict=True``).

    Returns:
        Dictionary with success, result_ids, errors, warnings.
    """
    request_payload = {
        "campaign_id": campaign_id,
        "results": [_payload_for_result(r) for r in results],
        "submitted_by": submitted_by,
        "source": source,
        "force": force,
        "atomic": atomic,
        "continue_on_error": continue_on_error,
        "verbosity": verbosity,
    }

    async def run(session: AsyncSession) -> dict[str, Any]:
        return await submit_results_operation(
            campaign_id=campaign_id,
            results=results,
            submitted_by=submitted_by,
            source=source,
            force=force,
            atomic=atomic,
            continue_on_error=continue_on_error,
            verbosity=verbosity,
            session=session,
        )

    return await apply_idempotency(
        tool_name="bo_submit_results",
        idempotency_key=idempotency_key,
        request_payload=request_payload,
        executor=run,
    )
