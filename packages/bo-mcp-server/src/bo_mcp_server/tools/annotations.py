"""Centralized safety-hint annotations for MCP tool registration.

The MCP spec (``ToolAnnotations``) exposes three hints that agents can use
to choose a retry policy without trial-and-error:

- ``readOnlyHint=True`` — the tool does not mutate server state. Always
  safe to retry.
- ``destructiveHint=True`` — the tool may make irreversible changes (e.g.
  terminating a campaign). Retries should be gated behind confirmation.
- ``idempotentHint=True`` — repeating the same call has no additional
  effect. Safe to retry on ambiguous network failures.

Hints are advisory per the MCP spec. The retry-safe path for
non-idempotent mutating tools is the ``idempotency_key`` argument added
by TODO 1.46.
"""

from mcp.types import ToolAnnotations

# Read-only tools (always safe to retry).
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)

# Idempotent state changes (pause/resume) — safe to retry, no compounding
# effect because the operation rejects no-op transitions.
IDEMPOTENT_MUTATION = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
)

# Irreversible destructive mutation (terminate). MCP spec calls this out
# specifically as the case where ``destructiveHint=True`` matters: the
# operation cannot be undone, so agents should gate retries on user
# confirmation.
DESTRUCTIVE_MUTATION = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
)

# Default mutating tools: not destructive, not idempotent. Agents that
# need retry safety should pass the ``idempotency_key`` argument (see
# TODO 1.46) — the server then deduplicates retried calls server-side.
NON_IDEMPOTENT_MUTATION = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
)
