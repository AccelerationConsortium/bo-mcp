"""Server-side constants for bo-mcp-server operations.

Centralizes magic numbers and thresholds used in operations so they are
documented, tunable, and consistent. Engine-level constants live in
``bo_engine.constants``; this module is for the MCP/API server layer.

Health-status thresholds (``HYPERVOLUME_STABILITY_THRESHOLD``,
``FALLBACK_HYPERVOLUME_IMPROVEMENT``) live in :mod:`bo_engine.constants`
so engine and server agree on what "stable" / "fallback improvement"
mean. They are re-exported below for callers that historically imported
them from this module.
"""

from bo_engine.constants import (
    FALLBACK_HYPERVOLUME_IMPROVEMENT,
    HYPERVOLUME_STABILITY_THRESHOLD,
)

__all__ = [
    "CONCURRENT_MODIFICATION_RETRY_AFTER_SECONDS",
    "DIVERSITY_HIGH_THRESHOLD",
    "DIVERSITY_MODERATE_THRESHOLD",
    "FALLBACK_HYPERVOLUME_IMPROVEMENT",
    "HYPERVOLUME_STABILITY_THRESHOLD",
    "MAX_GENERATION_BATCH_SIZE",
    "TRANSFER_SIMILARITY_GOOD",
    "TRANSFER_SIMILARITY_MODERATE",
    "TRANSFER_SIMILARITY_STRONG",
]

# =============================================================================
# Generation batch-size bounds (generate_suggestions, intake, REST)
# =============================================================================

# Upper bound for a generation request's ``batch_size``. q-batch
# acquisition optimization is superlinear in q, so an unbounded request
# can pin a worker for the full operation timeout. Submission
# (``MAX_BATCH_RESULTS``) and batch status already have caps; generation
# is the most expensive mutation and needs one too. Shared by the MCP
# operation, the intake schema, and the REST query parameter so every
# transport enforces the same bound.
MAX_GENERATION_BATCH_SIZE = 100

# =============================================================================
# Diversity Interpretation (diagnostics suggestions analysis)
# =============================================================================

# Score above which diversity is considered "high"
DIVERSITY_HIGH_THRESHOLD = 0.7

# Score above which diversity is considered "moderate" (below = "low")
DIVERSITY_MODERATE_THRESHOLD = 0.4

# =============================================================================
# Transfer Learning Similarity (transfer_candidates)
# =============================================================================

# Score above which a candidate is "strongly recommended"
TRANSFER_SIMILARITY_STRONG = 0.7

# Score above which a candidate is "recommended"
TRANSFER_SIMILARITY_GOOD = 0.6

# Score above which a candidate is "possible"
TRANSFER_SIMILARITY_MODERATE = 0.4

# =============================================================================
# Concurrency / Optimistic Locking (submit_results, campaign_lifecycle,
# generate_suggestions)
# =============================================================================

# Suggested backoff before retrying an operation that lost an optimistic-locking
# race. Surfaced to clients in the CONCURRENT_MODIFICATION error envelope so
# retries back off before re-hitting the same race.
CONCURRENT_MODIFICATION_RETRY_AFTER_SECONDS = 1.0
