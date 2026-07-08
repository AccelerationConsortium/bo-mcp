"""Request-size limits enforced at the REST transport boundary.

Centralising the bounds keeps schema annotations
(``Field(max_length=...)``), per-route file-size checks, and the
global JSON-body cap consistent so an operator reviewing one number
doesn't have to chase the others through unrelated modules. Each
limit's rationale is documented inline so a future bump (or
reduction) is grounded in a use-case argument, not arbitrary.

References:
----------
* RFC 9110 §15.5.14 (413 Content Too Large): the response status
  returned when a request payload exceeds a configured bound.
* OWASP API Security Top 10 ``API4: Unrestricted Resource
  Consumption`` — listing every input list/tuple field that lacks a
  cap is the surface this module closes.
"""

from __future__ import annotations

# Re-exported here so the REST transport keeps a single import surface
# for its caps; the value is owned by the shared operation layer (which
# enforces the same bound for every transport) and reaches the API
# through the client facade like every other server symbol.
from bo_mcp_server.client import MAX_GENERATION_BATCH_SIZE

__all__ = [
    "MAX_BATCH_CAMPAIGN_IDS",
    "MAX_BATCH_RESULTS",
    "MAX_COMPARE_CAMPAIGN_IDS",
    "MAX_GENERATION_BATCH_SIZE",
    "MAX_INTAKE_CONSTRAINTS",
    "MAX_INTAKE_OBJECTIVES",
    "MAX_INTAKE_PARAMETERS",
    "MAX_JSON_REQUEST_BODY_BYTES",
    "MAX_UPLOAD_FILE_SIZE_BYTES",
    "UPLOAD_READ_CHUNK_BYTES",
]

# ---------------------------------------------------------------------------
# Transport-level caps
# ---------------------------------------------------------------------------

# Hard ceiling for multipart file uploads. Experiment-result CSVs hit
# us at typically ~1 MB; we leave headroom for retroactive backfills
# (XLSX of an entire campaign's history) but refuse anything an
# operator would not expect through a normal client workflow. The cap
# is checked while streaming the upload so an oversized payload never
# materialises in process memory.
MAX_UPLOAD_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MiB

# Hard ceiling for JSON request bodies (everything that is not a
# multipart upload). Sized generously enough for the largest realistic
# create-campaign or batch-submit payload (each Result row is ~200 B
# of JSON; 5000 rows fit comfortably under this) yet small enough that
# a single OOM-grade POST can't tip a worker over.
MAX_JSON_REQUEST_BODY_BYTES = 5 * 1024 * 1024  # 5 MiB

# Chunk size for the streaming upload reader. Small enough to keep
# peak memory low if we end up rejecting partway, large enough that
# the per-iteration syscall cost is negligible for a 25 MiB upload.
UPLOAD_READ_CHUNK_BYTES = 64 * 1024  # 64 KiB

# ---------------------------------------------------------------------------
# Batch / collection caps (Pydantic ``Field(max_length=...)``)
# ---------------------------------------------------------------------------

# Maximum number of result rows accepted in a single batch submission.
# Larger backfills must be paged by the caller — at this size the
# operation already serialises ~1 MB of validated rows and re-fits the
# surrogate model from scratch, so a much larger batch wastes work
# more than it saves round-trips.
MAX_BATCH_RESULTS = 5000

# Maximum number of campaign ids accepted by the batch-status route.
# Resolution is read-only and per-id, so the bound is governed by
# response size rather than compute; 1000 keeps the response under a
# few hundred kilobytes at standard verbosity.
MAX_BATCH_CAMPAIGN_IDS = 1000

# Maximum number of campaigns in a single compare request. Pairwise
# trajectory joins are O(N²) in the worst case, and the response
# carries one summary block per campaign — 100 is generous for a UI
# comparison view, well past the dozen-ish that fit on a dashboard.
MAX_COMPARE_CAMPAIGN_IDS = 100

# ---------------------------------------------------------------------------
# Intake (campaign spec) complexity caps
# ---------------------------------------------------------------------------

# Maximum number of input parameters per campaign. Bayesian
# Optimization runs out of practical signal well before this many
# dimensions — SAASBO and TURBO papers report regimes up to a few
# hundred, and full-rank GPs above ~50 dimensions need careful
# tooling. Capping at 500 keeps the validator from spending O(N²)
# work on a spec the engine could not productively fit.
MAX_INTAKE_PARAMETERS = 500

# Maximum number of objectives per campaign. Multi-objective BO
# scales poorly beyond ~10 objectives (the Pareto front explodes
# combinatorially); 50 absorbs research-style sweeps while still
# rejecting obvious denial-of-service shapes.
MAX_INTAKE_OBJECTIVES = 50

# Maximum number of declared constraints per campaign. Each
# constraint enters the acquisition optimization as an additional
# evaluation; 200 keeps the optimizer's wall-time bounded.
MAX_INTAKE_CONSTRAINTS = 200
