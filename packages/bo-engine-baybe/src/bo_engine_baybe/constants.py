"""Named constants for the BayBE backend.

Per repo policy, defaults and thresholds live here instead of being
hardcoded at the call sites.
"""

from __future__ import annotations

# Integration-point budget for the active-learning acquisition (qNIPV,
# negated integrated posterior variance) on search spaces with a continuous
# part. BayBE's ``qNIPV.get_integration_points`` requires an explicit
# ``sampling_n_points`` for purely continuous spaces (its
# ``sampling_fraction`` default only applies to enumerable discrete
# candidates); 128 quasi-random integration points is in line with the
# Monte-Carlo point budgets used for qNIPV in the BoTorch active-learning
# literature and keeps per-recommend cost bounded.
DEFAULT_NIPV_SAMPLING_POINTS = 128

# ---------------------------------------------------------------------------
# Discrete search-space budgeting (large-categorical safeguard)
# ---------------------------------------------------------------------------

# Serialized-state byte budget for the enumerated discrete subspace. BayBE's
# ``Campaign.to_json`` embeds ``exp_rep`` / ``comp_rep`` as pickled+base64
# dataframes, so the persisted campaign scales with the enumerated candidate
# set; PostgreSQL additionally rejects any single protocol message over 1 GiB
# (https://www.postgresql.org/docs/current/protocol-message-formats.html),
# which is how the original incident killed the DB connection. Calibration
# (2026-07-07, BayBE 0.15.0): a purely categorical 8000-row OHE campaign
# serialized to 5.2 MB ≈ 1.03 × (exp_rep_bytes + comp_rep_bytes) from
# ``SubspaceDiscrete.estimate_product_space_size``; the 1.3 inflation factor
# below adds headroom for the pickle/base64 framing and the per-candidate
# ``_searchspace_metadata`` ledger. 32 MiB keeps every restore's
# read+unpickle cost operationally healthy (hundreds-of-MB states are legal
# for Postgres but unhealthy to restore per request).
DEFAULT_MAX_SEARCHSPACE_STATE_BYTES = 32 * 1024 * 1024

# Upper bound on the number of enumerated discrete candidates independent of
# their byte width. Matches the neutral engine's historical
# ``DISCRETE_ENUMERATION_MAX_POINTS`` product cap, so specs that previously
# built ``from_product`` keep building identically and specs that previously
# hit the enumeration-limit rejection now subsample instead.
DEFAULT_MAX_CANDIDATES = 10_000

# Safety multiplier applied to the (exp_rep + comp_rep) byte estimate before
# comparing against the byte budget — covers pickle+base64 framing and the
# metadata ledger (measured inflation ≈ 1.03; see calibration note above).
SEARCHSPACE_ESTIMATE_INFLATION_FACTOR = 1.3

# Bounded number of resample rounds when deduplication / constraint
# filtering shrinks a sampled candidate frame below the requested count.
# Spaces slightly above the threshold have high collision rates; the bound
# guarantees termination instead of looping toward the full enumeration.
SUBSAMPLE_TOPUP_MAX_ROUNDS = 8

# Minimum feasible candidate count for a subsampled discrete subspace. If
# constraint filtering cannot reach this floor within the bounded top-up
# rounds, the space is effectively infeasible for sampling and a typed
# error is raised instead of looping forever.
MIN_VIABLE_SUBSAMPLE_CANDIDATES = 10

# Upper bounds used by the construction-free budget pre-screen
# (``cheap_budget_prescreen``). They only have to be safe *over*-estimates:
# a spec whose upper-bound estimate already fits the budget is provably on
# the from_product path without building any BayBE parameter objects (in
# particular without computing substance descriptor tables inside
# ``validate_capabilities``). MORDRED — the widest substance encoding — is
# ~1800 descriptor columns; a substance parameter carrying
# ``kwargs_fingerprint`` has no safe fixed width bound and makes the
# prescreen abstain instead. The per-cell experimental-representation
# bound is label-aware (see ``_exp_label_bytes_bound``):
# EXP_LABEL_BYTES_UPPER_BOUND is the historical floor covering short
# labels and float cells, and long labels are bounded by
# ``sys.getsizeof`` of the declared label plus one object pointer per
# cell — the exact per-object measurement pandas'
# ``memory_usage(deep=True)`` (and therefore BayBE's estimate) sums. A
# character-count bound would under-estimate wide (UCS-2/UCS-4) Unicode
# labels, whose PEP 393 storage costs 2-4 bytes per character.
SUBSTANCE_COMP_WIDTH_UPPER_BOUND = 2048
EXP_LABEL_BYTES_UPPER_BOUND = 128
# CPython object-pointer size per cell of an object-dtype column
# (64-bit builds; pandas counts it on top of each cell's own bytes).
OBJECT_CELL_POINTER_BYTES = 8

# Maximum number of per-observation rows exposed by the row-level SHAP
# breakdown; bounds the diagnostics payload for large campaigns. The
# per-campaign override lives in
# ``backend_options['baybe'].insights.row_level_max_rows``.
DEFAULT_MAX_ROW_LEVEL_SHAP_ROWS = 32
