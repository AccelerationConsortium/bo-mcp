"""Server-side constants for bo-mcp-server operations.

Centralizes magic numbers and thresholds used in operations so they are
documented, tunable, and consistent. Engine-level constants live in
``bo_engine.constants``; this module is for the MCP/API server layer.
"""

# =============================================================================
# Hypervolume Convergence (batch_status, diagnostics health)
# =============================================================================

# Relative threshold for hypervolume stability detection.
# If max - min of recent HV values is below this fraction of the latest HV,
# convergence is considered likely.
HYPERVOLUME_STABILITY_THRESHOLD = 0.001

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
