"""Single source for intake / spec field documentation.

The same campaign fields are declared on three models — the domain
:class:`~bo_mcp_server.domain.campaign_spec.CampaignSpec`, the MCP
:class:`~bo_mcp_server.domain.intake_models.CampaignIntakeInput`, and the
REST ``IntakeData`` — and their ``Field(description=...)`` strings feed
the MCP tool schema and the REST OpenAPI document that agents read when
constructing payloads. Keeping each description (and the canonical
intake example) as a module-level constant here means the transports
cannot drift apart: an edit lands on every schema surface at once.

Descriptions that exist on only one model (nested parameter / objective
/ constraint fields) stay inline next to their ``Field``; only strings
shared across two or more models belong here.
"""

from typing import Any

DESCRIPTION_DOC = "Free-text human-readable note."

BATCH_SIZE_DOC = "Number of suggestions generated per call."

MAX_ITERATIONS_DOC = (
    "Cap on the number of completed BO iterations. Once reached, "
    "suggestion generation reports BUDGET_EXCEEDED instead of "
    "producing more suggestions."
)

MAX_OBSERVATIONS_DOC = (
    "Cap on the total number of observed results, irrespective of "
    "iteration grouping. Reaching it short-circuits suggestion "
    "generation even mid-iteration."
)

CONVERGENCE_TOLERANCE_DOC = (
    "Relative-improvement threshold below which the campaign is "
    "considered converged. Single-objective campaigns only — "
    "multi-objective campaigns are rejected at intake and must rely "
    "on hypervolume diagnostics instead."
)

INITIAL_DESIGN_SIZE_DOC = (
    "Number of space-filling (Sobol/random) warmup points before "
    "switching to the model-driven acquisition phase. None uses a "
    "dimension-adaptive default (BoTorch) or switches after the "
    "first measurement (BayBE). An explicitly set "
    "backend_options['baybe'].recommender.switch_after takes "
    "precedence over this field."
)

RANDOM_SEED_DOC = (
    "Campaign-level RNG seed. Optional. When supplied, the Sobol "
    "initial design and acquisition multi-start are deterministic "
    "within a fixed (torch version, device, deterministic-algorithms "
    "setting) triple; suggestions are NOT byte-identical across "
    "different torch versions, CPU vs. CUDA, or backend swaps. Set "
    "torch.use_deterministic_algorithms(True) for strictest behavior."
)

BACKEND_DOC = (
    "Optimization backend. 'auto' resolves to the deployment's "
    "configured default backend unless the spec uses features that "
    "only another installed backend can honor; resolution is driven "
    "by each backend's capability report (list the current "
    "per-backend feature matrix via the capability-listing "
    "tool/endpoint). Pin 'botorch' or 'baybe' explicitly to fail "
    "fast instead of silently switching."
)

BACKEND_OPTIONS_DOC = (
    "Backend-native option surface, keyed by backend name (currently "
    "only 'baybe' has a typed schema: BayBEBackendOptions / "
    "BayBEParameterOptions in the bo-engine-baybe package). Options "
    "addressed to a non-selected backend are rejected at intake when "
    "`backend` is pinned to a concrete name."
)

ACQUISITION_BETA_DOC = (
    "UCB exploration weight. Only valid with "
    "acquisition_method='upper_confidence_bound'; rejected otherwise."
)

SCALARIZER_DOC = (
    "Use 'mean' for arithmetic mean or 'geom_mean' for geometric mean; "
    "only valid with scalarization='desirability'. Null uses 'geom_mean'."
)

USE_INPUT_WARPING_DOC = (
    "Input warping for non-stationary objectives. BoTorch-only — "
    "reported UNSUPPORTED on the BayBE backend by default (see "
    "`acknowledge_degradations`)."
)

USE_COST_AWARE_DOC = (
    "Cost-aware acquisition (EIpu), weighting candidates by a cost "
    "model fit from the 'cost' metadata field of submitted results; "
    "without cost metadata, generation falls back to standard "
    "acquisition with a warning. BoTorch-only — reported UNSUPPORTED "
    "on the BayBE backend by default (see `acknowledge_degradations`)."
)

ACKNOWLEDGE_DEGRADATIONS_DOC = (
    "Opt-in list of attribute names (e.g. 'turbo_config', "
    "'outcome_constraints') whose BayBE-UNSUPPORTED status should "
    "downgrade to an IGNORED warning instead of rejecting the "
    "request, when running a BoTorch-only feature on "
    "backend='baybe'. 'transfer_learning' is not downgradable — "
    "declare a task parameter via parameter_options['baybe'] for "
    "BayBE-native transfer learning instead."
)

# Canonical minimal intake payload advertised as the schema example on
# both transports. Must validate against IntakeData and
# CampaignIntakeInput (both use extra='forbid'); a round-trip test pins
# that so the advertised example can never go stale.
INTAKE_EXAMPLES: list[dict[str, Any]] = [
    {
        "name": "example-baybe-campaign",
        "parameters": [{"name": "x", "type": "continuous", "bounds": {"lower": 0.0, "upper": 1.0}}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "backend": "baybe",
    }
]
