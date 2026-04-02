# BO-MCP-Server Tool Schemas

This document provides detailed schema documentation for all MCP tools exposed by the `bo-mcp-server` package.

**Related Documentation:**
- [AGENT_COOKBOOK.md](AGENT_COOKBOOK.md) - Quick reference for AI agents with decision trees and common patterns

---

## Agent Quick Reference

### Recommended Workflow

```
bo_create_campaign → [bo_generate_suggestions → bo_submit_results]* → bo_get_diagnostics
                       ↑__________________|
                       (repeat until convergence)
```

### Tool Selection Guide

| User Intent | Recommended Tool(s) |
|-------------|---------------------|
| Check backend features | `bo_list_capabilities` |
| Validate before creating | `bo_validate_intake` |
| Start new optimization | `bo_create_campaign` |
| List all campaigns | `bo_list_campaigns` |
| Get next experiments | `bo_generate_suggestions` |
| Record experiment outcomes | `bo_submit_results` or `bo_upload_results_file` |
| Check optimization progress | `bo_get_diagnostics` (includes `next_action_recommendation`) |
| Review results | `bo_list_results`, `bo_export_campaign` |
| Review suggestions | `bo_list_suggestions`, `bo_update_suggestion_status` |
| Monitor multiple campaigns | `bo_batch_get_status` |
| Understand a suggestion | `bo_get_suggestion_explanation` |
| Pause/resume/terminate | `bo_pause_campaign`, `bo_resume_campaign`, `bo_terminate_campaign` |
| Compare multiple optimizations | `bo_compare_campaigns` |
| Find related prior work | `bo_discover_transfer_candidates` |

### Common Error Recovery

| Error | Cause | Recovery |
|-------|-------|----------|
| "Campaign not found" | Invalid UUID or deleted campaign | Verify `campaign_id` format (UUID v4), check `campaigns://list` |
| "Invalid state transition" | Wrong campaign status | Check status with `campaign://{id}`, use appropriate lifecycle tool |
| "Duplicate result detected" | Same parameters submitted twice | Use `force: true` parameter to override, or skip |
| "Validation failed" | Invalid intake configuration | Review `errors` array, fix the intake, and retry `bo_create_campaign` |

---

## Overview

The MCP server exposes 19 tools organized into six categories:

| Category | Tools |
|----------|-------|
| **Server Health** | `bo_health_check`, `bo_list_capabilities` |
| **Campaign Management** | `bo_create_campaign`, `bo_list_campaigns`, `bo_validate_intake` |
| **Campaign Lifecycle** | `bo_pause_campaign`, `bo_resume_campaign`, `bo_terminate_campaign` |
| **Suggestion Generation** | `bo_generate_suggestions`, `bo_get_suggestion_explanation`, `bo_list_suggestions`, `bo_update_suggestion_status` |
| **Result Submission** | `bo_submit_results`, `bo_upload_results_file`, `bo_list_results`, `bo_export_campaign` |
| **Analysis & Strategy** | `bo_get_diagnostics`, `bo_compare_campaigns`, `bo_discover_transfer_candidates`, `bo_batch_get_status` |

---

## Automatic Method Selection

The server automatically selects optimal algorithms based on problem characteristics. Agents don't need to configure this - check `method_selection` in `bo_generate_suggestions` response to see what was used.

| Condition | Model | Acquisition | Strategy | Why |
|-----------|-------|-------------|----------|-----|
| < max(2, n_params) obs | Any | Sobol sequence | Initial design | Insufficient data for GP model fitting |
| 1 objective, <20 params | SingleTaskGP | qLogNEI | L-BFGS-B | Standard efficient approach |
| 1 objective, ≥20 params | SingleTaskGP | qLogNEI | TuRBO | Trust region prevents over-exploration in high dimensions |
| 1 objective, ≥50 params | SaasFullyBayesianGP | qLogEI | SAASBO | Sparse priors for automatic feature selection |
| 2+ objectives | ModelListGP | qLogNEHVI | L-BFGS-B | Hypervolume-based multi-objective optimization |
| With fidelity param | SingleTaskMultiFidelityGP | qMFKG | Cost-aware | Multi-fidelity Knowledge Gradient |
| With prior campaigns | RGPE (GP Ensemble) | qLogNEI | Transfer | Transfer learning from related campaigns |
| Categorical params | SingleTaskGP + one-hot | qLogNEI | L-BFGS-B | One-hot encoding for categorical variables |

### Initial Design Phase

Before Bayesian optimization begins, the system generates initial points using Sobol sequence for space-filling design.

**Default size**: `2 × n_parameters + 1`

**Override**: Set `initial_design_size` in campaign creation:
```json
{
  "intake_data": {
    "initial_design_size": 20,
    ...
  }
}
```

**Guidance for agents**:
| Parameters | Default Initial | Recommendation |
|------------|-----------------|----------------|
| 2-5 | 5-11 | Usually sufficient |
| 6-15 | 13-31 | Add extra for noisy objectives |
| >15 | 31+ | TuRBO/SAASBO auto-enabled |

---

## Conditional Response Fields

Some response fields only appear under certain conditions. This section documents when to expect these fields.

### `bo_get_diagnostics` Conditional Fields

| Field | Appears When | Description |
|-------|--------------|-------------|
| `outliers` | Outliers detected in results | Contains `count`, `details[]`, `recommendations` |
| `convergence` | ≥10 iterations completed | Contains `converged`, `convergence_score`, `recommendation` |
| `hypervolume_history` | Multi-objective + ≥1 result | Array of hypervolume values per iteration |
| `pareto_front` | Multi-objective only | Array of non-dominated objective value sets |
| `best_value` | Single-objective only | Best observed objective value |
| `feature_importance` | ≥2×n_params observations | Parameter importance scores |
| `loo_cv_metrics` | ≥5 observations | Leave-one-out cross-validation metrics |

### `bo_generate_suggestions` v2.5+ Fields

| Field | Appears When | Description |
|-------|--------------|-------------|
| `batch_diversity` | Always (v2.5+) | Diversity metrics for generated batch |
| `pending_points` | Pending suggestions exist | Info about filtered/expired pending points |
| `method_selection` | Always | Model and acquisition function selection info |

### `bo_submit_results` v2.5+ Fields

| Field | Appears When | Description |
|-------|--------------|-------------|
| `duplicates_detected` | Near-duplicate found | Details of detected duplicates |
| `warnings` | Non-blocking issues | Array of warning messages |

---

## Response Verbosity (v3.1+)

Several tools support a `verbosity` parameter to control response payload size:

| Level | ~Tokens | Description |
|-------|---------|-------------|
| `minimal` | 50 | Success status + key metrics only. Use for tight loops. |
| `standard` | 200 | Default. Excludes debugging fields like hyperparameters. |
| `detailed` | 500+ | All fields including LOO-CV metrics and hyperparameters. |

### Tools Supporting Verbosity

- `bo_get_diagnostics(verbosity="minimal|standard|detailed")`
- `bo_generate_suggestions(verbosity="minimal|standard|detailed")`
- `bo_compare_campaigns(verbosity="minimal|standard|detailed")`
- `bo_discover_transfer_candidates(verbosity="minimal|standard|detailed")`
- `bo_create_campaign(verbosity="minimal|standard|detailed")`- `bo_submit_results(verbosity="minimal|standard|detailed")`- `bo_list_campaigns(verbosity="minimal|standard|detailed")`- `bo_batch_get_status(verbosity="minimal|standard|detailed")`
---

## Error Codes (v3.1+)

Tools return structured errors with recovery guidance:

```json
{
  "success": false,
  "error": {
    "code": "E002",
    "message": "Campaign not found",
    "recovery_action": "Use campaigns://list resource to verify campaign exists."
  },
  "errors": ["Campaign not found"]  // Backward compatibility
}
```

### Error Code Reference

| Code | Message | Recovery Action |
|------|---------|-----------------|
| E001 | Invalid campaign_id format | Verify UUID v4 format. Use campaigns://list for valid IDs. |
| E002 | Campaign not found | Use campaigns://list to verify campaign exists. |
| E003 | Invalid state transition | Check status with campaign://{id}. See valid transitions. |
| E004 | Duplicate result detected | Use force=True to override, or skip result. |
| E005 | Validation failed | Review errors array, fix issues, retry. |
| E006 | Missing parameters | Add at least one parameter to intake_data.parameters. |
| E007 | Missing objectives | Add at least one objective to intake_data.objectives. |
| E008 | Constraint violation | Check that constraint parameters exist and bounds are valid. |
| E009 | Suggestion not found | Verify suggestion_id. Use suggestions://{campaign_id} to list. |
| E101 | Model fitting failed | Check data quality. May need more observations. |
| E102 | Acquisition optimization failed | Try reducing batch_size or check constraints. |
| E103 | Database error | Retry operation. Check DATABASE_URL if persistent. |
| E104 | Insufficient data | Need at least 2 observations for model fitting. |

---

## Server Health Tools

### `bo_health_check`

Verifies MCP server health and connectivity. Use this tool to confirm the server is running before starting optimization workflows.

**Input Schema:**
```json
{}  // No parameters required
```

**Output Schema:**
```json
{
  "healthy": "boolean",
  "version": "string",
  "database": "connected | error",
  "tools_available": "integer",
  "uptime_seconds": "integer"
}
```

---

## Campaign Management Tools

### `bo_create_campaign`

Creates a new optimization campaign from validated intake data.

**Input Schema:**
```json
{
  "intake_data": "object (campaign configuration payload)",
  "owner_id": "string (UUID)"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "campaign_id": "string (UUID) | null",
  "spec_id": "string (UUID) | null",
  "errors": ["string"]
}
```

---

### `bo_list_campaigns`
Lists all campaigns with optional filtering. Tool-based alternative to `campaigns://list` resource.

**Input Schema:**
```json
{
  "owner_id": "string (UUID, optional) - Filter by owner",
  "status": "created | running | paused | completed | failed (optional)",
  "limit": "integer (default: 20, max: 100)",
  "verbosity": "minimal | standard | detailed (default: standard)"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "campaigns": [
    {
      "campaign_id": "string (UUID)",
      "name": "string",
      "status": "string",
      "iteration": "integer (standard+)",
      "n_results": "integer (standard+)",
      "created_at": "string (standard+)",
      "spec_summary": "object (detailed)"
    }
  ],
  "total_count": "integer",
  "errors": ["string"]
}
```

---

### `bo_pause_campaign`

Pauses a running campaign.

**Input Schema:**
```json
{
  "campaign_id": "string (UUID)"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "campaign_id": "string",
  "status": "string (new status)",
  "previous_status": "string",
  "errors": ["string"]
}
```

**State Transition:** RUNNING → PAUSED

---

### `bo_resume_campaign`

Resumes a paused campaign.

**Input Schema:**
```json
{
  "campaign_id": "string (UUID)"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "campaign_id": "string",
  "status": "string (new status)",
  "previous_status": "string",
  "errors": ["string"]
}
```

**State Transition:** PAUSED → RUNNING

---

### `bo_terminate_campaign`

Terminates a campaign, marking it as completed.

**Input Schema:**
```json
{
  "campaign_id": "string (UUID)"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "campaign_id": "string",
  "status": "string (new status)",
  "previous_status": "string",
  "errors": ["string"]
}
```

**State Transition:** CREATED/RUNNING/PAUSED → COMPLETED

---

## Batch Status Tools

### `bo_batch_get_status`

Batch status retrieval for multiple campaigns. Reduces N calls to 1 for dashboard/monitoring scenarios.

**Input Schema:**
```json
{
  "campaign_ids": ["string (UUID)"] (max 20),
  "verbosity": "minimal | standard | detailed (default: minimal)"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "campaigns": {
    "campaign_id": {
      "name": "string",
      "status": "string",
      "iteration": "integer",
      "n_results": "integer",
      "health": "healthy | warning | critical (standard+)",
      "key_metric": "object (standard+)",
      "convergence": "object (detailed)"
    }
  },
  "failed_ids": ["string (UUID)"],
  "errors": ["string"]
}
```

---

## Suggestion Generation Tools

### `bo_generate_suggestions`

Generates the next batch of experiment suggestions for a campaign.

**Input Schema:**
```json
{
  "campaign_id": "string (UUID)",
  "batch_size": "integer | null (optional, default: campaign's batch_size)"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "suggestions": [
    {
      "id": "string (UUID)",
      "parameter_values": {
        "param_name": "number | string"
      },
      "provenance": {
        "iteration": "integer",
        "batch_index": "integer",
        "generation_method": "initial_design | bayesian_optimization",
        "acquisition_value": "number | null",
        "model_uncertainty": "number | null",
        "acquisition_function": "string | null",
        "model_type": "string | null",
        "random_seed": "integer | null",
        "model_version": "string | null",
        "confidence_level": "high | medium | low | null",
        "explanation": "string | null"
      },
      "created_at": "string (ISO 8601)"
    }
  ],
  "iteration": "integer",
  "errors": ["string"],
  "method_selection": {
    "model_type": "string",
    "acquisition_function": "string",
    "optimization_strategy": "string",
    "input_transforms": ["string"],
    "explanation": "string",
    "confidence": "string",
    "alternatives": ["string"],
    "warnings": ["string"]
  },
  "batch_diversity": {
    "min_pairwise_distance": "number",
    "mean_pairwise_distance": "number",
    "diversity_score": "number (0-1)",
    "is_diverse": "boolean"
  },
  "pending_points": {
    "total_pending": "integer",
    "valid_pending": "integer",
    "stale_expired": "integer",
    "note": "string"
  } | null
}
```

---

### `bo_get_suggestion_explanation`

Gets detailed explanation for why a suggestion was generated.

**Input Schema:**
```json
{
  "suggestion_id": "string (UUID)"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "explanation": "string | null",
  "provenance": {
    "iteration": "integer",
    "batch_index": "integer",
    "generation_method": "string",
    "acquisition_value": "number | null",
    "model_uncertainty": "number | null",
    "acquisition_function": "string | null",
    "model_type": "string | null",
    "random_seed": "integer | null",
    "confidence_level": "string | null"
  },
  "errors": ["string"]
}
```

---

## Result Submission Tools

### `bo_submit_results`

Submits experimental results for a campaign.

**Input Schema:**
```json
{
  "campaign_id": "string (UUID)",
  "results": [
    {
      "parameter_values": {
        "param_name": "number | string"
      },
      "objective_values": {
        "obj_name": "number"
      },
      "suggestion_id": "string (UUID, optional)",
      "metadata": "object (optional)"
    }
  ],
  "submitted_by": "string (UUID)",
  "source": "gui | file_upload | api (default: api)",
  "force": "boolean (default: false) - Override duplicate detection"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "result_ids": ["string (UUID)"],
  "errors": ["string"],
  "warnings": ["string"],
  "duplicates_detected": [
    {
      "result_index": "integer",
      "existing_result_id": "string (UUID)",
      "similarity": "number (0-1)"
    }
  ] | null
}
```

---

### `bo_upload_results_file`

Uploads experimental results from a CSV file.

**Input Schema:**
```json
{
  "campaign_id": "string (UUID)",
  "file_content": "string (CSV content)",
  "file_format": "csv (default: csv)",
  "submitted_by": "string (UUID, optional)"
}
```

**CSV Format:**
- Parameter columns: `param_<name>` (e.g., `param_temperature`, `param_pressure`)
- Objective columns: `obj_<name>` (e.g., `obj_yield`, `obj_cost`)
- Optional: `suggestion_id` column to link results to specific suggestions

### CSV Format Examples

**Basic Example** (2 parameters, 1 objective):
```csv
param_temperature,param_pressure,obj_yield
100.0,5.0,85.2
120.0,6.5,91.3
80.0,4.0,72.8
```

**With Suggestion ID** (linking to generated suggestions):
```csv
suggestion_id,param_temperature,param_pressure,obj_yield,obj_cost
550e8400-e29b-41d4-a716-446655440001,100.0,5.0,85.2,150.0
550e8400-e29b-41d4-a716-446655440002,120.0,6.5,91.3,180.0
```

**Column Naming Convention**:
- Parameters: `param_<name>` (e.g., `param_temperature`, `param_catalyst`)
- Objectives: `obj_<name>` (e.g., `obj_yield`, `obj_cost`)
- Optional: `suggestion_id` to link results to specific suggestions

**Output Schema:**
```json
{
  "success": "boolean",
  "results_created": "integer",
  "errors": ["string"]
}
```

---

## Analysis & Strategy Tools

### `bo_get_diagnostics`

Gets comprehensive diagnostic information for a campaign.

**Input Schema:**
```json
{
  "campaign_id": "string (UUID)"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "campaign_status": "created | running | paused | completed | failed",
  "iteration": "integer",
  "n_results": "integer",
  "n_pending_suggestions": "integer",
  "errors": ["string"],

  // Single-objective fields (null for multi-objective)
  "best_value": "number | null",
  "best_parameters": "object | null",
  "improvement_history": ["number"] | null,
  "improvement_rate": "number | null",

  // Multi-objective fields (null for single-objective)
  "pareto_front": [
    {
      "obj_name": "number"
    }
  ] | null,
  "hypervolume": "number | null",
  "n_pareto_points": "integer | null",
  "hypervolume_history": ["number"] | null,

  // Outlier detection (v2.5+, appears when outliers detected)
  "outliers": {
    "count": "integer",
    "details": [
      {
        "result_index": "integer",
        "objective": "string",
        "standardized_error": "number",
        "actual_value": "number",
        "predicted_value": "number"
      }
    ],
    "recommendation": "string"
  } | null,

  // Convergence analysis (v2.5+, appears when ≥10 iterations)
  "convergence": {
    "converged": "boolean",
    "convergence_score": "number (0-1)",
    "reason": "string | null",
    "avg_improvement": "number",
    "iterations_without_improvement": "integer",
    "recommendation": "string"
  } | null,

  // Model diagnostics (requires sufficient data)
  "objective_ranges": {
    "obj_name": {
      "min": "number",
      "max": "number",
      "direction": "minimize | maximize"
    }
  },
  "model_info": {
    "type": "string",
    "acquisition_function": "string",
    "batch_strategy": "string",
    "kernel": "string",
    "input_warping": "boolean"
  },
  "feature_importance": {
    "param_name": "number"
  } | null,
  "loo_cv_metrics": {
    "obj_name": {
      "rmse": "number",
      "mae": "number",
      "r_squared": "number"
    }
  } | null,
  "model_correlation": "number | null",

  // Health and progress
  "health_status": "healthy | warning | critical",
  "progress_status": "improving | stable | regressing",
  "warnings": ["string"],

  // Agent Usability Metrics (v2.4)
  "uncertainty_trend": {
    "mean_uncertainty": "number",
    "std_uncertainty": "number",
    "trend": "decreasing | stable | increasing",
    "slope": "number",
    "interpretation": "string"
  } | null,
  "exploration_exploitation": {
    "exploration_ratio": "number (0-1)",
    "diversity_score": "number (0-1)",
    "average_distance_to_best": "number",
    "balance_assessment": "exploration_heavy | balanced | exploitation_heavy",
    "recommendation": "string"
  } | null,
  "hyperparameters": {
    "lengthscales": {
      "param_name": "number"
    },
    "noise_variance": "number",
    "output_scale": "number",
    "kernel_type": "string",
    "model_type": "string",
    "interpretation": "string"
  } | null,
  "constraint_satisfaction": {
    "satisfaction_rate": "number (0-1)",
    "recent_satisfaction_rate": "number (0-1)",
    "feasible_count": "integer",
    "infeasible_count": "integer",
    "trend": "improving | stable | worsening",
    "interpretation": "string"
  } | null,
  "suggestion_diversity": {
    "diversity_score": "number (0-1)",
    "n_suggestions": "integer",
    "interpretation": "string"
  } | null
}
```

---

### `bo_compare_campaigns`

Compares multiple optimization campaigns.

**Input Schema:**
```json
{
  "campaign_ids": ["string (UUID)"] // 2-10 campaigns
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "campaigns": [
    {
      "campaign_id": "string",
      "campaign_name": "string",
      "status": "string",
      "n_results": "integer",
      "iteration": "integer",
      "n_parameters": "integer",
      "n_objectives": "integer",
      "is_multi_objective": "boolean",
      "best_value": "number | null",
      "improvement_rate": "number | null",
      "sample_efficiency": "number",
      "hypervolume": "number | null",
      "n_pareto_points": "integer | null"
    }
  ],
  "comparison": {
    "best_sample_efficiency": "string (campaign name)",
    "best_single_objective": "string (campaign name) | null",
    "best_multi_objective": "string (campaign name) | null",
    "total_campaigns_compared": "integer",
    "recommendation": "string"
  },
  "errors": ["string"]
}
```

---

### `bo_discover_transfer_candidates`

Discovers campaigns suitable for transfer learning.

**Input Schema:**
```json
{
  "campaign_id": "string (UUID)",
  "similarity_threshold": "number (0-1, default: 0.5)",
  "max_candidates": "integer (default: 5)"
}
```

**Output Schema:**
```json
{
  "success": "boolean",
  "target_campaign": {
    "campaign_id": "string",
    "name": "string",
    "n_parameters": "integer",
    "n_objectives": "integer",
    "parameter_names": ["string"],
    "objective_names": ["string"]
  },
  "candidates": [
    {
      "campaign_id": "string",
      "name": "string",
      "status": "string",
      "n_results": "integer",
      "iteration": "integer",
      "similarity_score": "number (0-1)",
      "component_scores": {
        "parameter_similarity": "number (0-1)",
        "objective_similarity": "number (0-1)",
        "bounds_overlap": "number (0-1)",
        "data_richness": "number (0-1)"
      },
      "recommendation": "string"
    }
  ],
  "overall_recommendation": "string",
  "errors": ["string"]
}
```

---

## Error Handling

All tools follow a consistent error handling pattern:

1. **Input Validation Errors**: Return `success: false` with specific error messages
2. **Not Found Errors**: Return `success: false` with `"<Entity> not found"` message
3. **State Validation Errors**: Return `success: false` with state-specific error messages

Example error response:
```json
{
  "success": false,
  "errors": ["Invalid campaign_id format", "Campaign not found"],
  "warnings": []
}
```

---

## Type Reference

### Campaign Status
- `created`: Campaign created but no suggestions generated yet
- `running`: Campaign is actively generating suggestions
- `paused`: Campaign is paused
- `completed`: Campaign has finished (optimization complete)
- `failed`: Campaign encountered an error

### Parameter Types
- `continuous`: Continuous numerical parameter with bounds [lower, upper]
- `discrete`: Integer parameter with bounds or explicit values
- `categorical`: Categorical parameter with explicit categories

### Constraint Types
- `sum_equals`: Sum of parameters equals value
- `sum_less_than`: Sum of parameters is less than value
- `sum_greater_than`: Sum of parameters is greater than value
- `linear`: Linear combination with coefficients

### Result Source
- `gui`: Submitted via web interface
- `file_upload`: Uploaded from file
- `api`: Submitted via API

---

## Version History

- Added Agent Efficiency improvements:
  - `bo_list_campaigns` - Tool-based campaign listing with filters
  - `bo_batch_get_status` - Multi-campaign status in one call
  - `bo_pause_campaign`, `bo_resume_campaign`, `bo_terminate_campaign` - Individual lifecycle tools
  - Verbosity parameter added to `bo_create_campaign` and `bo_submit_results`
  - `next_action_recommendation` added to `bo_get_diagnostics`
- **v3.1**: Added `bo_health_check` tool, response verbosity parameter, structured error codes with recovery actions, and AGENT_COOKBOOK.md reference
- **v2.5**: Added batch diversity metrics, pending points tracking, outlier detection, convergence analysis, duplicate detection with `force` override, and agent quick reference
- **v2.4**: Added agent usability tools (`bo_compare_campaigns`, `bo_discover_transfer_candidates`) and enhanced diagnostics
- **v2.0**: Added transfer learning, multi-fidelity, and outcome constraints
- **v1.1**: Added LOO-CV metrics and feature importance
- **v1.0**: Initial release with core tools
