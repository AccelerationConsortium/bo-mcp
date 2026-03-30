# BO-MCP Agent Cookbook

Quick reference for AI agents using the Bayesian Optimization MCP server.

## Quick Start (3 Steps)

```
1. bo_create_campaign → Start optimization
2. Loop: bo_generate_suggestions → bo_submit_results → bo_get_diagnostics(verbosity="minimal")
```

---

## MCP Resources vs Tools

Resources are read-only data sources accessed via URI. Tools perform actions.

| Interaction Type | When to Use | Example |
|------------------|-------------|---------|
| **Tool call** | Actions, mutations, queries with filters | `session.call_tool("bo_create_campaign", {...})` |
| **Resource read** | Static data retrieval by ID | `session.read_resource("campaign://abc-123")` |

### When to Use Each

```
Need to list campaigns?
├── With filters (owner, status, limit) → Use tool: bo_list_campaigns
└── Simple list all → Either: bo_list_campaigns OR resource: campaigns://list

Need campaign details?
├── Quick ID lookup → Resource: campaign://{id}
└── Full status + next_action → Tool: bo_get_diagnostics

Need pending suggestions?
└── Resource: suggestions://{campaign_id}
```

### Python SDK Example

```python
from mcp import ClientSession

async def example(session: ClientSession):
    # Tool call (preferred for most operations)
    result = await session.call_tool(
        "bo_list_campaigns",
        {"owner_id": "user-uuid", "status": "running", "limit": 10}
    )

    # Resource read (for simple lookups)
    campaign = await session.read_resource("campaign://abc-123-def")
    suggestions = await session.read_resource("suggestions://abc-123-def")
```

## Minimal Workflow Example

### Step 1: Create Campaign
```json
// Tool: bo_create_campaign
{
  "intake_data": {
    "name": "Quick Optimization",
    "parameters": [
      {"name": "x", "type": "continuous", "bounds": [0, 10]}
    ],
    "objectives": [
      {"name": "y", "direction": "minimize"}
    ]
  },
  "owner_id": "user-uuid-here"
}
// Response: {"success": true, "campaign_id": "abc-123-..."}
```

### Step 2: Optimization Loop
```json
// Tool: bo_generate_suggestions
{"campaign_id": "abc-123-..."}
// Response: {"suggestions": [{"id": "...", "parameter_values": {"x": 5.2}}]}

// Tool: bo_submit_results
{
  "campaign_id": "abc-123-...",
  "results": [{"parameter_values": {"x": 5.2}, "objective_values": {"y": 12.3}}],
  "submitted_by": "user-uuid-here"
}

// Tool: bo_get_diagnostics (use verbosity=minimal for tight loops)
{"campaign_id": "abc-123-...", "verbosity": "minimal"}
// Response: {"health": "healthy", "converged": false, "key_metric": {"best_value": 12.3}}
```

## Decision Trees

### When to Stop Optimization
```
bo_get_diagnostics.converged == true? → STOP (optimization converged)
bo_get_diagnostics.health == "critical"? → WARN user, consider stopping
bo_get_diagnostics.iteration > max_iterations? → STOP (budget exhausted)
Otherwise → CONTINUE
```

### Error Recovery
```
Error Code E002 "Campaign not found"
→ Action: Call campaigns://list to get valid campaign IDs

Error Code E003 "Invalid state transition"
→ Action: Check campaign status with campaign://{id}, use appropriate lifecycle tool

Error Code E004 "Duplicate result detected"
→ Action: Use force=True to override, or skip this result

Error Code E101 "Model fitting failed"
→ Action: Check data quality, may need more observations (minimum 2)
```

## Verbosity Selection Guide

| Verbosity | ~Tokens | Use Case |
|-----------|---------|----------|
| minimal   | 50      | Tight optimization loops, monitoring many campaigns |
| standard  | 200     | Normal operation, debugging workflows |
| detailed  | 500+    | Deep debugging, analyzing model behavior |

## Tool Selection Quick Reference

| User Intent | Recommended Tool(s) |
|-------------|---------------------|
| Check server is up | `bo_health_check` |
| Start new optimization | `bo_create_campaign` |
| List all campaigns | `bo_list_campaigns` (v3.3+) |
| Get next experiments | `bo_generate_suggestions` |
| Record outcomes | `bo_submit_results` or `bo_upload_results_file` |
| Check progress | `bo_get_diagnostics` (has `next_action` hint) |
| Monitor many campaigns | `bo_batch_get_status` (v3.3+) |
| Understand a suggestion | `bo_get_suggestion_explanation` |
| Pause/resume/terminate | `bo_manage_campaign_lifecycle` (v3.3+ consolidated) |
| Compare runs | `bo_compare_campaigns` |
| Find related prior work | `bo_discover_transfer_candidates` |

## Common Patterns

### Pattern 1: Monitor Multiple Campaigns (v3.3+)
```json
// Use bo_batch_get_status for efficiency (1 call vs N calls)
{"campaign_ids": ["id1", "id2", "id3"], "verbosity": "minimal"}
// Response: {"campaigns": {"id1": {...}, "id2": {...}, ...}, "failed_ids": []}

// Or use bo_compare_campaigns for detailed comparison
{"campaign_ids": ["id1", "id2", "id3"]}
```

### Pattern 1b: Follow Proactive Guidance
```json
// bo_get_diagnostics includes next_action_recommendation
{"campaign_id": "abc-123", "verbosity": "minimal"}
// Response includes: {"next_action": {"action": "bo_submit_results", "reason": "..."}}
// Agent follows the recommendation without manual interpretation
```

### Pattern 2: Upload Historical Data
```json
// Use bo_upload_results_file with CSV format
{
  "campaign_id": "...",
  "file_content": "param_x,param_y,obj_z\n1.0,2.0,3.0\n...",
  "submitted_by": "user-uuid"
}
```

### Pattern 3: Warm Start with Transfer Learning
```json
// 1. Find related campaigns
{"campaign_id": "new-campaign-id"}  // bo_discover_transfer_candidates

// 2. Results show similar prior campaigns that can inform the new one
// Transfer learning is applied automatically based on parameter similarity
```

## Response Time Expectations

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| bo_health_check | <100ms | Immediate |
| bo_create_campaign | <500ms | Database writes |
| bo_generate_suggestions (initial) | 1-3s | Sobol sampling |
| bo_generate_suggestions (with model) | 3-30s | Depends on data size, dimensions |
| bo_submit_results | <500ms | Per result |
| bo_get_diagnostics | 1-5s | Cached for 30s |

**Note**: High-dimensional problems (>20 params) and large batches increase suggestion generation time.

## Batch Operations

### Atomic Mode (Default)
All results succeed or all fail:
```json
{
  "campaign_id": "...",
  "results": [...],
  "submitted_by": "...",
  "atomic": true
}
```

### Continue on Error Mode
Process all results, get partial results:
```json
{
  "campaign_id": "...",
  "results": [...],
  "submitted_by": "...",
  "atomic": false,
  "continue_on_error": true
}
// Response includes partial_results mapping index to result_id or error
```

## Caching Behavior

- `bo_get_diagnostics` is cached for 30 seconds by default
- Cache is automatically invalidated when:
  - `bo_submit_results` completes successfully
  - `bo_generate_suggestions` completes successfully
- Use `use_cache=false` to force fresh computation

---

## Complete Workflow Example: Chemical Process Optimization

This example demonstrates a full multi-objective optimization workflow with real parameter types.

### Problem Setup

Optimize a chemical reaction for maximum yield and minimum cost:
- **Parameters**: temperature (continuous), pressure (discrete), catalyst (categorical)
- **Objectives**: yield (maximize), cost (minimize)
- **Constraint**: sum of reagent fractions equals 1.0

### Step 1: Create Campaign

```json
// Tool: bo_create_campaign
{
  "intake_data": {
    "name": "Catalyst Optimization",
    "description": "Multi-objective optimization for reaction yield and cost",
    "parameters": [
      {"name": "temperature", "type": "continuous", "bounds": [50.0, 200.0], "description": "Reaction temperature in Celsius"},
      {"name": "pressure", "type": "discrete", "bounds": [1, 10], "description": "Pressure in bar"},
      {"name": "catalyst", "type": "categorical", "categories": ["Pt", "Pd", "Rh"], "description": "Catalyst type"},
      {"name": "reagent_A", "type": "continuous", "bounds": [0.0, 1.0]},
      {"name": "reagent_B", "type": "continuous", "bounds": [0.0, 1.0]}
    ],
    "objectives": [
      {"name": "yield", "direction": "maximize", "unit": "%"},
      {"name": "cost", "direction": "minimize", "unit": "USD"}
    ],
    "constraints": [
      {"type": "sum_equals", "parameters": ["reagent_A", "reagent_B"], "value": 1.0}
    ],
    "batch_size": 3
  },
  "owner_id": "user-uuid"
}
// Expected: {"success": true, "campaign_id": "...", "errors": []}
```

### Step 2: Create Campaign

```json
// Tool: bo_create_campaign
{
  "intake_data": { /* same as above */ },
  "owner_id": "550e8400-e29b-41d4-a716-446655440000"
}
// Response: {"success": true, "campaign_id": "abc-123-...", "spec_id": "def-456-..."}
```

### Step 3: Optimization Loop (5 iterations)

```python
import json

CAMPAIGN_ID = "abc-123-..."
USER_ID = "550e8400-e29b-41d4-a716-446655440000"
MAX_ITERATIONS = 5

for iteration in range(MAX_ITERATIONS):
    # Generate suggestions
    suggestions = await session.call_tool(
        "bo_generate_suggestions",
        {"campaign_id": CAMPAIGN_ID, "verbosity": "minimal"}
    )
    suggestion_data = json.loads(suggestions.content[0].text)

    # Run experiments (your lab/simulation code here)
    results = []
    for sugg in suggestion_data["suggestions"]:
        params = sugg["parameter_values"]
        # Example: call your experiment function
        yield_value, cost_value = run_experiment(
            temperature=params["temperature"],
            pressure=params["pressure"],
            catalyst=params["catalyst"],
            reagent_A=params["reagent_A"],
            reagent_B=params["reagent_B"]
        )
        results.append({
            "parameter_values": params,
            "objective_values": {"yield": yield_value, "cost": cost_value}
        })

    # Submit results
    await session.call_tool(
        "bo_submit_results",
        {
            "campaign_id": CAMPAIGN_ID,
            "results": results,
            "submitted_by": USER_ID,
            "verbosity": "minimal"
        }
    )

    # Check diagnostics and next action
    diagnostics = await session.call_tool(
        "bo_get_diagnostics",
        {"campaign_id": CAMPAIGN_ID, "verbosity": "minimal"}
    )
    diag_data = json.loads(diagnostics.content[0].text)

    # Follow next_action recommendation
    next_action = diag_data.get("next_action", {})
    if next_action.get("action") == "consider_stopping":
        print(f"Converged at iteration {iteration + 1}")
        break

# Final results
final_diag = await session.call_tool(
    "bo_get_diagnostics",
    {"campaign_id": CAMPAIGN_ID, "verbosity": "detailed"}
)
print(f"Pareto front: {json.loads(final_diag.content[0].text)['pareto_front']}")
```

### Expected Output After 5 Iterations

```json
{
  "health_status": "healthy",
  "iteration": 5,
  "n_results": 15,
  "n_pareto_points": 4,
  "hypervolume": 0.823,
  "pareto_front": [
    {"yield": 95.2, "cost": 250.0},
    {"yield": 88.5, "cost": 150.0},
    {"yield": 82.1, "cost": 100.0},
    {"yield": 75.3, "cost": 80.0}
  ]
}
```

---

## Troubleshooting Decision Trees

### "Cannot generate suggestions"

```
Is campaign status RUNNING?
├── No → What is the status?
│   ├── CREATED → First call auto-transitions to RUNNING (should work)
│   ├── PAUSED → Call: bo_manage_campaign_lifecycle(action="resume")
│   ├── COMPLETED → Campaign finished; create new campaign
│   └── FAILED → Check errors; may need new campaign
└── Yes → Are there any results?
    ├── No results → Uses Sobol sampling (should always work)
    │   └── Still fails? → Check bo_create_campaign validation errors for spec issues
    └── Has results → Model fitting issue
        ├── < 2 results → Add 1+ more observations
        ├── All NaN/Inf values? → Submit valid numeric results
        └── >= 2 valid results → Check bo_get_diagnostics for model health
```

### "Model fitting failed" (Error E101)

```
Check bo_get_diagnostics output for health_status:
├── "Insufficient data" (E104)
│   └── Action: Submit at least 2 results with valid objective values
├── "Numerical issues"
│   ├── Check for NaN/Inf in submitted results
│   ├── Check for extreme outliers (>10 std from mean)
│   └── Action: Use force=true to resubmit clean data
├── "Constraint conflict"
│   ├── Verify constraint parameters exist in spec
│   └── Verify constraint bounds allow feasible region
└── "Model health: critical"
    └── Action: Review outliers in diagnostics, consider restart
```

### "Duplicate result detected" (Error E004)

```
Same parameter values submitted twice?
├── Intentional (re-run experiment) → Use force=true to override
├── Accidental duplicate → Skip this result
└── Similar but not identical → Check duplicate similarity threshold
    └── If similarity < 1.0, values are slightly different; submit normally
```

### "Invalid state transition" (Error E003)

```
Check current campaign status first:
│
├── CREATED → Can: start (auto), terminate
│         → Cannot: pause, resume
│
├── RUNNING → Can: pause, terminate
│          → Cannot: resume (already running)
│
├── PAUSED → Can: resume, terminate
│         → Cannot: pause (already paused)
│
└── COMPLETED → Cannot: any lifecycle changes
            → Action: Create new campaign for further optimization
```

---

## Understanding Convergence

**Note**: Convergence detection requires at least 10 observations. Before that, `convergence` field will show insufficient data.

### Single-Objective Campaigns

```
bo_get_diagnostics.convergence.converged == true means:
├── Improvement rate < 1% for last 5 iterations
├── OR max_iterations reached
└── OR explicit stopping criterion met

Agent Actions:
├── converged=true → Recommend stopping, show best_parameters
├── converged=false + improving → Continue optimization
└── converged=false + regressing → Review for data quality issues
```

### Multi-Objective Campaigns

```
bo_get_diagnostics.convergence.converged == true means:
├── Hypervolume improvement < 1% for last 5 iterations
├── Pareto front is stable (minimal additions)
└── No single "best" point - present trade-offs

Agent Actions:
├── converged=true → Present Pareto front options to user
│   └── "Choose between high-yield/high-cost vs low-yield/low-cost"
├── converged=false → Continue to expand Pareto front
└── n_pareto_points increasing → Good progress, continue
```

### Early Convergence Warning

```
If converged=true BUT n_results < 20:
└── Warn user: "Optimization converged early. This may indicate:
    • Simple objective function (good!)
    • Local optimum (consider: new campaign with different seed)
    • Overly constrained search space (review constraints)"
```

---

## Initial Design Size Guidance

The system uses Sobol sequence for initial exploration before model-based optimization.

**Default**: `2 × n_parameters + 1`

| Parameters | Default Initial Points | Recommendation |
|------------|------------------------|----------------|
| 2-5 | 5-11 | Usually sufficient |
| 6-10 | 13-21 | Add 5-10 more for noisy objectives |
| 11-20 | 23-41 | Consider TuRBO (auto-enabled >20 params) |
| >20 | 41+ | SAASBO auto-enabled for >50 params |

**Override in campaign creation**:
```json
{
  "intake_data": {
    "initial_design_size": 30,  // Explicit override
    ...
  }
}
```

**When to increase initial design**:
- Highly noisy objectives (measurement uncertainty)
- Many local optima expected
- Categorical parameters with >5 categories
- Multi-fidelity campaigns (need diverse fidelity samples)

---

## Related Documentation

- [TOOL_SCHEMAS.md](TOOL_SCHEMAS.md) - Complete input/output schemas for all tools
- [../../DESIGN_REVIEW.md](../../DESIGN_REVIEW.md) - System architecture and contracts
- [../../README.md](../../README.md) - Setup and installation
