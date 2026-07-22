# BO-MCP Agent Cookbook

Quick reference for AI agents using the Bayesian Optimization MCP server.

## Quick Start (4 Steps)

```
1. bo_health_check → Verify connectivity
2. bo_validate_intake → Dry-run validate spec
3. bo_create_campaign → Start optimization
4. Loop: bo_generate_suggestions → bo_submit_results → bo_get_diagnostics(verbosity="minimal")
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
        {"status": "running", "limit": 10}
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
  }
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
  "results": [{"parameter_values": {"x": 5.2}, "objective_values": {"y": 12.3}}]
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

### Per-Error-Code Recovery Reference

| Code | Symptom | Concrete recovery |
|------|---------|-------------------|
| E001 (`VALIDATION_FAILED`) | Payload rejected by Pydantic | Inspect `field_errors` for dotted-path location; resend with the field corrected. |
| E002 (`CAMPAIGN_NOT_FOUND`) | Campaign UUID unknown | Call `bo_list_campaigns` (or read `campaigns://list`) to recover the real id. |
| E003 (`INVALID_STATE_TRANSITION`) | Wrong lifecycle action for current status | Read `campaign://{id}` for the current status; use the matrix in *Suggestion-Status Lifecycle* below for the legal next move. |
| E004 (`DUPLICATE_RESULT_DETECTED`) | Same parameter row already submitted | Re-submit with `force=true` if the duplicate is intentional; otherwise drop the row from the batch. |
| E005 (`CONCURRENT_MODIFICATION`) | Optimistic-lock conflict | Re-read the campaign, rebuild your write against the new `version`, and retry. |
| E006 (`IDEMPOTENCY_CONFLICT`) | Same idempotency key reused with a different payload | Generate a fresh UUIDv7 idempotency key for the new payload. |
| E101 (`MODEL_FITTING_FAILED`) | GP fit blew up on noisy / extreme data | Drop NaN/Inf rows, dampen outliers, or submit more clean observations before retrying. |
| E104 (`INSUFFICIENT_DATA`) | <2 observations available | Submit at least two valid result rows before calling `bo_generate_suggestions`. |
| E105 (`SUGGESTION_NOT_FOUND`) | Suggestion id unknown or already terminal | Re-list suggestions via `bo_list_suggestions`; only `pending` / `accepted` rows are mutable. |

## Verbosity Selection Guide

| Verbosity | ~Tokens | Use Case |
|-----------|---------|----------|
| minimal   | 50      | Tight optimization loops, monitoring many campaigns |
| standard  | 200     | Normal operation, debugging workflows |
| detailed  | 500+    | Deep debugging, analyzing model behavior |

### Per-Tool Default Verbosity

| Tool | Default verbosity | Notes |
|------|-------------------|-------|
| `bo_create_campaign` | `standard` | Mutation: emit warnings + spec summary by default. |
| `bo_generate_suggestions` | `standard` | Drops to `minimal` inside agent loops to save tokens. |
| `bo_submit_results` | `standard` | Pre-set in the tool decorator. |
| `bo_get_diagnostics` | `standard` | `minimal` retains the `next_action_recommendation`. |
| `bo_batch_get_status` | `minimal` | Optimized for monitoring many campaigns. |
| `bo_list_campaigns` / `bo_list_suggestions` / `bo_list_results` | `standard` | Query tools — switch to `detailed` for full provenance. |
| `bo_compare_campaigns` | `standard` | Use `detailed` when you need per-campaign breakdowns. |
| `bo_discover_transfer_candidates` | `standard` | `detailed` exposes per-component similarity scores. |
| `bo_health_check`, `bo_list_capabilities` | (no verbosity) | Always returns the full envelope. |

## Tool Selection Quick Reference

| User Intent | Recommended Tool(s) |
|-------------|---------------------|
| Check server is up | `bo_health_check` |
| Check backend features | `bo_list_capabilities` |
| Validate spec before creating | `bo_validate_intake` |
| Start new optimization | `bo_create_campaign` |
| List campaigns | `bo_list_campaigns` |
| Get next experiments | `bo_generate_suggestions` |
| Record outcomes | `bo_submit_results` or `bo_upload_results_file` |
| Check progress | `bo_get_diagnostics` (has `next_action` hint) |
| Review results | `bo_list_results` or `bo_export_campaign` |
| Review suggestions | `bo_list_suggestions` |
| Accept/reject suggestions | `bo_update_suggestion_status` |
| Understand a suggestion | `bo_get_suggestion_explanation` |
| Monitor many campaigns | `bo_batch_get_status` |
| Pause a campaign | `bo_pause_campaign` |
| Resume a campaign | `bo_resume_campaign` |
| Terminate a campaign | `bo_terminate_campaign` |
| Compare runs | `bo_compare_campaigns` |
| Find related prior work | `bo_discover_transfer_candidates` |

## Common Patterns

### Pattern 1: Monitor Multiple Campaigns

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

### `next_action_recommendation` Decision Tree

`bo_get_diagnostics` (any verbosity) and `bo_batch_get_status` (minimal verbosity)
embed a `next_action_recommendation` block with an `action`, a human-readable
`reason`, and an `urgency` (`low` | `normal` | `high`). Use this table to map
each action directly to a concrete follow-up tool call:

| `action` | When emitted | Concrete follow-up |
|----------|--------------|--------------------|
| `bo_generate_suggestions` | Campaign is healthy and ready for the next batch (or has no results yet). | Call `bo_generate_suggestions` with the same `campaign_id`. |
| `bo_submit_results` | One or more pending suggestions await results. | Run the experiments, call `bo_submit_results` for those rows. |
| `consider_stopping` | Convergence detector tripped (improvement rate or hypervolume stable). | Read `bo_get_diagnostics` at `detailed` verbosity to confirm, then call `bo_terminate_campaign` (consider `dry_run=true` first). |
| `review_outliers` | Diagnostics flag suspicious result rows (≥1 outlier among >5 results). | Inspect the offending result rows via `bo_list_results`; resubmit corrected values or call `bo_update_suggestion_status` with `rejected`. |
| `monitor_progress` | Campaign health is `warning`. | Keep optimizing but request `bo_get_diagnostics` at `standard` verbosity after each iteration. |
| `investigate_issues` | Campaign health is `critical`. | Read the `warnings` list, inspect `bo_get_diagnostics` at `detailed`, escalate to a human; do not auto-recover. |
| `review_campaign_status` | Campaign is `paused`, `completed`, or `failed`. | Read `campaign://{id}`; if intentional resume with `bo_resume_campaign`, otherwise create a new campaign. |

### Pattern 2: Upload Historical Data
```json
// Use bo_upload_results_file with CSV format
{
  "campaign_id": "...",
  "file_content": "param_x,param_y,obj_z\n1.0,2.0,3.0\n..."
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
| bo_get_diagnostics | 1-5s | Cached for 120s |

**Note**: High-dimensional problems (>20 params) and large batches increase suggestion generation time.

## Concurrency Envelope

The server uses optimistic concurrency on every campaign mutation
(`bo_pause_campaign`, `bo_resume_campaign`, `bo_terminate_campaign`,
`bo_submit_results`, `bo_update_suggestion_status`). A `version`
counter is incremented on every successful write; the next writer
that re-uses a stale `version` receives the structured envelope:

```json
{
  "success": false,
  "errors": ["..."],
  "code": "E005",
  "details": {
    "expected_version": 7,
    "actual_version": 8,
    "campaign_id": "abc-123",
    "action": "pause"
  }
}
```

**Recovery recipe.** Re-read the resource (`campaign://{id}` or
`bo_get_diagnostics`), rebuild the mutation against the fresh
`version`, then retry. Multi-step orchestrations should also re-fetch
any cached suggestion / result lists touched by the same transaction.
Do not blindly retry without re-reading — the second write would just
hit the same conflict.

**Idempotency keys** (`bo_create_campaign`, `bo_submit_results`,
`bo_update_suggestion_status`) protect against retry storms: re-using
the same UUIDv7 with the same payload replays the original response
verbatim (with `idempotency_replay: true`); re-using with a different
payload returns a `VALIDATION_FAILED` envelope flagged
`details.idempotency_conflict=true`.

**Dry-run preview.** All state-mutating tools accept `dry_run=true`.
The response carries `dry_run: true` and a `preview` block describing
what *would* change without persisting anything. Use this before
irreversible operations (`bo_terminate_campaign`) or to confirm a
plan with the human in the loop.

## Suggestion-Status Lifecycle

Suggestions move through the following states. Transitions marked **manual**
require a `bo_update_suggestion_status` call; **automatic** transitions are
written by the server when results are submitted.

```text
        ┌────────── manual ──────────┐
        │                            ▼
   PENDING ──manual──▶ ACCEPTED ──manual──▶ REJECTED
        │                  │                 │
        │                  │                 │
        ├──manual──▶ EXPIRED ◀──manual───────┘
        │                  ▲
        └──automatic──▶ COMPLETED (set by bo_submit_results)
```

**State semantics.**

- `PENDING` — generated by `bo_generate_suggestions`, awaiting human review.
- `ACCEPTED` — explicitly approved (e.g. queued for an experimental run).
- `REJECTED` — operator declined; suggestion is dropped from the active queue
  but *retained* in storage so future calls to `bo_list_suggestions` can show
  it for audit purposes. **Declines this suggestion instance only.** The row
  itself is terminal, but the parameter values are not excluded — the
  optimizer may generate a new suggestion at the same coordinates later.
- `EXPIRED` — suggestion is no longer relevant (e.g. instrument changed).
  Same persistence semantics as `REJECTED`; the distinction is *intent* —
  `REJECTED` is "I evaluated this and said no", `EXPIRED` is "context changed
  out from under it". Use `EXPIRED` when the suggestion was sound at
  generation time but stopped being applicable.
- `COMPLETED` — `bo_submit_results` accepted a result row whose
  `suggestion_id` matched this suggestion. Set automatically; manual
  callers cannot write `COMPLETED` directly.

**Re-usability.** Terminal statuses (`REJECTED`, `EXPIRED`, `COMPLETED`)
are absorbing: the same suggestion cannot be revived. If you change your
mind, call `bo_generate_suggestions` for a fresh batch — the prior row
remains queryable but inactive.

## Workflow Trace Propagation

Multi-step agent workflows can correlate their audit + log trail by
attaching an opaque trace id once per workflow:

- **REST** — pass the trace id in the `X-Trace-Id` request header. The
  middleware binds it for the duration of the request and echoes it
  back on the response header.
- **MCP** — the trace context is honored when the host binds it before
  invoking a tool (e.g. by wrapping the call in
  `bo_mcp_server.trace_context.bind_trace_id(...)`).

When set, every audit event records `trace_id` inside
`input_summary`, and every formatted response includes it under
`_metadata.trace_id`. When unset, neither field appears, so the
metadata envelope stays compact for one-off calls.

## Batch Operations

### Atomic Mode (Default)
All results succeed or all fail:
```json
{
  "campaign_id": "...",
  "results": [...],
  "atomic": true
}
```

### Continue on Error Mode
Process all results, get partial results:
```json
{
  "campaign_id": "...",
  "results": [...],
  "atomic": false,
  "continue_on_error": true
}
// Response includes partial_results mapping index to result_id or error
```

## Caching Behavior

- `bo_get_diagnostics` is cached for 120 seconds by default
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
  }
}
// Expected: {"success": true, "campaign_id": "...", "errors": []}
```

### Step 2: Optimization Loop (5 iterations)

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
│   ├── PAUSED → Call: bo_resume_campaign
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
├── CREATED → Can: start (auto), bo_terminate_campaign
│         → Cannot: bo_pause_campaign, bo_resume_campaign
│
├── RUNNING → Can: bo_pause_campaign, bo_terminate_campaign
│          → Cannot: bo_resume_campaign (already running)
│
├── PAUSED → Can: bo_resume_campaign, bo_terminate_campaign
│         → Cannot: bo_pause_campaign (already paused)
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

## Resource Subscriptions for Long-Running Workflows

Long-running orchestration loops can stop polling `campaign://{id}` and
let the server push state-change notifications instead. The server
implements MCP `resources/subscribe` so the initialize handshake
advertises `resources.subscribe = true`.

### Lifecycle

1. **Subscribe** to a campaign URI immediately after `bo_create_campaign`
   returns (or any time before you would otherwise poll):

   ```python
   await session.subscribe_resource("campaign://abc-123-def")
   ```

2. **Listen** for `notifications/resources/updated` on your MCP session
   transport. The server emits one notification per campaign-status
   transition (CREATED→RUNNING via `bo_generate_suggestions`,
   RUNNING↔PAUSED via `bo_pause_campaign` / `bo_resume_campaign`,
   ANY→COMPLETED via `bo_terminate_campaign`).

3. **Re-read** the resource when notified to fetch the new state:

   ```python
   updated = await session.read_resource("campaign://abc-123-def")
   ```

4. **Unsubscribe** when the campaign reaches a terminal state
   (`COMPLETED`, `FAILED`) or when your agent finishes:

   ```python
   await session.unsubscribe_resource("campaign://abc-123-def")
   ```

### What you do NOT receive

- **Iteration bumps.** Subscriptions push on `Campaign.status`
  transitions only; new suggestion batches against an already-RUNNING
  campaign do not push. Use `bo_get_diagnostics` to poll iteration
  counts when needed.
- **Result submissions.** `bo_submit_results` does not change campaign
  status, so it does not push. Use `bo_list_results` if you need to
  observe new results.
- **Errors.** A failed lifecycle transition (rejected by the state
  machine) does not push -- subscriptions only signal observed state
  changes.

### Notification delivery semantics

- Best-effort: a subscriber whose transport fails delivery is silently
  dropped from the registry (re-subscribe to re-arm).
- Sessions are tracked weakly: dropped transports do not need an
  explicit `unsubscribe` to be cleaned up.
- Notifications fire after the campaign-status write commits, so a
  subscriber that immediately re-reads the resource sees the new
  status.

## Related Documentation

- [TOOL_SCHEMAS.md](TOOL_SCHEMAS.md) - Complete input/output schemas for all tools
- [../../DESIGN_REVIEW.md](../../DESIGN_REVIEW.md) - System architecture and contracts
- [../../README.md](../../README.md) - Setup and installation
