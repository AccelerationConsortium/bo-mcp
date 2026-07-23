# BO-MCP Operating Manual

Last updated: 2026-07-23 · verified against BO MCP API version 0.1.0.

This is the canonical, narrative manual for operating BO-MCP — the Bayesian
Optimization service behind this API. It tells you **what order to call things
in, who owns which state, how to recover from failures, and which rules every
client (human or agent) must follow**. It deliberately does *not* enumerate
request and response fields: the field-level schema reference is the OpenAPI
document at [/docs](/docs) (Swagger UI), [/redoc](/redoc) (ReDoc), and
[/openapi.json](/openapi.json).

How to read this manual:

- If you are integrating for the first time, read [§1](#1-overview-and-mental-model)
  and [§2](#2-end-to-end-campaign-procedure) in full.
- If you are writing an autonomous agent or a campaign script, the
  non-negotiable rules live in [§3](#3-state-ownership-and-continuation) and
  [§4](#4-reliability-and-error-handling).
- If something failed, start at [§4.7](#47-recovery-walk-throughs) and the
  troubleshooting table in [§5.7](#57-troubleshooting-quick-table).
- The complete endpoint ↔ MCP-tool map is in [§6](#6-endpoint-map).

Agents: fetch this document as plain Markdown from `GET /manpage.md`, or read
the MCP resource `docs://manpage` — both serve exactly this text.

## 1. Overview and mental model

### 1.1 What BO-MCP is

BO-MCP runs Bayesian Optimization campaigns as a service. You describe an
experimental design problem once (the *intake*), and the service proposes
candidate experiments (*suggestions*), learns from the outcomes you report
back (*results*), and tells you when to stop. The layers:

```text
  your client / agent            (owns: evaluation of candidates)
        │
        ├── REST API  /api/v1/*  (this service; X-API-Key auth)
        └── MCP server bo_* tools + resources (same operations, agent transport)
        │
  shared operations layer        (owns: campaign state, idempotency,
        │                         lifecycle rules, next-action recommendation)
        │
  optimization backends          (BoTorch engine, BayBE adapter — selected
        │                         per campaign, discoverable via capabilities)
        │
  PostgreSQL storage             (single source of truth for all state)
```

The one thing BO-MCP never does is run your experiments. Evaluating a
suggested candidate — in a lab, a simulation, a quantum-chemistry code — is
owned entirely by the caller. Everything else about campaign progress is owned
by the server ([§3](#3-state-ownership-and-continuation)).

Both transports expose the same operations: every REST endpoint has a
matching MCP tool (mapped in [§6](#6-endpoint-map)), backed by the same
operations layer, the same database, and the same idempotency cache
([§4.1](#41-idempotency-keys)).

### 1.2 Core nouns

| Noun | Meaning |
|---|---|
| **Campaign** | One optimization run: a search space, objectives, and the accumulated observations. Identified by a UUID. |
| **Intake** | The campaign specification submitted at creation: parameters, objectives, constraints, batch size, backend options. **Immutable** after creation. |
| **Suggestion** | A candidate parameter configuration proposed by the backend, with its own UUID and status machine ([§1.4](#14-suggestion-status-machine)). |
| **Result / measurement** | An externally evaluated outcome for one parameter configuration: finite objective values, optionally linked to the suggestion that proposed it. |
| **Iteration** | Server-maintained counter of generate→submit rounds. You never track this yourself. |
| **Status** | Campaign lifecycle state ([§1.3](#13-campaign-lifecycle-state-machine)) or suggestion state ([§1.4](#14-suggestion-status-machine)). |
| **Diagnostics** | Server-computed model health, convergence assessment, best-so-far / Pareto front. Expensive — see [§5.1](#51-next-action-versus-diagnostics-and-diagnostics-cost). |
| **Next-action recommendation** | The server's own answer to "what should this campaign's operator do next" — the loop-control signal ([§2.2](#22-the-canonical-loop-skeleton)). |
| **Backend** | The optimization engine executing the campaign (BoTorch or BayBE). Chosen at intake, discoverable via `GET /api/v1/capabilities`. |
| **Capability** | A feature flag advertised by a backend (parameter types, constraint types, acquisition options). Construct intakes only from advertised capabilities. |

Field-level shapes for all of these live in the OpenAPI schema at
[/docs](/docs), including the per-backend option schemas spliced into the
intake ([§5.2](#52-capability-discovery-and-backend-options)).

### 1.3 Campaign lifecycle state machine

```text
                 validate (dry-run, creates nothing)
                     │
   POST /api/v1/campaigns  ──▶  CREATED
                                   │  first suggestion generation
                                   │  (automatic transition)
                                   ▼
              ┌──── pause ──── RUNNING ◀──────────────┐
              ▼                  │  ▲                 │
           PAUSED ── resume ─────┘  │               reopen
              │                     │                 │
              │                 terminate             │
              │                     │                 │
              └── terminate ──▶ COMPLETED ────────────┘

           FAILED  (terminal; server-side failure marker)
```

Valid lifecycle actions per state, as enforced by the server
(`POST /api/v1/campaigns/{campaign_id}/lifecycle` with
`{"action": "pause" | "resume" | "terminate" | "reopen"}`):

| Current status | Allowed actions |
|---|---|
| `created` | `terminate` (generation auto-starts the campaign; there is no explicit "start") |
| `running` | `pause`, `terminate` |
| `paused` | `resume`, `terminate` |
| `completed` | `reopen` (returns the campaign to `running` for continuation) |
| `failed` | none — create a new campaign |

Two rules worth internalizing:

- `created → running` happens **automatically** on the first successful
  suggestion generation. Never look for a "start" action.
- `reopen` is the continuation path for a finished campaign ("run another N
  experiments"). It is deliberately distinct from `resume` so that an
  accidental extra `resume` is harmless while reviving a completed campaign
  requires explicit intent. Never rebuild a finished campaign from scratch —
  see [§3.4](#34-pause-resume-reopen-terminate).

### 1.4 Suggestion status machine

Suggestions move through five states. *Manual* transitions go through
`POST /api/v1/suggestions/{suggestion_id}/status` (MCP:
`bo_update_suggestion_status`); *automatic* transitions are written by the
server when a submitted result row references the suggestion. This is the
complete edge set — no other transition exists:

| From | To | Trigger |
|---|---|---|
| `pending` | `accepted` | manual — approve for execution |
| `pending` | `rejected` | manual — evaluated and declined |
| `pending` | `expired` | manual — no longer applicable |
| `accepted` | `rejected` | manual — declined after approval |
| `accepted` | `expired` | manual — no longer applicable |
| `pending` | `completed` | automatic — result submitted with this suggestion's id |
| `accepted` | `completed` | automatic — result submitted with this suggestion's id |

- `pending` — generated, awaiting execution or review.
- `accepted` — explicitly approved (e.g. queued for an experimental run).
- `rejected` — operator declined ("I evaluated this and said no"). Retained
  in storage for audit. Rejection retires **this suggestion instance only**:
  the parameter values are not excluded from future generation, and the
  optimizer may propose the same coordinates again in a later batch.
- `expired` — superseded or no longer applicable ("context changed out from
  under it" — instrument swapped, batch superseded). Same persistence
  semantics as `rejected`; the distinction is intent.
- `completed` — a submitted result row referenced this suggestion. Written
  automatically; clients cannot set `completed` directly.

Terminal statuses (`rejected`, `expired`, `completed`) are **absorbing**: the
same suggestion cannot be revived. If you change your mind, generate a fresh
batch — the prior row remains queryable but inactive.

Note on field naming: a suggestion's identity key is `suggestion_id`, and it
is the same key everywhere the value appears — generate, list, and query
responses, result rows, and status updates. Copy it verbatim when submitting
a result; only the key copies over (the result row schema rejects the
suggestion's other fields).

### 1.5 Conventions

- **Auth** — every `/api/v1/*` request carries an `X-API-Key` header. The
  documentation surfaces (`/docs`, `/redoc`, `/openapi.json`, `/manpage`,
  `/health`) are public.
- **Versioned paths** — all operational endpoints live under `/api/v1/`.
  This manual documents only `/api/v1/*` paths.
- **Response envelope** — campaign-operation responses carry
  `schema_version` (the breaking-change signal: pre-1.0, payload changes
  land directly on `/api/v1` and increment this integer — currently `2`),
  a boolean `success`, and on operation-level failure
  `errors` plus a structured `error` object with `code`, `message`,
  `recovery_action`, `retryable`, and `retry_after`. Over REST, some
  rejections surface instead as an HTTP error status whose `detail` body
  carries that same structured error object
  ([§4.2](#42-transport-errors-versus-operation-rejections)).
  **Always check `success` — a 2xx status does not mean the operation was
  accepted.**
- **Correlation** — send `X-Request-ID` to name an individual request (one
  is generated when absent and echoed back), and `X-Trace-Id` to correlate a
  multi-step workflow across requests
  ([§4.6](#46-request-correlation)).
- **Dry runs** — every state-mutating operation accepts `dry_run=true`
  (a REST request-body field; a query parameter on suggestion generation
  and file upload; a parameter on the MCP tools) and returns a `preview`
  block describing what would change without persisting anything. Dry runs
  bypass the idempotency cache. Use them before irreversible operations
  (terminate) or to confirm a plan with a human in the loop.

## 2. End-to-end campaign procedure

### 2.1 The ten steps

MCP tool names in parentheses; the full map is in [§6](#6-endpoint-map).

1. **Discover capabilities** — `GET /api/v1/capabilities`
   (`bo_list_capabilities`). Learn which backends are available and which
   parameter types, constraint types, and options each supports. Construct
   an intake only from advertised capabilities.
2. **Dry-run validation** — `POST /api/v1/campaigns/validate`
   (`bo_validate_intake`). Field errors come back without creating anything.
   Fix everything here before creating.
3. **Create** — `POST /api/v1/campaigns` (`bo_create_campaign`) with an
   `Idempotency-Key` header ([§4.1](#41-idempotency-keys)). Returns
   `campaign_id`; record it — it is the only handle you need to keep.
4. **Ask the server what to do next** — `POST /api/v1/campaigns/status/batch`
   (`bo_batch_get_status`), which embeds a `next_action_recommendation`
   block per campaign; diagnostics responses carry the same recommendation
   as `next_action`. This replaces any client-side progress bookkeeping:
   the recommendation, not a local counter, decides whether another round
   is warranted ([§2.2](#22-the-canonical-loop-skeleton)).
5. **Generate suggestions** —
   `POST /api/v1/suggestions/{campaign_id}/generate`
   (`bo_generate_suggestions`) — **or reuse pending suggestions** after an
   interrupted run: `POST /api/v1/suggestions/{campaign_id}/query`
   (`bo_list_suggestions`) with `status_filter="pending"`. Generating a new
   batch while evaluable pending suggestions exist wastes experiments.
6. **Evaluate candidates externally** — caller-owned. Run the experiment,
   the simulation, the calculation.
7. **Submit results** — `POST /api/v1/results/{campaign_id}`
   (`bo_submit_results`) with an `Idempotency-Key` header. Objective values
   must be finite numbers; link each row to its suggestion via
   `suggestion_id` where one exists. Bulk historical data can go through
   `POST /api/v1/results/{campaign_id}/upload` (`bo_upload_results_file`).
   An intentional replicate of already-measured coordinates needs
   `force=true`: a body field on JSON submission (REST and the MCP
   `bo_submit_results` tool — pair it with a fresh idempotency key,
   [§4.1](#41-idempotency-keys)), or a query parameter on the REST upload
   route, which is not idempotency-cached. The MCP
   `bo_upload_results_file` tool has no `force` override — submit an
   intentional replicate through `bo_submit_results` instead.
8. **Retire unusable suggestions** —
   `POST /api/v1/suggestions/{suggestion_id}/status`
   (`bo_update_suggestion_status`) with `rejected` (evaluated and declined,
   duplicate, outside the active space) or `expired` (no longer applicable).
9. **Diagnostics at boundaries, not per iteration** —
   `GET /api/v1/diagnostics/{campaign_id}` (`bo_get_diagnostics`). Expensive
   and recomputed from all results — call at invocation boundaries with a
   generous timeout ([§5.1](#51-next-action-versus-diagnostics-and-diagnostics-cost)).
10. **Wrap up the invocation** — export artifacts via
    `GET /api/v1/campaigns/{campaign_id}/export` (`bo_export_campaign`), then
    `POST /api/v1/campaigns/{campaign_id}/lifecycle` (`bo_pause_campaign` /
    `bo_resume_campaign` / `bo_terminate_campaign` / `bo_reopen_campaign`).
    Default to **pause** at the end of an invocation; terminate only when the
    user explicitly asks ([§3.4](#34-pause-resume-reopen-terminate)).

### 2.2 The canonical loop skeleton

Language-neutral pseudocode. Every rule it encodes is normative and explained
in [§3](#3-state-ownership-and-continuation):

```text
campaign_id = argv.campaign_id            # attach to an existing campaign …
if campaign_id is missing:                # … or create one, exactly once
    validate_intake(intake)                       # POST /api/v1/campaigns/validate
    campaign_id = create_campaign(intake, idempotency_key=new_uuid())

while invocation_budget_remaining():      # bounds THIS PROCESS, not the campaign
    decision = next_action(campaign_id)           # POST /api/v1/campaigns/status/batch
    if decision.action != "bo_generate_suggestions":
        break                             # server says: don't generate more now

    batch = pending_suggestions(campaign_id)      # POST /api/v1/suggestions/{campaign_id}/query
    if batch is empty:
        batch = generate_suggestions(campaign_id) # POST /api/v1/suggestions/{campaign_id}/generate

    outcomes = evaluate_externally(batch)         # caller-owned
    submit_results(campaign_id, outcomes,         # POST /api/v1/results/{campaign_id}
                   idempotency_key=new_uuid())    #   rows carry the suggestion's suggestion_id
    reject_unusable(batch, outcomes)              # POST /api/v1/suggestions/{suggestion_id}/status

diagnostics(campaign_id, generous_timeout)        # GET /api/v1/diagnostics/{campaign_id}
lifecycle(campaign_id, action="pause")            # POST /api/v1/campaigns/{campaign_id}/lifecycle
```

The `next_action_recommendation` actions and the follow-up each one demands:

| `action` | When emitted | Concrete follow-up |
|---|---|---|
| `bo_generate_suggestions` | Campaign healthy and ready for the next batch (or has no results yet). | Generate (or reuse pending) suggestions. |
| `bo_submit_results` | Pending suggestions await results. | Evaluate them and submit result rows. |
| `consider_stopping` | Convergence detector tripped (improvement rate or front stable). | Confirm via diagnostics at `detailed` verbosity, then terminate (consider `dry_run=true` first). |
| `terminate_campaign` | The campaign's `max_iterations` budget is spent and no pending suggestions remain — the intake is immutable, so the budget cannot be extended ([§3.2](#32-invocation-budgets-are-not-campaign-budgets)). | Review results, then `POST /api/v1/campaigns/{campaign_id}/lifecycle` with `terminate`; further optimization needs a new campaign. |
| `review_outliers` | Diagnostics flag suspicious result rows. | Inspect via `GET /api/v1/results/{campaign_id}`; resubmit corrected values or reject the suggestion. |
| `monitor_progress` | Campaign health is `warning`. | Keep optimizing; check diagnostics at `standard` verbosity each iteration. |
| `investigate_issues` | Campaign health is `critical`. | Read the warnings, inspect `detailed` diagnostics, escalate to a human. Do not auto-recover. |
| `review_campaign_status` | Campaign is `paused`, `completed`, or `failed`. | Resume / reopen if continuation is intended; otherwise stop. |

### 2.3 A complete runnable session

Recorded 2026-07-23 against a development stack (dev auth, BayBE default
backend). One continuous parameter, one minimized objective, one full loop
iteration — including an idempotent-retry proof. The block is directly
executable: ids are extracted from the live responses, and the idempotency
keys are minted **once** and stored so a retry reuses them verbatim, and
`--fail-with-body` makes any HTTP-level error abort the run under `set -e`
while still printing the response body.
Response comments show the actual server output, abridged to the fields
discussed in this manual.

```bash
BASE=http://localhost:8000
KEY='dev-api-key-12345'        # development key; use your real key
json() { python3 -c "import json,sys; print(json.load(sys.stdin)$1)"; }

# Idempotency keys are minted once, up front: a retry must reuse the SAME
# key with the byte-identical payload (a fresh key would create a second
# logical operation).
CREATE_KEY=$(uuidgen)
SUBMIT_KEY=$(uuidgen)
INTAKE='{"intake":{"name":"Manpage demo","parameters":[{"name":"x","type":"continuous","bounds":[0,10]}],"objectives":[{"name":"y","direction":"minimize"}]}}'

# 0. Liveness (public, no key)
curl --fail-with-body -sS $BASE/health
# {"healthy":true,"service":"api","version":"0.1.0","database":"connected",...}

# 1. Discover capabilities
curl --fail-with-body -sS -H "X-API-Key: $KEY" $BASE/api/v1/capabilities
# {"schema_version":2,"backend":"baybe","supported_features":["categorical",
#  "mixed_search_space","multi_objective"],"conditional_features":{...},
#  "available_backends":["botorch","baybe"],"default_backend":"baybe",
#  "server_version":"0.1.0"}

# 2. Dry-run validation (creates nothing)
curl --fail-with-body -sS -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d "$INTAKE" $BASE/api/v1/campaigns/validate
# {"schema_version":2,"valid":true,"errors":[],"warnings":[],
#  "spec_summary":{"name":"Manpage demo","n_parameters":1,"n_objectives":1,...}}

# 3. Create, with the saved idempotency key
CID=$(curl --fail-with-body -sS -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -H "Idempotency-Key: $CREATE_KEY" -d "$INTAKE" \
  $BASE/api/v1/campaigns | json "['campaign_id']")
echo "campaign: $CID"
# {"schema_version":2,"success":true,"campaign_id":"1e5480ef-...","spec_id":"...",
#  "warnings":[],"errors":[],"error":null,"idempotency_replay":false,...}

# 3b. Retry-safety proof: the SAME key + byte-identical payload replays the
#     original response instead of creating a second campaign.
curl --fail-with-body -sS -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -H "Idempotency-Key: $CREATE_KEY" -d "$INTAKE" $BASE/api/v1/campaigns
# {...,"campaign_id":"1e5480ef-...(same id)","idempotency_replay":true,...}

# 4. Ask the server what to do next
curl --fail-with-body -sS -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d "{\"campaign_ids\":[\"$CID\"],\"verbosity\":\"minimal\"}" \
  $BASE/api/v1/campaigns/status/batch
# {"schema_version":2,"success":true,"campaigns":{"1e5480ef-...":{
#   "name":"Manpage demo","status":"created","iteration":0,"n_results":0,
#   "next_action_recommendation":{"action":"bo_generate_suggestions",
#     "reason":"No results yet — generate initial suggestions to start
#      optimization.","urgency":"normal"}}},"failed_ids":[],"errors":[],...}

# 5. Generate a suggestion batch (auto-transitions created -> running)
GEN=$(curl --fail-with-body -sS -X POST -H "X-API-Key: $KEY" \
  "$BASE/api/v1/suggestions/$CID/generate?batch_size=1")
SID=$(echo "$GEN" | json "['suggestions'][0]['suggestion_id']")
XVAL=$(echo "$GEN" | json "['suggestions'][0]['parameter_values']['x']")
echo "suggestion: $SID  x=$XVAL"
# {"schema_version":2,"success":true,"suggestions":[{"suggestion_id":"246b2534-...",
#   "campaign_id":"1e5480ef-...","parameter_values":{"x":4.0290},
#   "status":"pending","provenance":{"iteration":1,
#     "generation_method":"initial_design",...}}],...}

# 6. Evaluate externally (your code), then
# 7. Submit the measurement with the saved key, linked to the suggestion
RESULTS="{\"results\":[{\"parameter_values\":{\"x\":$XVAL},\"objective_values\":{\"y\":12.3},\"suggestion_id\":\"$SID\"}],\"source\":\"api\"}"
curl --fail-with-body -sS -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -H "Idempotency-Key: $SUBMIT_KEY" -d "$RESULTS" $BASE/api/v1/results/$CID
# {"schema_version":2,"success":true,"result_ids":["aa5ded40-..."],"errors":[],
#  "warnings":[],"field_errors":{},"error":null,"idempotency_replay":false,...}

# 7b. Uncertain whether the submit landed? Retry with the SAME key:
curl --fail-with-body -sS -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -H "Idempotency-Key: $SUBMIT_KEY" -d "$RESULTS" $BASE/api/v1/results/$CID
# {...,"result_ids":["aa5ded40-...(same id)"],"idempotency_replay":true,...}

# 8. End-of-invocation diagnostics (generous timeout on grown campaigns)
curl --fail-with-body -sS -H "X-API-Key: $KEY" "$BASE/api/v1/diagnostics/$CID?verbosity=minimal"
# {"schema_version":2,"success":true,"iteration":1,"n_results":1,
#  "status":"running","health":"healthy","progress":"stable",
#  "key_metric":{"best_value":12.3},"converged":false,
#  "next_action":{"action":"bo_generate_suggestions","reason":"Campaign healthy
#   with 1 results. Ready for next batch of suggestions.","urgency":"normal"},...}

# 9. Pause — the campaign continues in a later invocation with action=resume
curl --fail-with-body -sS -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"action":"pause"}' $BASE/api/v1/campaigns/$CID/lifecycle
# {"schema_version":2,"success":true,"campaign_id":"1e5480ef-...",
#  "status":"paused","previous_status":"running","errors":[],"error":null,...}
```

## 3. State ownership and continuation

This section is the contract between BO-MCP and every client that drives a
campaign loop. Violating it produces campaigns that cannot be resumed,
duplicated experiments, and split-brain progress tracking.

### 3.1 The server is the single source of truth

BO-MCP owns campaign progress: status, results, the iteration counter, and
the next-action recommendation. Clients must not persist loop state — no
`campaign_state.json`, no local iteration counters, no exhaustion flags, no
"done" markers. Ask the server
(`POST /api/v1/campaigns/status/batch`) every time; branch on the returned
`action`, never on remembered state. Two processes that both follow this rule
can operate the same campaign without coordinating with each other.

### 3.2 Invocation budgets are not campaign budgets

A per-process bound ("run at most 20 evaluations tonight",
`--max-successes`, a CLI loop cap) limits **one invocation** and lives in
the process, never in the campaign. The intake is **immutable**: encoding a
per-invocation budget as the intake's `max_iterations` fossilizes it — a
reopened campaign then refuses suggestions forever because its cap was
reached long ago. Leave `max_iterations` unset unless the user names a
whole-campaign budget.

### 3.3 Resuming an interrupted run

A killed, crashed, or paused run re-attaches by campaign id and re-derives
its position from the server. Give every campaign entrypoint an optional
`--campaign-id` argument; when present, skip creation and enter the loop
directly — step 4 of [§2.1](#21-the-ten-steps) tells you where the campaign
stands. Then reuse before generating: query
`POST /api/v1/suggestions/{campaign_id}/query` with
`status_filter="pending"` and evaluate those first — an interrupted run may
have generated suggestions it never evaluated, and generating a fresh batch
on top of them wastes experiments
([§4.7](#47-recovery-walk-throughs)).

### 3.4 Pause, resume, reopen, terminate

- **End of invocation → pause.** A paused campaign is a bookmark, not a
  failure.
- **Continue a paused campaign → `resume`.**
- **Continue a completed campaign → `reopen`.** That is the supported "run
  another N experiments" path.
- **Terminate only on explicit request.** Termination marks the campaign
  `completed`; it is the "accept the current best and close out" action.
- **Never rebuild a campaign by replaying its results as seeds.** Creating
  a fresh campaign and re-submitting old results as new measurements
  destroys provenance (iteration history, suggestion links, audit trail)
  and biases the model's noise estimates with duplicated observations.
  Resume or reopen instead — the server keeps all history.

### 3.5 Artifacts are provenance, not state

Result JSONL files, exports (`GET /api/v1/campaigns/{campaign_id}/export`),
diagnostics history, reports: write them freely, append-only, for analysis
and figures. The loop must never read them back to decide what to do next —
that decision belongs to [§3.1](#31-the-server-is-the-single-source-of-truth).

### 3.6 Failed experiments are information, not always penalties

Do not hard-penalize an isolated failed experiment (e.g. by fabricating a
terrible objective value). Isolated failures are usually noise — equipment
hiccups, transient numerical issues. Reject the suggestion
(`POST /api/v1/suggestions/{suggestion_id}/status`, status `rejected`) and
move on — rejection does not fence off the region, so the optimizer may
revisit those coordinates later, which is exactly right for a transient
failure ([§1.4](#14-suggestion-status-machine)). Only when failures are
*systematic* in a region of the search space
do they carry signal worth encoding — and then prefer declaring the region
infeasible via intake constraints on a new campaign, or discuss with the
campaign owner, over inventing sentinel objective values.

## 4. Reliability and error handling

### 4.1 Idempotency keys

Mutations that must not double-apply accept an `Idempotency-Key` request
header (REST) or `idempotency_key` parameter (MCP tools):

- `POST /api/v1/campaigns` — campaign creation
- `POST /api/v1/suggestions/{campaign_id}/generate` — suggestion generation
- `POST /api/v1/results/{campaign_id}` — result submission

Semantics:

- Use a fresh UUID per logical operation; retry the **byte-identical**
  payload with the **same** key.
- A retry with the same key and same payload replays the original response
  verbatim, flagged `idempotency_replay: true`.
- The same key with a *different* payload is rejected (`E015`,
  idempotency conflict): generate a fresh key for a genuinely new request.
- Rejections are cached too. Concretely for result submission: a
  duplicate-result rejection (`E004`) is stored under the submitted key as
  a terminal outcome, and `force` participates in the request hash — so
  following the rejection's "use `force=true`" recovery hint requires a
  **fresh** key. Reusing the rejected key returns `E015` instead of
  running the forced submission.
- While the original call is still executing, a concurrent retry gets
  `E014` (in progress): wait briefly and retry with the same key.
- The cache namespace is **shared across REST and MCP** — a retry on either
  transport sees the other's prior response.

### 4.2 Transport errors versus operation rejections

Two failure planes, and you must handle both:

1. **Transport error** — non-2xx status: auth failures (401/403), malformed
   requests (400/422), not found (404), oversized payloads (413), server
   faults (5xx).
2. **Operation rejection** — **2xx with `success: false`** in the body: the
   request was well-formed and processed, but the operation was refused
   (intake incompatible with the backend, invalid state transition, model
   fitting failure...).

Checking only the HTTP status code silently hides the second plane. Always
inspect the body's `success` field; on `false`, read `errors` and the
structured `error` object — its `retryable` and `retry_after` fields tell
you whether backoff-and-retry has any chance of working, and
`recovery_action` says what to do instead.

Where the structured `error` object lives, per transport:

- **MCP** — every failure envelope carries it at the top level.
- **REST, 2xx rejection** — mutation responses (create, generate, submit,
  upload, suggestion status) carry an `error` key: the structured object
  on rejection, `null` on success.
- **REST, HTTP-status errors** — rejections promoted to an error status
  (lifecycle conflicts, idempotency conflict/in-progress, invalid state)
  return `{"detail": ...}` where `detail` is the structured error object;
  plain transport failures (auth, malformed request, not found) return
  `{"detail": "<message>"}` with a string detail.

### 4.3 Validation failures and field errors

Validation rejections carry a `field_errors` list with dotted-path locations
(`intake.parameters.2.bounds`) so an agent can fix the exact offending field
and resend. Use `POST /api/v1/campaigns/validate` to surface all of them
before creating anything.

### 4.4 Concurrent modification and version conflicts

Every campaign mutation uses optimistic concurrency: a `version` counter
increments on each successful write. A writer that lost the race receives
`E010` (concurrent modification) with `expected_version` / `actual_version`
in `details` and a `retry_after` hint. Recovery: **re-read, rebuild, retry**
— fetch the current campaign state, rebuild the mutation against what you
learned, wait `retry_after`, then retry. Do not blindly retry the original
request; it will hit the same conflict. Multi-step orchestrations should
also re-fetch any cached suggestion or result lists touched by the same
transaction.

### 4.5 Error-code catalog

Every structured error carries one of these codes, plus a `recovery_action`
string, a `retryable` flag, and an optional `retry_after`. The
recovery column below summarizes the server's own guidance.

| Code | Name | Retry? | Recovery |
|---|---|---|---|
| E001 | Invalid campaign id | no | The id is not a valid UUID. List campaigns to recover the real id. |
| E002 | Campaign not found | no | List campaigns (`GET /api/v1/campaigns`) to verify the id; recover from a mistyped or hallucinated id. |
| E003 | Invalid state transition | no | Read the campaign's current status; pick a legal action from the matrix in [§1.3](#13-campaign-lifecycle-state-machine). |
| E004 | Duplicate result | no | Intentional replicate → re-submit with `force=true`: JSON submit needs a fresh idempotency key ([§4.1](#41-idempotency-keys)); REST upload takes `?force=` with no key involved; MCP upload has no `force` — switch to `bo_submit_results`. Accidental → drop the row. |
| E005 | Validation failed | no | Inspect `errors` / `field_errors`, fix the named fields, resend. |
| E006 | Missing parameters | no | Intake needs at least one entry in `parameters`. |
| E007 | Missing objectives | no | Intake needs at least one entry in `objectives`. |
| E008 | Constraint violation | no | Check that constraint parameters exist in the spec and leave a feasible region. |
| E009 | Suggestion not found | no | Re-list suggestions; only `pending`/`accepted` rows are mutable ([§1.4](#14-suggestion-status-machine)). |
| E010 | Concurrent modification | yes | Re-read → rebuild → retry after `retry_after` ([§4.4](#44-concurrent-modification-and-version-conflicts)). |
| E011 | Search space exhausted | no | A finite (typically all-categorical) space has no unseen combinations; terminate the campaign. |
| E012 | Budget exceeded | no | The campaign's configured iteration/observation budget is reached. Terminate, or continue under a new campaign with a larger budget — and see [§3.2](#32-invocation-budgets-are-not-campaign-budgets) for why this cap should rarely exist. |
| E013 | Campaign converged | no | Accept the current best (terminate), or submit a fresh campaign if a better optimum is plausible. |
| E014 | Idempotency in progress | yes | The original call with this key is still running; wait briefly, retry with the same key. |
| E015 | Idempotency conflict | no | Same key, different payload. Use a fresh key for the new request. |
| E101 | Model fitting failed | no | Check data quality via diagnostics; drop NaN/Inf rows, dampen extreme outliers, or submit more clean observations. |
| E102 | Acquisition optimization failed | no | Reduce `batch_size` or check for conflicting constraints. |
| E103 | Database error | yes | Retry; if persistent, the deployment's database is misconfigured or down. |
| E104 | Insufficient data | no | Model-based generation needs ≥ 2 valid observations; submit more results first. |
| E105 | Backend transient error | yes | Numerical hiccup or resource pressure; retry after the suggested backoff. |
| E106 | Backend incompatibility | no | The selected backend cannot handle this spec; switch backend or remove the unsupported feature. |
| E107 | Backend internal error | maybe | Retry once to confirm reproducibility; report persistent failures to the operator. |
| E108 | Data integrity error | no | A persisted row is corrupt; retrying is futile. Give the `request_id` to an operator. |
| E109 | Backend state too large | no | Deterministic: shrink the enumerated search space, switch encoding/backend, or raise the server's state-size limit. |
| E199 | Internal error | maybe | Unclassified server fault; retry once, then report with the `X-Request-ID`. |

### 4.6 Request correlation

- `X-Request-ID` — identifies one HTTP request. Generated server-side when
  absent, always echoed on the response, and attached to server logs.
  Quote it when reporting a failure.
- `X-Trace-Id` — an opaque workflow id you mint **once per multi-step
  workflow** and attach to every request in it. The server binds it for the
  request, echoes it back, records it in audit events, and stamps it into
  response `_metadata`. On the MCP transport the same trace context is
  honored when the host binds it around tool calls.

### 4.7 Recovery walk-throughs

**Interrupted between generate and submit** (process died after suggestions
were created): re-attach with the campaign id, query pending suggestions
(`POST /api/v1/suggestions/{campaign_id}/query`, `status_filter="pending"`),
evaluate those, submit. Do not generate a new batch first.

**Unsure whether a submit landed** (timeout mid-request): retry the same
`POST /api/v1/results/{campaign_id}` with the **same** `Idempotency-Key` and
byte-identical payload. Either it lands once, or the cached response replays
with `idempotency_replay: true`. Never re-submit with a fresh key "to be
safe" — that is how duplicate observations corrupt the model.

**Interrupted diagnostics or lifecycle call**: both are safe to re-issue.
Diagnostics is read-only; lifecycle transitions are guarded by the state
machine, so a repeated `pause`/`resume` either succeeds or returns `E003`,
which tells you the state already advanced — re-read the status and
continue.

**Campaign id lost**: `GET /api/v1/campaigns` (or the MCP resources
`campaigns://list` / `campaigns://recent`) and match on name/created-at.

## 5. Operational guidance

### 5.1 Next-action versus diagnostics, and diagnostics cost

They answer different questions at very different prices:

- **Next-action** (`POST /api/v1/campaigns/status/batch`, minimal
  verbosity) is the cheap loop-control signal — status, iteration, and the
  recommendation. Call it every iteration.
- **Diagnostics** (`GET /api/v1/diagnostics/{campaign_id}`) recomputes model
  health, convergence, and front/best-so-far **from all results**, so its
  cost grows with the campaign. Call it at invocation boundaries (start
  and/or end), not per iteration, and give the call a generous timeout —
  minutes on a grown campaign are expected.
- The server caches diagnostics for **120 seconds** (invalidated
  automatically when results are submitted or suggestions generated); pass
  `use_cache=false` only when you must force a fresh computation.

### 5.2 Capability discovery and backend options

`GET /api/v1/capabilities` advertises the available backends (BayBE,
BoTorch) and what each supports. Backend-specific intake options — per
parameter (`parameters[].parameter_options`) and per campaign
(`backend_options`) — have typed schemas spliced into the OpenAPI document,
discoverable at [/docs](/docs). Build intakes from what is advertised, and
validate before creating; do not infer unsupported request shapes.

### 5.3 Parallel evaluation and batch operations

- Independent **pending** suggestions from one batch can be evaluated in
  parallel; submit results as they arrive or together. Request a batch
  sized to your parallel capacity (`batch_size` on generate).
- Result submission is batched: `atomic=true` (default) makes the whole
  batch succeed or fail together; `atomic=false` with
  `continue_on_error=true` processes each row independently and reports a
  per-row `partial_results` mapping.
- Bulk historical backfills go through
  `POST /api/v1/results/{campaign_id}/upload` (CSV/XLSX) instead of huge
  JSON bodies.

### 5.4 Verbosity and token budgets

Most read/query operations accept `verbosity`:

| Verbosity | ~Tokens | Use case |
|---|---|---|
| `minimal` | ~50 | Tight loops, monitoring many campaigns. Retains `next_action_recommendation`. |
| `standard` | ~200 | Normal operation and debugging (the default on most operations). |
| `detailed` | 500+ | Deep debugging, model behavior analysis, full provenance. |

Agents in tight loops should pass `minimal` explicitly; batch status
defaults to `minimal` because it exists to monitor many campaigns at once.

### 5.5 Exports are raw bytes

`GET /api/v1/campaigns/{campaign_id}/export` (with `?format=csv`) returns a
downloadable file — raw bytes with a `Content-Disposition` header, not an
envelope. Never JSON-parse an export response. Everything else in the API
speaks JSON.

### 5.6 Initial design and convergence

**Initial design.** Before a model can be fit, the campaign explores with a
Sobol sequence. The default initial design size is `2 × n_parameters + 1`,
overridable in the intake (`initial_design_size`):

| Parameters | Default initial points | Recommendation |
|---|---|---|
| 2–5 | 5–11 | Usually sufficient. |
| 6–10 | 13–21 | Add 5–10 more for noisy objectives. |
| 11–20 | 23–41 | Trust-region methods engage automatically at high dimension. |
| > 20 | 41+ | Sparse-axis methods engage automatically for very high dimension. |

Increase the initial design when objectives are noisy, many local optima are
expected, categorical parameters have many categories, or fidelity levels
need diverse coverage.

**Convergence.** Detection needs at least 10 observations; before that,
diagnostics report insufficient data rather than a verdict. A
single-objective campaign is considered converged when the improvement rate
stays below 1% across the recent window (5 iterations); multi-objective
campaigns use hypervolume stability and a stable Pareto front instead — there
is no single best point, so present the trade-offs.

**Early convergence warning.** If diagnostics report `converged=true` with
fewer than ~20 results, warn the operator before stopping: it may mean a
genuinely simple objective (good), a local optimum (consider a fresh
campaign with a different seed), or an over-constrained search space
(review the constraints). Confirm with `detailed` diagnostics.

### 5.7 Troubleshooting quick table

| Symptom | Check |
|---|---|
| Cannot generate suggestions | Campaign status first: `paused` → resume; `completed` → reopen; `failed` → new campaign. Status fine? With < 2 results generation uses Sobol sampling and should work — re-check intake validation warnings. With ≥ 2 results, a fitting problem: see E101/E104 in [§4.5](#45-error-code-catalog). |
| Suggestions repeat or campaign refuses to continue | You may be replaying results into a fresh campaign, or a fossilized `max_iterations` cap is set ([§3.2](#32-invocation-budgets-are-not-campaign-budgets)). |
| Duplicate result rejected (E004) | Intentional re-measurement → `force=true`; the mechanics differ per transport — see the E004 row in [§4.5](#45-error-code-catalog). Accidental → drop the row. |
| Results accepted but suggestions stay `pending` | Rows submitted without `suggestion_id` are stored unlinked and complete nothing. Copy each suggestion's `suggestion_id` into its result row ([§1.4](#14-suggestion-status-machine)). `source="api"` submissions get a response warning when open suggestions exist; GUI and file-upload rows are legitimately unlinked, so no warning fires there. |
| Model fitting failed (E101) | NaN/Inf rows? Extreme outliers? < 2 valid observations? Diagnostics names the culprit. |
| Version conflict (E010) | Another writer won the race: re-read, rebuild, retry ([§4.4](#44-concurrent-modification-and-version-conflicts)). |
| Converged suspiciously early (few results) | May be a simple objective (fine), a local optimum, or an over-constrained space. Confirm with `detailed` diagnostics before stopping. |
| 401 / 403 | Missing or wrong `X-API-Key`; or the key's user does not own this campaign. |
| Slow diagnostics | Expected on grown campaigns ([§5.1](#51-next-action-versus-diagnostics-and-diagnostics-cost)); raise your client timeout, don't cancel-and-retry in a loop. |

### 5.8 MCP transport specifics

Everything above applies to MCP tools unchanged (same operations, same
envelopes, same idempotency cache). MCP additionally offers:

- **Resources vs tools.** Tools act and query with filters; resources are
  cheap read-only lookups by URI: `campaign://{campaign_id}` (details),
  `campaigns://list`, `campaigns://recent` (id recovery),
  `suggestions://{campaign_id}` (pending suggestions),
  `suggestion://{suggestion_id}`, `events://{campaign_id}` (audit trail),
  and `docs://manpage` (this manual).
- **Resource subscriptions.** The server implements `resources/subscribe`:
  subscribe to `campaign://{campaign_id}` and receive a
  `notifications/resources/updated` push on every campaign **status**
  transition, then re-read the resource. Not pushed: iteration bumps,
  result submissions, rejected transitions. Delivery is best-effort — a
  subscriber whose transport drops is silently removed; re-subscribe to
  re-arm. Unsubscribe when the campaign reaches a terminal state.
- **Per-tool verbosity defaults.** Mutating and query tools default to
  `standard`; `bo_batch_get_status` defaults to `minimal`;
  `bo_health_check` / `bo_list_capabilities` have no verbosity knob. In
  loops, pass `minimal` explicitly.
- **Enum completion.** The server implements `completion/complete` for
  enum-valued arguments (`status`, `action`, `backend`, …) — ask it rather
  than guessing valid values.
- **Minimal SDK usage:**

```python
from mcp import ClientSession

async def one_round(session: ClientSession, campaign_id: str) -> None:
    status = await session.call_tool(
        "bo_batch_get_status",
        {"campaign_ids": [campaign_id], "verbosity": "minimal"},
    )
    # branch on next_action_recommendation.action, then e.g.:
    await session.call_tool(
        "bo_generate_suggestions",
        {"campaign_id": campaign_id, "verbosity": "minimal"},
    )
    manual = await session.read_resource("docs://manpage")  # this text
```

## 6. Endpoint map

Method + path, the matching MCP tool, and a one-line purpose. Request and
response field shapes live in the OpenAPI document at [/docs](/docs) — not
here. Rows marked *(no MCP tool)* are REST-only conveniences; MCP-side
lookups use the resources named in [§5.8](#58-mcp-transport-specifics).

### 6.1 Campaigns

| Endpoint | MCP tool | Purpose |
|---|---|---|
| `POST /api/v1/campaigns/validate` | `bo_validate_intake` | Dry-run intake validation; returns field errors, creates nothing. |
| `POST /api/v1/campaigns` | `bo_create_campaign` | Create a campaign (idempotent via `Idempotency-Key`). |
| `GET /api/v1/campaigns` | `bo_list_campaigns` | List campaigns. |
| `POST /api/v1/campaigns/query` | `bo_list_campaigns` | Filtered campaign query (status, owner, limits). |
| `POST /api/v1/campaigns/status/batch` | `bo_batch_get_status` | Status + `next_action_recommendation` for many campaigns — the loop-control call. |
| `POST /api/v1/campaigns/compare` | `bo_compare_campaigns` | Side-by-side comparison of campaign performance. |
| `POST /api/v1/campaigns/{campaign_id}/lifecycle` | `bo_pause_campaign`, `bo_resume_campaign`, `bo_terminate_campaign`, `bo_reopen_campaign` | Lifecycle actions ([§1.3](#13-campaign-lifecycle-state-machine)). |
| `POST /api/v1/campaigns/{campaign_id}/transfer-candidates` | `bo_discover_transfer_candidates` | Find prior campaigns whose data can warm-start this one. |
| `GET /api/v1/campaigns/{campaign_id}/export` | `bo_export_campaign` | Download campaign results as a file ([§5.5](#55-exports-are-raw-bytes)). |
| `GET /api/v1/campaigns/{campaign_id}` | *(no MCP tool — resource `campaign://{campaign_id}`)* | Campaign record: status, iteration, version. |
| `GET /api/v1/campaigns/{campaign_id}/config` | *(no MCP tool — resource `campaign://{campaign_id}`)* | Sanitized campaign setup snapshot. |
| `GET /api/v1/campaigns/spec/{spec_id}` | *(no MCP tool)* | The immutable intake spec as stored. |

### 6.2 Suggestions

| Endpoint | MCP tool | Purpose |
|---|---|---|
| `POST /api/v1/suggestions/{campaign_id}/generate` | `bo_generate_suggestions` | Generate the next candidate batch (idempotent via `Idempotency-Key`). |
| `POST /api/v1/suggestions/{campaign_id}/query` | `bo_list_suggestions` | Query suggestions with `status_filter` — the resume path ([§3.3](#33-resuming-an-interrupted-run)). |
| `GET /api/v1/suggestions/{campaign_id}` | `bo_list_suggestions` | List a campaign's suggestions. |
| `POST /api/v1/suggestions/{suggestion_id}/status` | `bo_update_suggestion_status` | Accept / reject / expire a suggestion ([§1.4](#14-suggestion-status-machine)). |
| `GET /api/v1/suggestions/{suggestion_id}/explanation` | `bo_get_suggestion_explanation` | Why the model proposed this candidate. |

### 6.3 Results

| Endpoint | MCP tool | Purpose |
|---|---|---|
| `POST /api/v1/results/{campaign_id}` | `bo_submit_results` | Submit measurements (idempotent via `Idempotency-Key`; atomic or per-row). |
| `POST /api/v1/results/{campaign_id}/upload` | `bo_upload_results_file` | Bulk upload historical results (CSV/XLSX). |
| `POST /api/v1/results/{campaign_id}/query` | `bo_list_results` | Filtered result query. |
| `GET /api/v1/results/{campaign_id}` | `bo_list_results` | List a campaign's results. |

### 6.4 Diagnostics

| Endpoint | MCP tool | Purpose |
|---|---|---|
| `GET /api/v1/diagnostics/{campaign_id}` | `bo_get_diagnostics` | Model health, convergence, best-so-far / Pareto front — expensive ([§5.1](#51-next-action-versus-diagnostics-and-diagnostics-cost)). |

### 6.5 Capabilities and service

| Endpoint | MCP tool | Purpose |
|---|---|---|
| `GET /api/v1/capabilities` | `bo_list_capabilities` | Backends and their supported features — read before building an intake. |
| `GET /health` | `bo_health_check` | Liveness + database connectivity (public). |

MCP-only helpers with no REST twin: `bo_check_progress` (compact progress
summary) and the resources listed in
[§5.8](#58-mcp-transport-specifics).
