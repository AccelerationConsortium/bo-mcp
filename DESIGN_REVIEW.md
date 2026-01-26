# Bayesian Optimization MCP Service — Design Review Document

**Version**: 2.1
**Status**: v1.0.1 + v1.1 + v1.2 + v1.3 + v2.0 + v2.1 Complete | Implementation Documented
**Last Updated**: 2026-01-15
**Scope**: System architecture, responsibilities, contracts, and evolution strategy

**Related Documentation:**
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Practical implementation details, configuration, and technical patterns

### Recent Updates (v2.1 - 2026-01-15)
- ✅ `upload_results_file` tool implemented
- ✅ `get_suggestion_explanation` tool implemented
- ✅ `pause_campaign` / `resume_campaign` / `terminate_campaign` tools implemented
- ✅ `bo_mcp_server` package restructured for standalone PyPI installation
- ✅ Constants module added to `bo_engine` for configurable thresholds
- ✅ TuRBO warning added for multi-objective optimization
- ✅ Scripts import paths updated to use `bo_mcp_server` package

---

## 0a. Stakeholder Requirements (Clarified)

This section documents requirements gathered through stakeholder Q&A. These are **locked decisions** that constrain the implementation.

### Deployment & Infrastructure

| Requirement | Decision | Notes |
|-------------|----------|-------|
| **Deployment model** | Containerized services | Must deploy to both AWS and on-premises servers |
| **Container rationale** | Confirmed appropriate | Environment consistency across deployment targets; MCP + GUI map to separate containers |
| **GPU requirements** | Not required for MVP | BO computation is CPU-based; GPU acceleration deferred |

### Users & Collaboration

| Requirement | Decision | Notes |
|-------------|----------|-------|
| **User expertise** | Mixed audience | Domain scientists (chemists, biologists, materials scientists) AND data scientists |
| **Collaboration model** | Full multi-user | Multiple users can submit results, request suggestions on same campaign |
| **Conflict resolution** | Optimistic locking | Campaign version tracked; concurrent edits trigger clear conflict notification with refresh prompt |
| **Authentication (MVP)** | Simple/minimal | Basic auth sufficient for MVP |
| **Authentication (later)** | SSO/SAML + OAuth | Enterprise integration required post-MVP |

### Integration Requirements

| Requirement | Decision | Notes |
|-------------|----------|-------|
| **External systems** | Multiple | LIMS, ELN, robotic platforms |
| **Result ingestion** | Three methods | Manual entry via GUI, file upload (CSV/Excel), AND API push from external systems |
| **Historical data** | Important for v1 | Not required for MVP; warm-start from existing data needed soon after initial release |

### Scale & Performance

| Requirement | Decision | Notes |
|-------------|----------|-------|
| **Concurrent campaigns** | Few (1-5) | Small scale initially; architecture should not preclude growth |
| **Experimental throughput** | Low (1-5/day) | Manual/semi-manual experiments; not high-frequency automation |
| **Campaign duration** | Days to weeks | Short campaigns with relatively quick iteration cycles |
| **Suggestion latency** | Fast as possible | Understood that large data + large batches will be slower; acceptable |

### MVP Scope (Feature-Complete Before First Deploy)

| Requirement | Decision | Notes |
|-------------|----------|-------|
| **Optimization type** | Multi-objective required | 2-4 objectives typical; Pareto front tracking from day one |
| **Input parameter types** | All types | Continuous, discrete, AND categorical |
| **Constraint complexity** | Sum/mixture constraints | Parameters summing to 1 (mixture formulations) |
| **Input dimensionality** | Medium (6-15 parameters) | Moderate complexity design spaces |
| **MVP priority** | Feature completeness | All listed features required before first deployment |
| **Compliance** | None/minimal | Standard security practices sufficient |

### Design Implications from Requirements

1. **Multi-objective is not optional** — Pareto front tracking, hypervolume diagnostics, and trade-off visualization are core MVP features, not v1 additions

2. **Collaboration adds architectural complexity** — User identity tracking, optimistic locking, activity attribution in audit logs required from start

3. **Three ingestion paths multiply validation surface** — Each method (GUI, file, API) needs consistent validation, error handling, and audit trail

4. **Mixed parameter types require careful encoding** — Categorical parameters need one-hot or similar encoding; affects model engine and CampaignSpec structure

5. **Original milestone plan superseded** — MVP scope is closer to original v1.0; plan revised below

---

## 0. Executive Summary (Implementation-Oriented)

### Design Summary (15 Bullets)

1. **MCP server is the single source of truth** for all BO logic, model state, and suggestion generation — the GUI never performs optimization calculations.
2. **Campaign lifecycle** is strictly ordered: Intake → Validation → CampaignSpec → Initialization → Iteration Loop → Termination/Archive.
3. **All suggestions are immutable and auditable** — once generated, a suggestion's provenance (model version, seed, settings) is permanently recorded.
4. **Separation of concerns**: GUI handles human interaction and visualization; MCP handles decision logic; a thin integration layer handles transport only.
5. **Resources follow append-only or immutable patterns** wherever possible to support reproducibility and audit requirements.
6. **Diagnostics are first-class citizens** — every suggestion includes confidence metrics, and model health is continuously assessed.
7. **Deterministic replay is a design goal** — with explicit documentation of where and why it cannot be guaranteed.
8. **Intake validation is fail-fast** — malformed or dangerous configurations are rejected before campaign creation.
9. **Multi-objective optimization requires distinct diagnostic approaches** — Pareto progress replaces scalar improvement tracking.
10. **Constraints are modeled explicitly** — feasibility, rejection, and constraint violation are distinct concepts with different system behaviors.
11. **Model state and audit state are separate concerns** — audit state is immutable; model state may evolve but is versioned.
12. **Controls and replicates are explicitly tracked** — they are not optimized but are essential for experimental validity.
13. **The system exposes "why" alongside "what"** — users can always understand the reasoning behind suggestions.
14. **Evolution is planned** — multi-fidelity, cost-aware BO, and other advanced features are explicitly deferred with reversible hooks.
15. **MVP is feature-complete** — multi-objective optimization, all parameter types, mixture constraints, and full collaboration are required before first deployment (see Section 0a).

### Top 5 Non-Obvious Design Decisions

| # | Decision | Why It Matters |
|---|----------|----------------|
| 1 | **Intake ≠ CampaignSpec** — Intake is user-facing, CampaignSpec is system-internal with derived quantities and validated defaults | Prevents user-facing simplicity from polluting internal consistency requirements |
| 2 | **Suggestions are immutable resources, not ephemeral responses** — each suggestion is a permanently addressable, versioned artifact | Required for reproducibility, audit, and deterministic replay |
| 3 | **Model quality is exposed as user-facing confidence, not GP internals** — uncertainty is translated to actionable language | Users need to know "is BO helping?" without understanding acquisition functions |
| 4 | **Constraint violation vs. feasibility vs. rejection are three distinct states** | "Infeasible" (model believes unlikely), "violated" (measured), and "rejected" (user override) have different downstream effects |
| 5 | **Diagnostic health checks can trigger "BO ineffective" warnings** — the system can recommend stopping optimization | Users deserve to know when they're wasting experimental resources on random-walk behavior |

---

## 0b. Key Design Decisions and Configuration

This section documents fundamental design decisions and configuration parameters that shape the BO engine's behavior.

### Algorithm Selection Design Decisions

#### TuRBO and Multi-Objective

TuRBO (Trust Region Bayesian Optimization) is only supported for single-objective optimization. When TuRBO is requested for multi-objective problems, a warning is issued and standard L-BFGS-B optimization is used instead.

**Rationale**: TuRBO maintains a trust region state that contracts/expands based on improvement. This state tracking is well-defined for scalar objectives but ambiguous for Pareto improvement in multi-objective settings.

#### Acquisition Method Selection

When `AcquisitionMethod.AUTO` is specified:
- **Single-objective**: Uses `qLogNEI` (or `EIpu` if cost-aware)
- **Multi-objective**: Uses `qLogNEHVI` (or `qLogNParEGO` if specified)

**Rationale**: These defaults balance exploration/exploitation effectively for typical optimization problems. qLogNEI handles noisy observations well for single-objective; qLogNEHVI provides efficient hypervolume improvement for multi-objective.

#### Model Type Selection

| Problem Type | Model | Rationale |
|--------------|-------|-----------|
| Single-objective | `SingleTaskGP` | Simpler, faster than ModelListGP when m=1 |
| Multi-objective | `ModelListGP` (list of `SingleTaskGP`) | Independent modeling of each objective |
| High-dimensional (>20 params) | TuRBO with `SingleTaskGP` | Trust region prevents over-exploration in high-D |
| Very high-dimensional (>50 params) | SAASBO with sparse priors | Automatic feature selection via sparsity |
| Multi-fidelity | `SingleTaskMultiFidelityGP` | Correlates observations across fidelity levels |
| Transfer learning | RGPE ensemble | Leverages prior campaign knowledge |

### BO Engine Constants

All configurable thresholds are defined in `bo_engine/constants.py`. These values represent carefully chosen defaults based on BoTorch recommendations and practical experience.

#### Dimensionality Thresholds

| Constant | Value | Rationale |
|----------|-------|-----------|
| `HIGH_DIM_THRESHOLD` | 20 | Standard BO with L-BFGS-B struggles above ~20 parameters due to acquisition optimization challenges |
| `VERY_HIGH_DIM_THRESHOLD` | 50 | SAASBO's sparse priors become necessary when standard GP lengthscale estimation fails |

#### Model Quality Thresholds

| Constant | Value | Rationale |
|----------|-------|-----------|
| `MIN_DATA_FOR_BO` | 2 | Minimum observations before GP fitting is meaningful (need variance) |
| `STAGNATION_THRESHOLD_DEFAULT` | 5 | Iterations without improvement before warning user; balances patience with feedback |
| `MODEL_CORRELATION_WARNING_THRESHOLD` | 0.3 | Below this, GP predictions poorly correlate with observed values |

#### Confidence Level Thresholds

| Constant | Value | Rationale |
|----------|-------|-----------|
| `UNCERTAINTY_HIGH_THRESHOLD` | 0.1 | CV below 10% indicates high confidence in prediction |
| `UNCERTAINTY_MEDIUM_THRESHOLD` | 0.3 | CV below 30% is medium confidence; above is low |

#### TuRBO Defaults

| Constant | Value | Rationale |
|----------|-------|-----------|
| `TURBO_INITIAL_LENGTH` | 0.8 | Start with 80% of normalized space; conservative but not too restrictive |
| `TURBO_LENGTH_MIN` | 0.5^7 | ~0.008; triggers restart when trust region becomes too small |
| `TURBO_LENGTH_MAX` | 1.6 | Allows slight expansion beyond unit hypercube for boundary exploration |
| `TURBO_SUCCESS_TOLERANCE` | 10 | Successes needed before expansion; prevents premature expansion |

#### Acquisition Optimization

| Constant | Value | Rationale |
|----------|-------|-----------|
| `DEFAULT_NUM_RESTARTS` | 20 | Balance between optimization quality and compute time |
| `DEFAULT_RAW_SAMPLES` | 512 | Initial samples for multi-start optimization; sufficient for moderate dimensionality |

---

## 1. System Responsibilities & Boundaries

### 1.1 MCP Server Responsibilities

The MCP server is the **authoritative decision layer**. It owns:

- **Campaign lifecycle management**
  - Validate intake and create CampaignSpec
  - Initialize model state
  - Generate suggestions (initial and iterative)
  - Ingest experimental results
  - Terminate/archive campaigns

- **Bayesian Optimization logic**
  - Model fitting and hyperparameter management
  - Acquisition function selection and evaluation
  - Constraint handling and feasibility modeling
  - Multi-objective Pareto front maintenance

- **Suggestion generation with provenance**
  - Every suggestion includes: model version, random seed, acquisition settings, transform configurations
  - Suggestions are immutable once generated

- **Diagnostics and health monitoring**
  - Model quality metrics
  - Optimization progress indicators
  - "BO effectiveness" health checks
  - Early warning for ineffective optimization

- **Audit trail maintenance**
  - Complete history of all decisions
  - Sufficient state to attempt deterministic replay
  - Versioned snapshots at critical points

### 1.2 Web GUI Responsibilities

The GUI is the **human interaction layer**. It owns:

- **Intake collection**
  - Present structured forms matching intake document
  - Client-side validation (format, ranges, obvious errors)
  - Submission to MCP for authoritative validation

- **Campaign visualization**
  - Display current state, progress, Pareto fronts
  - Show suggestion history and result history
  - Render diagnostics in user-friendly language

- **User actions**
  - Request next suggestions
  - Submit experimental results
  - Mark suggestions as rejected/skipped
  - Request diagnostic reports
  - Archive/terminate campaigns

- **Transparency features**
  - Display provenance information for any suggestion
  - Show "why this suggestion" explanations
  - Provide access to settings/transforms used

### 1.3 Integration Layer (Thin, If Unavoidable)

If MCP cannot be directly consumed by the GUI:

- **Transport only** — no business logic
- **Stateless translation** — MCP resources/tools ↔ transport protocol
- **No caching of decision state** — all state lives in MCP
- **Validation passthrough** — does not duplicate MCP validation

### 1.4 What the MCP Server Must NOT Do

| Responsibility | Why Excluded |
|----------------|--------------|
| Render HTML/UI components | Separation of concerns; UI changes independently |
| Store user authentication/sessions | Security boundary; handled by infrastructure layer |
| Execute physical experiments | Out of scope; MCP generates suggestions, not commands |
| Make business decisions about experiment value | Domain remains with human; MCP advises, user decides |
| Persist data outside its defined resources | Audit integrity requires controlled persistence |

### 1.5 Why These Boundaries Matter for BO Systems

**Reproducibility requirement**: BO suggestions depend on exact model state, random seeds, and settings. If decision logic is distributed across components, reproducibility becomes impossible.

**Audit requirement**: Regulatory and scientific contexts demand complete provenance. A single authoritative source simplifies audit trails.

**Trust boundary**: Users must trust that suggestions are scientifically sound. Centralizing BO logic enables rigorous testing and validation of that single component.

**Evolution safety**: BO methods evolve rapidly. Isolating optimization logic allows method upgrades without GUI changes.

---

## 2. Core Concepts and Vocabulary

### 2.1 Canonical Definitions

#### Campaign
A **Campaign** is the complete lifecycle of an optimization effort:
- Created from a validated CampaignSpec
- Contains all iterations, suggestions, results, and model states
- Has defined states: created, running, paused, completed, failed
- Is the primary unit of audit and reproducibility

#### Intake vs. CampaignSpec

| Concept | Definition | Owner |
|---------|------------|-------|
| **Intake** | User-facing form data exactly as submitted; may contain ambiguities, implicit defaults, human-readable descriptions | GUI |
| **CampaignSpec** | System-internal, fully resolved configuration; all defaults explicit, all derived quantities computed, all validations passed | MCP |

**Key principle**: The transformation from Intake → CampaignSpec is explicit, logged, and auditable. Users can see exactly what defaults were applied.

#### Objective vs. KPI

| Concept | Definition | Role in Optimization |
|---------|------------|----------------------|
| **Objective** | A KPI that is being actively optimized (minimized or maximized) | Drives acquisition function |
| **KPI** | Any measurable outcome of interest; may or may not be optimized | Tracked and reported |

**Clarification**: All Objectives are KPIs, but not all KPIs are Objectives. A KPI might be tracked for diagnostics or reporting without being part of the optimization target.

#### Constraint vs. Feasibility vs. Rejection

| Concept | Definition | System Behavior |
|---------|------------|-----------------|
| **Constraint** | A rule that defines the valid region of input or output space | Modeled; influences suggestions |
| **Feasibility** | Model's belief about whether a point satisfies constraints | Probabilistic; affects acquisition |
| **Constraint Violation** | Measured outcome that violates a constraint | Recorded; updates model |
| **Rejection** | User's decision to exclude a suggestion regardless of feasibility | Recorded; does not update constraint model |

**Critical distinction**: A "rejected" suggestion might be feasible but unwanted (e.g., impractical to execute). A "violated" result is measured evidence of infeasibility.

#### Suggestion vs. Control vs. Replicate

| Concept | Definition | Optimized? |
|---------|------------|------------|
| **Suggestion** | A novel experimental design proposed by BO | Yes |
| **Control** | A standard/baseline experiment included for validation | No |
| **Replicate** | A repeat of a previous experiment for statistical purposes | No |

**Key principle**: Controls and replicates are tracked separately. They contribute to result data but are not optimization targets.

#### Model State vs. Audit State

| Concept | Definition | Mutability |
|---------|------------|------------|
| **Model State** | Current GP hyperparameters, training data, cached computations | Mutable (versioned) |
| **Audit State** | Complete history of all actions, decisions, and their provenance | Append-only |

### 2.2 Ambiguities in Intake Form

| Intake Term | Ambiguity | Proposed Resolution |
|-------------|-----------|---------------------|
| "target" for KPI | Unclear if target is an optimization goal or a constraint | Introduce explicit fields: `optimization_direction` (min/max/none) and `constraint_bound` (optional) |
| "evaluation method" | Mixes measurement method with timing and reliability | Separate into: `measurement_method`, `measurement_uncertainty`, `measurement_delay` |
| "unique designs" | Does this include controls? | Define: `batch_size` (suggestions only), `controls_per_batch`, `total_experiments_per_iteration` |
| "cycle initialization" | Vague; conflates initial design with iteration start | Separate: `initial_design_strategy` (first iteration) vs `iteration_continuation_strategy` |
| "scaling strategy" | Out of scope for core BO; conflates experiment scale with optimization | Treat as metadata only; scaling is not BO's responsibility |
| "constraints" section | Mixes input constraints (design space) with operating constraints (execution) | Separate: `design_constraints` (affect suggestions) vs `execution_constraints` (affect feasibility) |

---

## 3. MCP Resources (Conceptual)

### 3.1 Minimum Necessary Resources

| Resource | Purpose | Lifecycle | Primary Readers |
|----------|---------|-----------|-----------------|
| `campaign` | Root resource; contains campaign metadata and current state | Mutable (state transitions) | GUI, Agent, Auditor |
| `campaign_spec` | Immutable resolved configuration created from intake | Immutable | Agent, Auditor |
| `iteration` | A single cycle of suggestions → experiments → results | Append-only | GUI, Agent |
| `suggestion` | A single experimental design with full provenance | Immutable | GUI, Agent, Automation, Auditor |
| `result` | Measured outcomes for a suggestion | Append-only (can add late measurements) | Agent, GUI |
| `model_snapshot` | Versioned capture of model state at a point in time | Immutable | Agent, Auditor |
| `diagnostic_report` | Point-in-time assessment of model health and progress | Immutable | GUI, Agent, Auditor |
| `audit_log` | Append-only log of all actions and decisions | Append-only | Auditor |
| `user` | User identity and permissions (for multi-user collaboration) | Mutable (profile) | Auth layer, Auditor |
| `campaign_membership` | Links users to campaigns with roles | Append-only (add members) | GUI, Auth layer |

### 3.2 Resource Details

#### `campaign`
- **Purpose**: Central coordination resource for the optimization effort
- **Contents**: ID, name, status (created/running/paused/completed/failed), creation time, current iteration, references to campaign_spec, **version number** (for optimistic locking)
- **Lifecycle**: Created → Running → (Paused ↔ Running) → Completed
- **Reproducibility support**: Links to all immutable children; status transitions logged
- **Collaboration support**: Version number increments on every mutation; concurrent edits detected via version mismatch

#### `campaign_spec`
- **Purpose**: The fully resolved, validated configuration that defines the optimization problem
- **Contents**: Input parameters (with resolved types, ranges, transforms), objectives (with directions), constraints (with types), batch settings, model configuration defaults
- **Lifecycle**: Immutable; created once during campaign initialization
- **Reproducibility support**: Immutable by design; any change requires new campaign

#### `suggestion`
- **Purpose**: A proposed experimental design with complete provenance
- **Contents**: Parameter values, iteration reference, generation timestamp, model_snapshot reference, random seed, acquisition function settings, confidence/uncertainty metrics, human-readable explanation
- **Lifecycle**: Immutable once generated
- **Reproducibility support**: Contains all information needed to attempt regeneration

#### `model_snapshot`
- **Purpose**: Captured state of the surrogate model at suggestion generation time
- **Contents**: Version ID, hyperparameters, training data hash, transform configurations, timestamp
- **Lifecycle**: Immutable; new snapshot created when model is updated
- **Reproducibility support**: Enables "replay" attempts with same model state

#### `user` (Collaboration Support)
- **Purpose**: Track user identity for multi-user collaboration and audit attribution
- **Contents**: User ID, display name, authentication reference (external ID for SSO), role defaults
- **Lifecycle**: Mutable (profile updates); ID is immutable
- **Reproducibility support**: User ID in audit log enables attribution of all actions

#### `campaign_membership` (Collaboration Support)
- **Purpose**: Associate users with campaigns and define access levels
- **Contents**: Campaign ID, user ID, role (owner/editor/viewer), joined timestamp
- **Lifecycle**: Append-only; members can be added but not removed (inactive flag instead)
- **Reproducibility support**: Membership history preserved for audit

### 3.3 Resource Versioning Strategy

| Resource Type | Versioning Approach | Rationale |
|---------------|---------------------|-----------|
| `campaign_spec` | Immutable (no versions; change = new campaign) | Configuration changes fundamentally alter the optimization problem |
| `suggestion` | Immutable with unique IDs | Suggestions are scientific records; no modification allowed |
| `model_snapshot` | Sequential version numbers within campaign | Model evolves; versions enable replay and comparison |
| `result` | Append-only with timestamps | Late-arriving measurements are valid; modifications are appends |
| `audit_log` | Append-only | Audit integrity requires no modification or deletion |

---

## 4. MCP Tools (Intent-Level Contracts)

### 4.1 Tool Overview

> **Collaboration Note**: All mutating tools require authenticated user context. User ID is recorded in audit log for every action. Tools that modify campaign state include expected version number for optimistic locking.

| Tool | Intent | Category | Status |
|------|--------|----------|--------|
| `validate_intake` | Check intake for errors before campaign creation | Setup | ✅ Implemented |
| `create_campaign` | Initialize a new campaign from validated intake | Setup | ✅ Implemented |
| `generate_suggestions` | Produce next batch of experimental designs | Core Loop | ✅ Implemented |
| `submit_results` | Record measured outcomes for suggestions | Core Loop | ✅ Implemented |
| `get_diagnostics` | Assess model health and optimization progress | Monitoring | ✅ Implemented |
| `get_suggestion_explanation` | Explain why a specific suggestion was generated | Transparency | ✅ Implemented (2026-01-15) |
| `replay_suggestion` | Attempt deterministic regeneration of a past suggestion | Audit | 🔄 Planned |
| `pause_campaign` / `resume_campaign` | Suspend/continue active optimization | Lifecycle | ✅ Implemented (2026-01-15) |
| `terminate_campaign` | End optimization and archive campaign | Lifecycle | ✅ Implemented (2026-01-15) |
| `add_campaign_member` | Grant a user access to a campaign | Collaboration | 🔄 Planned |
| `upload_results_file` | Parse and submit results from CSV/Excel file | Ingestion | ✅ Implemented (2026-01-15) |

### 4.2 Tool Contracts

#### `validate_intake`

- **Intent**: Determine if an intake can be successfully converted to a CampaignSpec
- **Preconditions**: Raw intake data provided
- **Outputs**:
  - Validation result (pass/fail)
  - List of errors (blocking issues)
  - List of warnings (non-blocking concerns)
  - Preview of derived defaults (what will be assumed)
- **Failure modes**:
  - Invalid parameter types or ranges → Error list
  - Inconsistent constraints → Error list
  - Missing required fields → Error list
  - Ambiguous specifications → Warning list with assumed resolutions
- **Side effects**: None (pure validation)

#### `create_campaign`

- **Intent**: Initialize a new optimization campaign from intake
- **Preconditions**:
  - Intake passes validation
  - User has confirmed acceptance of derived defaults
- **Outputs**:
  - Campaign ID
  - Resolved CampaignSpec (read-only reference)
  - Initial model snapshot ID
- **Failure modes**:
  - Validation failures → Rejected with error list
  - Internal initialization failure → Error with rollback
- **Side effects**:
  - Creates `campaign` resource
  - Creates `campaign_spec` resource
  - Creates initial `model_snapshot`
  - Appends to `audit_log`

#### `generate_suggestions`

- **Intent**: Produce the next batch of experimental designs
- **Preconditions**:
  - Campaign is in running state
  - Previous iteration's results submitted (or explicit skip)
  - Requested batch size ≤ configured maximum
- **Outputs**:
  - List of suggestions (each with full provenance)
  - Iteration ID
  - Model confidence summary
  - Optional: Controls/replicates if configured
- **Failure modes**:
  - Model fitting failure → Error with diagnostic info
  - No feasible region found → Warning with constrained space analysis
  - Optimization stalled → Warning with health assessment
- **Side effects**:
  - Creates `suggestion` resources (immutable)
  - Creates `model_snapshot` if model updated
  - Creates `iteration` resource
  - Appends to `audit_log`

#### `submit_results`

- **Intent**: Record measured outcomes for executed experiments
- **Preconditions**:
  - Suggestion IDs reference existing suggestions
  - Campaign is running or paused
- **Outputs**:
  - Confirmation of recorded results
  - Updated constraint violation status (if any)
  - Preliminary model update summary
- **Failure modes**:
  - Unknown suggestion ID → Error
  - Mismatched KPI names → Error
  - Duplicate submission → Warning with idempotency handling
- **Side effects**:
  - Creates/updates `result` resources
  - Triggers model update (lazy or immediate based on config)
  - Appends to `audit_log`

#### `get_diagnostics`

- **Intent**: Assess current model health and optimization progress
- **Preconditions**: Campaign exists (any state)
- **Outputs**:
  - Model quality indicators (see Section 5)
  - Progress metrics (improvement trajectory, Pareto front evolution)
  - Health status (healthy/warning/critical)
  - Human-readable summary
  - Recommendations (if health is warning/critical)
- **Failure modes**:
  - Insufficient data for diagnostics → Partial report with caveats
- **Side effects**:
  - Creates `diagnostic_report` resource
  - Appends to `audit_log`

#### `get_suggestion_explanation`

- **Intent**: Explain why a specific suggestion was generated
- **Preconditions**: Suggestion ID exists
- **Outputs**:
  - Acquisition function value breakdown
  - Predicted objective values with uncertainty
  - Constraint satisfaction probability
  - Comparison to alternative candidates
  - Human-readable narrative
- **Failure modes**:
  - Suggestion not found → Error
- **Side effects**: Appends to `audit_log`

#### `replay_suggestion`

- **Intent**: Attempt to regenerate a suggestion deterministically
- **Preconditions**:
  - Original suggestion exists with complete provenance
  - Referenced model snapshot is available
- **Outputs**:
  - Regenerated suggestion values
  - Match status (identical/close/divergent)
  - If divergent: explanation of divergence sources
- **Failure modes**:
  - Model snapshot unavailable → Error
  - Non-deterministic component identified → Explanation with affected parameters
- **Side effects**:
  - Creates `audit_log` entry documenting replay attempt

#### `add_campaign_member` (Collaboration)

- **Intent**: Grant a user access to a campaign with specified role
- **Preconditions**:
  - Calling user is campaign owner or has admin privileges
  - Target user exists in system
  - Campaign exists and is not completed
- **Outputs**:
  - Confirmation of membership creation
  - Updated member list for campaign
- **Failure modes**:
  - Insufficient permissions → Error with required role
  - User already member → Warning (idempotent success)
  - Campaign completed → Error
- **Side effects**:
  - Creates `campaign_membership` resource
  - Appends to `audit_log` with both acting user and target user

#### `upload_results_file` (Ingestion)

- **Intent**: Parse and submit results from an uploaded file (CSV/Excel)
- **Preconditions**:
  - User has editor role on campaign
  - File format is supported (CSV, XLSX)
  - File contains required columns (suggestion ID or parameters, KPI values)
- **Outputs**:
  - Parse summary (rows found, rows valid, rows with errors)
  - List of created/updated results
  - Detailed error report for invalid rows
- **Failure modes**:
  - Unsupported file format → Error
  - Missing required columns → Error with expected schema
  - Partial parse failure → Warning with successful rows processed, errors listed
  - Suggestion ID not found → Error for that row
- **Side effects**:
  - Creates `result` resources for valid rows
  - Appends to `audit_log` (single entry for batch upload with row count)
  - **Does NOT** fail atomically — valid rows are committed even if some rows error

### 4.3 Why Contracts Must Be Stabilized Before Endpoints/Schemas

1. **Intent precedes format**: The tool contracts define *what decisions are supported*. Endpoint design is about *how to invoke* — a lower-level concern.

2. **Schema stability**: Field-level schemas will be derived from contracts. Designing schemas first risks encoding the wrong abstractions.

3. **Failure mode completeness**: Contracts force enumeration of failure modes before implementation. Schemas designed first often omit error structures.

4. **Versioning clarity**: Contract versioning is semantic (what changed about intent). Schema versioning is syntactic. Semantic versioning must be designed first.

5. **Multi-transport support**: The same contracts can be exposed via REST, gRPC, or native MCP. Premature endpoint design locks in a single transport.

---

## 5. Model Performance, Diagnostics, and "Is BO Actually Working?"

### 5.1 What "Working" Means — User-Facing Definitions

| Scenario | "Working" Definition | "Not Working" Indicators |
|----------|----------------------|--------------------------|
| **Single-objective, unconstrained** | Objective value improves over iterations; model predictions correlate with observations | No improvement trend; predictions consistently wrong; suggestions indistinguishable from random |
| **Single-objective, constrained** | Feasible region is explored efficiently; best feasible objective improves | Repeated constraint violations; model cannot find feasible points; feasible region exhausted |
| **Multi-objective** | Pareto front expands (hypervolume increases); new non-dominated solutions found | Hypervolume stagnant; same Pareto points recycled; objectives treated as independent |
| **All scenarios** | Uncertainty reduces over iterations; model learns from data | Uncertainty constant or increasing; model ignores new data |

### 5.2 Required Diagnostics

#### Single-Objective Optimization

| Metric | Description | User Interpretation |
|--------|-------------|---------------------|
| **Best observed value** | Best objective value achieved | "How good is my best result?" |
| **Improvement trajectory** | Best value vs. iteration number | "Am I still improving?" |
| **Model correlation** | Predicted vs. actual values (rank correlation) | "Can I trust predictions?" |
| **Exploration/exploitation ratio** | Balance between novel vs. refined suggestions | "Is BO being too conservative/aggressive?" |
| **Constraint satisfaction rate** | % of suggestions that satisfy constraints | "Am I wasting experiments on infeasible points?" |

#### Multi-Objective Optimization

| Metric | Description | User Interpretation |
|--------|-------------|---------------------|
| **Pareto front size** | Number of non-dominated solutions | "How many good options do I have?" |
| **Hypervolume** | Volume dominated by Pareto front | "Overall progress indicator" |
| **Hypervolume improvement** | Change in hypervolume per iteration | "Am I still finding better trade-offs?" |
| **Objective-wise progress** | Best value for each objective individually | "How good is my best for each goal?" |
| **Pareto front diversity** | Spread of solutions across objective space | "Do I have varied options or clustered ones?" |

### 5.3 Interpreting Quality Under Complex Conditions

#### When Progress Is Pareto-Based Rather Than Scalar

- **DO NOT** report "best suggestion" — there is no single best
- **DO** report Pareto front evolution, hypervolume delta
- **DO** show trade-off visualization (objectives as axes)
- **DO** allow users to filter Pareto front by preference regions

#### When Objectives Conflict

- **Explicitly state**: "Objectives X and Y appear to conflict; improving one degrades the other"
- **Show**: Conflict correlation from model
- **Recommend**: User may need to prioritize or define acceptable trade-off ranges

#### When Constraints Dominate Search

- **Detect**: Feasibility rate < threshold (e.g., < 10% of suggestions feasible)
- **Report**: "Constraint satisfaction is challenging; most of the design space appears infeasible"
- **Recommend**: Review constraints, consider relaxation, or accept slower progress

### 5.4 Minimum Metrics Always Available

These diagnostics are computed for every campaign, every iteration:

1. **Iteration count**: How many optimization cycles completed
2. **Suggestion count**: Total suggestions generated
3. **Result count**: Total results received
4. **Feasibility rate**: % of executed suggestions that were feasible
5. **Model fit indicator**: High-level quality score (good/moderate/poor)
6. **Progress indicator**: Improving/stagnant/regressing

### 5.5 Early Warning: BO Ineffective ("Random Walk" Detection)

| Warning Condition | Detection Method | User Message |
|-------------------|------------------|--------------|
| No improvement for N iterations | Track best value; check for plateau | "Optimization has not improved in X iterations. Consider: reviewing constraints, expanding search space, or stopping." |
| Model predictions uncorrelated | Track rank correlation < threshold | "Model predictions are not matching experimental results. The model may need more data or the problem may not suit BO." |
| Suggestions cluster in small region | Track diversity of suggestions | "Suggestions are clustered; exploration may be stuck. Consider adjusting acquisition function." |
| Uncertainty not decreasing | Track average uncertainty over time | "Model uncertainty is not decreasing. New experiments are not informing the model effectively." |

### 5.6 GUI Display vs. Internal Metrics

#### What the GUI Should Show

- Progress charts (objective vs. iteration, Pareto front evolution)
- Health status (green/yellow/red with plain-language explanation)
- Best results (for single-objective) or Pareto front browser (multi-objective)
- Suggestion-level confidence ("high/medium/low confidence this is a good experiment")
- Warnings and recommendations when health degrades
- Comparative view: "How does current iteration compare to previous?"

#### What Remains Internal (Not Shown to Users)

- GP hyperparameter values (length scales, noise variance)
- Acquisition function numerical values
- Kernel matrices and covariance structures
- Optimization trace details (acquisition optimizer iterations)
- Raw model residuals

**Principle**: Users see *outcomes and confidence*, not *mechanism details*.

### 5.7 Feature Importance and Parameter Sensitivity

Understanding which parameters most influence outcomes helps users:
- Validate that the model aligns with domain knowledge
- Focus experimental resources on influential parameters
- Gain insight into the optimization landscape

#### Methods

| Method | Source | Interpretation |
|--------|--------|----------------|
| **Inverse Length Scales** | GP kernel (Matérn 5/2) | Smaller length scale = higher sensitivity = more important |
| **SHAP Values** | Shapley additive explanations | Average |SHAP value| across predictions |

Both methods are computed every iteration when sufficient data exists (≥2× number of parameters).

#### Display

- **Aggregate importance**: Horizontal bar chart ranking all parameters
- **Per-objective breakdown**: Importance shown separately for each objective
- **Categorical parameters**: Importance shown per category (e.g., catalyst_Pt, catalyst_Pd)

#### User Interpretation

| Metric | Example Interpretation |
|--------|------------------------|
| Parameter ranking | "Temperature and pressure matter most; catalyst has moderate effect" |
| Per-objective differences | "Temperature affects yield strongly, but pressure affects cost more" |
| Equal importance warning | "All parameters have similar importance - model may need more data" |

---

## 6. Transparency, Determinism, and Auditability

### 6.1 Full Transparency Definition

For experimental design generation, **full transparency** means:

1. **Strategy visibility**: Users know which optimization strategy is active (acquisition function class, batch strategy)
2. **Settings visibility**: All configurable parameters are documented and accessible
3. **Transform visibility**: Any input/output transformations are explicit (normalization, log transforms, encoding)
4. **Assumption visibility**: Implicit assumptions (noise model, stationarity, smoothness) are documented
5. **Code path traceability**: In principle, users can identify the exact code version that generated a suggestion

### 6.2 User-Visible Information

| Category | Visible Information | Access Method |
|----------|---------------------|---------------|
| **Strategy** | Acquisition function type, batch generation method | CampaignSpec inspection |
| **Settings** | Exploration/exploitation balance, batch size, constraint handling mode | CampaignSpec inspection |
| **Transforms** | Input normalization, output transforms, constraint encodings | CampaignSpec inspection |
| **Assumptions** | Noise model, model class, stationarity assumption | CampaignSpec inspection |
| **Provenance** | For any suggestion: seed, model snapshot ID, settings hash | Suggestion provenance field |
| **Explanation** | Why this suggestion was chosen over alternatives | `get_suggestion_explanation` tool |

### 6.3 Deterministic Generation: Feasibility Analysis

#### What Must Be Fixed/Versioned for Determinism

| Component | Versioning Approach | Notes |
|-----------|---------------------|-------|
| Random seed | Captured per-suggestion | Primary source of determinism |
| Model snapshot | Versioned, immutable | Model state at generation time |
| Training data | Hash stored in snapshot | Inputs to model |
| Hyperparameters | Part of model snapshot | GP/model configuration |
| Transform configs | Part of CampaignSpec | Pre/post processing |
| Acquisition settings | Part of CampaignSpec | Optimization parameters |
| Code version | Stored in audit log | Enables exact replay environment |

#### Where Determinism Breaks Down

| Source of Non-Determinism | Mitigation | Communication |
|---------------------------|------------|---------------|
| Floating-point non-associativity | Use deterministic linear algebra modes where available | "Replay may differ at machine precision level" |
| Acquisition optimizer randomness | Capture internal optimizer seed | Included in provenance |
| Library version changes | Record full dependency versions | "Replay requires original library versions" |
| Hardware differences (GPU vs CPU) | Document execution environment | "Replay on different hardware may produce different results" |
| Parallel execution ordering | Force sequential where determinism required | Trade-off with performance |

### 6.4 Replay Mechanism

#### Request: "Re-generate suggestion for iteration t"

**Procedure**:
1. Retrieve original suggestion provenance (seed, model snapshot, settings)
2. Load model snapshot state
3. Configure identical settings
4. Generate suggestion with same seed
5. Compare to original

**Outcomes**:

| Match Status | Meaning | User Communication |
|--------------|---------|---------------------|
| **Identical** | All values match exactly | "Replay successful; suggestion is reproducible" |
| **Close** | Values differ by < tolerance | "Replay produced equivalent result (floating-point tolerance)" |
| **Divergent** | Significant differences | "Replay differs; [explanation of divergence source]" |

**Divergence Explanations**:
- "Library version differs; original used X.Y.Z, replay used A.B.C"
- "Model snapshot incomplete; cannot restore exact state"
- "Non-deterministic component identified: [name]"

---

## 7. Intake → BO Translation

### 7.1 Translation Overview

```
Intake (user-facing) → Validation → Resolution → CampaignSpec (system-internal)
```

### 7.2 Required vs. Optional Inputs

#### Required (Campaign Cannot Start Without)

| Intake Field | Why Required |
|--------------|--------------|
| At least one KPI marked as optimization target | No optimization target = no BO |
| At least one input parameter with range | No design space = no suggestions |
| Optimization direction for each objective | Cannot optimize without knowing min/max |
| Batch size (unique designs per iteration) | Controls suggestion generation |

#### Optional (Defaults Applied)

| Intake Field | Default | Rationale |
|--------------|---------|-----------|
| Maximum iterations | None (unbounded) | Users may not know upfront |
| Controls per batch | 0 | Not all experiments need controls |
| Initial design strategy | Latin Hypercube | Standard BO practice |
| Constraint handling mode | Probabilistic | Most flexible approach |
| Noise estimation | Inferred from data | Users rarely know noise a priori |

### 7.3 Derived Quantities and Defaults

| Derived Quantity | Derivation Logic | Explicit in CampaignSpec |
|------------------|------------------|-------------------------|
| Input dimensionality | Count of input parameters | Yes |
| Output dimensionality | Count of optimization objectives | Yes |
| Design space bounds | Extracted from parameter ranges | Yes |
| Constraint count | Count of explicit constraints | Yes |
| Normalization transforms | Auto-derived from parameter types/ranges | Yes |
| Categorical encoding | Derived from categorical parameter levels | Yes |

### 7.4 Validations

#### Fail-Fast Validations (Block Campaign Creation)

| Check | Error |
|-------|-------|
| Continuous parameter range min ≥ max | "Parameter X has invalid range: min must be < max" |
| Categorical parameter with 0 or 1 levels | "Parameter X must have at least 2 levels" |
| Constraint references unknown parameter | "Constraint references unknown parameter: X" |
| Optimization direction missing for objective | "Objective X must specify minimize or maximize" |
| Sum constraint doesn't match parameter count | "Sum constraint references X parameters but Y are marked for sum" |
| Conflicting constraints (algebraically infeasible) | "Constraints are mutually exclusive; no feasible region exists" |

#### Warnings (Proceed with Caution)

| Check | Warning |
|-------|---------|
| Very high dimensionality (>20 parameters) | "High-dimensional problems may require many iterations for BO to be effective" |
| Very small design space | "Design space is very small; exhaustive search may be more efficient" |
| No constraints but all KPIs have targets | "Consider whether targets should be constraints or just tracking goals" |
| Historical data scale doesn't match current scale | "Historical data is from different scale; transfer learning limitations apply" |

### 7.5 Underspecified or Dangerous Intake Questions

| Question | Problem | Proposed Improvement |
|----------|---------|----------------------|
| "target" for KPI | Conflates goal vs. constraint | Split: "optimization direction" (none/min/max) + "constraint bounds" (optional) |
| "evaluation method" | Too unstructured | Add: "measurement uncertainty estimate" (optional), "measurement delay" (optional) |
| "controls" | What exactly is a control? | Add: "control type" (baseline/reference/blank) + "control values" (explicit specification) |
| "cycle initialization" | First cycle ≠ later cycles | Split: "initial design" (first cycle) vs "continuation" (later cycles) |
| Operating constraints | Mixed with design constraints | Separate sections: "Design space constraints" vs "Execution constraints" |
| Historical data | Assumed compatible | Add: "data quality assessment" (complete/partial/uncertain) |

### 7.6 Intake Improvements for Reduced Ambiguity

**Proposed Intake Structure Revisions**:

1. **Explicit objective designation**: Each KPI gets: `is_optimization_target: yes/no`, `direction: minimize/maximize`

2. **Constraint separation**:
   - Design constraints (affect what BO suggests)
   - Measurement constraints (affect feasibility interpretation)

3. **Uncertainty quantification**:
   - Measurement noise estimate per KPI (optional, default: infer)
   - Parameter setting precision per input (optional)

4. **Batch configuration clarity**:
   - `suggestions_per_iteration`: novel points from BO
   - `controls_per_iteration`: fixed control experiments
   - `replicates_per_iteration`: repeated experiments

5. **Historical data qualification**:
   - Scale compatibility flag
   - Data completeness indicator
   - Known data quality issues

---

## 8. Evolution Strategy and Explicit Non-Goals

### 8.1 Features Explicitly Deferred

| Feature | Why Deferred | Reversibility |
|---------|--------------|---------------|
| **Multi-fidelity BO** | Adds complexity; requires fidelity-aware model | High — can add fidelity dimension later |
| **Cost-aware acquisition** | Requires cost model and different acquisition | High — acquisition function is pluggable |
| **Transfer learning from historical campaigns** | Requires cross-campaign model linking | Medium — needs upfront data architecture |
| **Reinforcement Learning policies** | Different optimization paradigm entirely | High — separate from BO pathway |
| **Real-time streaming data** | Batch mode sufficient for physical experiments | Medium — requires async architecture changes |
| **Automatic hyperparameter tuning** | Start with sensible defaults; optimize later | High — internal to model fitting |
| **Custom acquisition functions** | Standard acquisitions sufficient for v1 | High — plugin architecture planned |

### 8.2 Decisions That Must Remain Reversible

| Decision | Why Reversible | Mechanism |
|----------|----------------|-----------|
| Acquisition function choice | May need problem-specific acquisitions | Plugin interface planned |
| Constraint handling mode | Different problems suit different approaches | CampaignSpec flag |
| Model class (GP variant) | May need different kernels, likelihoods | Model factory pattern |
| Batch generation strategy | q-EI, KB, etc. may need swapping | Strategy pattern in suggestion generation |
| Normalization approach | Some problems need different transforms | Transform registry |

### 8.3 Decisions Costly to Undo Once Coded

| Decision | Why Costly | Mitigation |
|----------|------------|------------|
| **Suggestion immutability** | Data model assumes immutable records; mutable suggestions require migration | Lock in immutability; design for it |
| **Model snapshot storage format** | Changing format breaks replay capability | Use versioned, self-describing format |
| **Audit log schema** | Historical logs must remain parseable | Append-only schema with version field |
| **Campaign-to-CampaignSpec relationship** | 1:1 vs. 1:N has architectural implications | Lock in 1:1 (new config = new campaign) |
| **Result append semantics** | Mutable vs. append-only affects consistency model | Lock in append-only |

### 8.4 BoTorch Feature Roadmap

For detailed feature roadmap with implementation status, version history, code examples, and BoTorch references, see **[IMPLEMENTATION_PLAN.md - BoTorch Feature Roadmap](IMPLEMENTATION_PLAN.md#botorch-feature-roadmap)**.

**Summary of Current Status**:
- v1.0-v1.3: ✅ Completed (single/multi-objective, TuRBO, LOO CV, Input Warping, Outcome Constraints)
- v2.0: Partially complete (Multi-Fidelity ✅, RGPE ✅, SAASBO ✅, Risk-Averse ⏳, Full EIpu ⏳)

---

## 9. Implementation Planning Hooks

For module/component breakdown diagrams and dependency direction details, see **[IMPLEMENTATION_PLAN.md - System Architecture Overview](IMPLEMENTATION_PLAN.md#system-architecture-overview)**.

### 9.1 Milestone Plan (Revised per Section 0a)

> **Note**: Original milestone plan superseded by stakeholder requirements. MVP scope expanded to include multi-objective, all parameter types, and full collaboration.

#### MVP (Feature-Complete Release)

**Scope** (all required before first deployment):

*Core Optimization*
- Multi-objective optimization (2-4 objectives)
- Pareto front tracking with hypervolume diagnostics
- Continuous, discrete, AND categorical parameters
- Sum/mixture constraints (parameters summing to 1)
- Medium dimensionality support (6-15 parameters)

*Collaboration*
- Multi-user campaigns with full collaboration
- Optimistic locking for conflict resolution
- User attribution in audit logs
- Basic authentication

*Result Ingestion*
- Manual entry via GUI
- File upload (CSV/Excel)
- API push from external systems

*Diagnostics & Transparency*
- Pareto progress visualization
- Hypervolume improvement tracking
- Model health monitoring with warnings
- Suggestion provenance and explanation
- "BO not working" detection

*Infrastructure*
- Containerized deployment (AWS + on-prem)
- Suggestion generation with full provenance
- Audit log for all actions

**Success criteria**:
- Multi-objective test case (3 objectives) shows Pareto front expansion
- Mixed parameter types (continuous + categorical) handled correctly
- Mixture constraint (sum = 1) satisfied by all suggestions
- Multiple users can collaborate on same campaign without data corruption
- Three ingestion methods produce consistent campaign state
- Diagnostics correctly identify Pareto progress vs. stagnation
- Replay produces identical suggestions (within tolerance)
- Audit log complete with user attribution

#### v1.1 (Post-MVP Enhancements)

**Scope (adds to MVP)**:
- Historical data integration (warm-start campaigns)
- SSO/SAML + OAuth authentication
- Advanced batch strategies
- Performance optimization for large campaigns
- Enhanced LIMS/ELN integration

**Success criteria**:
- Historical data warm-start improves convergence vs. cold start
- Enterprise SSO integration functional
- Large campaign (100+ results) suggestion latency acceptable

#### v2.0 (Future)

**Scope (adds to v1.1)**:
- Transfer learning across campaigns
- Custom acquisition function plugins
- Multi-fidelity BO
- Cost-aware acquisition
- Advanced constraint types (conditional, nonlinear)

### 9.4 MVP Acceptance Criteria (Pass/Fail) — Revised

| Criterion | Pass | Fail |
|-----------|------|------|
| **Campaign Lifecycle** | | |
| Create campaign from valid intake | Campaign resource exists; CampaignSpec immutable | Any error during creation |
| Reject invalid intake | Validation errors returned; no campaign created | Campaign created with invalid config |
| **Multi-Objective Optimization** | | |
| Generate Pareto-aware suggestions | Suggestions target Pareto front expansion | Suggestions ignore multi-objective structure |
| Track Pareto progress | Hypervolume increases over iterations | Hypervolume stagnant or unmeasured |
| Handle 3-4 objectives | Campaign with 3 objectives functions correctly | Fails with > 2 objectives |
| **Parameter Types** | | |
| Handle continuous parameters | Suggestions within bounds; properly normalized | Out-of-bounds or unnormalized |
| Handle discrete parameters | Suggestions are valid integers in range | Non-integer or out-of-range values |
| Handle categorical parameters | Valid category values; proper encoding | Invalid categories; encoding errors |
| **Constraints** | | |
| Enforce mixture constraints | Sum constraint satisfied by all suggestions | Suggestions violate sum constraint |
| **Collaboration** | | |
| Multi-user access | Multiple users can view/modify same campaign | Single-user lock or errors |
| Conflict detection | Concurrent edits trigger clear conflict message | Silent data overwrite |
| User attribution | Audit log records which user performed each action | Actions lack user identity |
| **Result Ingestion** | | |
| Manual entry works | Results submitted via GUI recorded correctly | GUI submission fails or corrupts |
| File upload works | CSV/Excel upload parses and records correctly | Upload rejected or misparses |
| API push works | External API call records results correctly | API returns errors or misrecords |
| **Diagnostics** | | |
| Pareto progress reporting | Hypervolume trend displayed; Pareto front visualized | Only scalar "best" shown |
| Health warnings | "BO not working" detected on random-noise objective | No warning on pure noise |
| **Reproducibility** | | |
| Suggestion replay | Same seed + model state = same suggestion (within tolerance) | Suggestions differ with identical inputs |
| Audit log complete | All actions logged with timestamps and user ID | Missing actions or user attribution |

---

## 10. End-to-End Test Cases

### 10.1 Primary Test Case: Basic Optimization Flow

#### Example Intake (Simplified)

```yaml
project_title: "Catalyst Optimization Demo"
description: "Optimize catalyst composition for reaction yield"

kpis:
  - name: "reaction_yield"
    unit: "%"
    optimization_direction: maximize
    target: "> 90"

inputs:
  - name: "catalyst_A_fraction"
    type: continuous
    range: [0.0, 1.0]
  - name: "catalyst_B_fraction"
    type: continuous
    range: [0.0, 1.0]

constraints:
  - type: sum_equals
    parameters: ["catalyst_A_fraction", "catalyst_B_fraction"]
    value: 1.0

batch_size: 3
max_iterations: 10
```

#### Synthetic Objective Function

```
yield(A, B) = 80 + 15 * A - 5 * B + 10 * A * B + noise(σ=2)

where A + B = 1 (enforced by constraint)

True optimum: A = 1.0, B = 0.0 → yield ≈ 95%
```

#### Expected Operation Sequence

1. **Validate intake**
   - Call: `validate_intake(intake)`
   - Expected: Pass; no errors; confirm sum constraint recognized

2. **Create campaign**
   - Call: `create_campaign(intake)`
   - Expected: Campaign ID returned; CampaignSpec created with:
     - 2 input dimensions (normalized to [0,1])
     - 1 objective (maximize)
     - 1 sum constraint
   - Resources created: `campaign`, `campaign_spec`, `model_snapshot[0]`

3. **Generate initial suggestions**
   - Call: `generate_suggestions(campaign_id, batch_size=3)`
   - Expected: 3 suggestions returned
   - Each suggestion has: parameter values (satisfying A + B = 1), provenance (seed, model snapshot ref)
   - Resources created: `iteration[0]`, `suggestion[0,1,2]`

4. **Compute synthetic results**
   - Apply objective function to each suggestion
   - Example: If suggestion[0] = {A: 0.33, B: 0.67} → yield ≈ 85%

5. **Submit results**
   - Call: `submit_results([{suggestion_id: 0, yield: 85}, ...])`
   - Expected: Confirmation; no constraint violations
   - Resources updated: `result[0,1,2]` created

6. **Generate next suggestions (iteration 1)**
   - Call: `generate_suggestions(campaign_id, batch_size=3)`
   - Expected: 3 new suggestions
   - Model should favor higher A values (learning from data)
   - Resources created: `iteration[1]`, `suggestion[3,4,5]`, `model_snapshot[1]`

7. **Repeat iterations 2-9**
   - Expected: Suggestions converge toward A ≈ 1.0
   - Best observed yield should exceed 90% within 5-7 iterations

8. **Run diagnostics**
   - Call: `get_diagnostics(campaign_id)`
   - Expected:
     - Improvement trajectory shows upward trend
     - Model correlation: positive (predictions match reality)
     - Health status: green
     - Best observed: > 90%

#### Expected Outcomes

| Resource | Expected State |
|----------|----------------|
| `campaign` | status: running; iteration_count: 10 |
| `campaign_spec` | Immutable; matches validated intake |
| `suggestion` | 30 total (3 × 10 iterations); all have provenance |
| `result` | 30 total; linked to suggestions |
| `model_snapshot` | At least 10 (one per iteration) |
| `diagnostic_report` | At least 1; shows improvement |
| `audit_log` | Contains all operations with timestamps |

#### Pass/Fail Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| All suggestions satisfy A + B = 1 | Sum within tolerance (1e-6) | Any suggestion violates sum |
| Yield improves over iterations | Best yield at iteration 10 > best at iteration 1 | No improvement |
| Suggestions converge | Final iteration suggestions near A ≈ 1 | Suggestions remain random |
| Deterministic (within tolerance) | Same campaign recreation produces similar trajectory | Wildly different outcomes |
| Provenance complete | Every suggestion has seed, snapshot, settings | Missing provenance fields |

#### What Should Be Deterministic vs. Allowed to Vary

| Aspect | Determinism Expectation | Rationale |
|--------|-------------------------|-----------|
| Constraint satisfaction | Deterministic | Algebraic computation |
| Suggestion values (given seed) | Deterministic | Core reproducibility requirement |
| Model predictions | Deterministic | Same training data + hyperparameters |
| Optimization trajectory | Approximately deterministic | Floating-point tolerance |
| Best observed value | Allowed to vary slightly | Noise in objective function |
| Iteration to reach target | Allowed to vary | Stochastic exploration |

---

### 10.2 Additional Test Cases

#### Test Case 2: Constrained Feasibility Stress Test

**Intake sketch**:
- 3 continuous inputs
- 1 objective (maximize)
- Constraints: Multiple overlapping bounds creating small feasible region (~5% of design space)

**Synthetic objective**:
- Optimum lies near feasibility boundary

**Expected behavior**:
- Early iterations: Many infeasible suggestions (expected)
- Model should learn feasibility boundary
- Later iterations: Higher feasibility rate
- Diagnostics should NOT trigger "BO not working" (it's working, just constrained)

**Pass criteria**:
- Feasibility rate improves over iterations
- Best feasible value improves
- No false "BO ineffective" warnings

---

#### Test Case 3: Multi-Objective Pareto Progress

**Intake sketch**:
- 2 continuous inputs
- 2 objectives (maximize yield, minimize cost) — known to conflict

**Synthetic objective**:
- Yield and cost are negatively correlated
- Clear Pareto front exists

**Expected behavior**:
- Pareto front expands over iterations (hypervolume increases)
- Diverse solutions discovered (not just single-objective optima)
- Diagnostics correctly report Pareto progress, not scalar "best"

**Pass criteria**:
- Hypervolume increases monotonically (or near-monotonically)
- Pareto front contains ≥ 5 distinct solutions by iteration 10
- No single-objective "best" reported (multi-objective mode)

---

#### Test Case 4: "BO Not Working" Detection

**Intake sketch**:
- 3 continuous inputs
- 1 objective (maximize)
- No constraints

**Synthetic objective**:
- Pure noise: `f(x) = noise(σ=10)` with no signal
- BO cannot learn anything useful

**Expected behavior**:
- Model predictions should have no correlation with outcomes
- Suggestions should appear random (no convergence)
- Diagnostics should detect and warn

**Pass criteria**:
- Warning triggered by iteration 5-7: "Optimization is not showing improvement; model predictions are not correlated with observations"
- System does NOT falsely report progress
- Audit log records warning issuance

---

## Appendix: Open Questions and Resolved Decisions

### A.1 Resolved Through Stakeholder Q&A

| Question | Resolution | Reference |
|----------|------------|-----------|
| Single-user vs. multi-user campaigns | **Multi-user with full collaboration** | Section 0a |
| Conflict handling for concurrent edits | **Optimistic locking with version tracking** | Section 0a |
| Experimental throughput | **Low (1-5/day)** — no high-frequency optimization needed | Section 0a |
| User technical level | **Mixed** — scientists + data scientists | Section 0a |
| Acceptable suggestion latency | **Fast as possible** — with understanding that large batches are slower | Section 0a |
| MVP scope priority | **Feature completeness** — all features before first deploy | Section 0a |
| Deployment model | **Containerized** — AWS + on-prem | Section 0a |
| Authentication approach | **Simple for MVP; SSO/SAML + OAuth later** | Section 0a |

### A.2 Still Requires Decision Before Implementation

| Question | Options | Recommendation | Trade-offs |
|----------|---------|----------------|------------|
| Model snapshot granularity | Per-iteration vs per-suggestion | Per-iteration | Storage vs. replay precision |
| Constraint violation vs. measurement error | Distinct handling vs. unified | Distinct | Complexity vs. semantic clarity |
| Real-time diagnostics vs. on-demand | Always computed vs. requested | On-demand with caching | Compute cost vs. freshness |
| Suggestion batch atomicity | All-or-nothing vs. partial success | All-or-nothing | Simplicity vs. resilience |
| Categorical encoding strategy | One-hot vs. ordinal vs. target encoding | One-hot (safe default) | Dimensionality vs. model quality |
| Pareto front storage | Store all non-dominated vs. sample | Store all | Storage vs. completeness |

### A.3 Assumptions Validated

| Original Assumption | Status | Notes |
|---------------------|--------|-------|
| Batch mode is sufficient (no streaming) | **Validated** | Low throughput (1-5/day) confirms batch mode appropriate |
| Single concurrent user per campaign | **Invalidated** | Full collaboration required — design adjusted |
| Experimental noise is roughly stationary | **Not yet validated** | Still an assumption; may need non-stationary handling |
| Users will submit all results before requesting new suggestions | **Not yet validated** | May need partial-result handling |

### A.4 Remaining User Research Questions

| Question | Why It Matters | Priority |
|----------|----------------|----------|
| How often do users abort/restart campaigns? | Affects campaign lifecycle design | Medium |
| What file formats are actually used for result upload? | Affects parser implementation | High |
| What LIMS/ELN systems need integration? | Affects v1.1 integration scope | Medium |
| Typical Pareto front size at convergence? | Affects visualization and storage | Low |

---

## Appendix B: Implementation Status (Updated 2026-01-14)

This section tracks what has been implemented versus what remains from the design.

### B.1 MCP Tools Implementation Status

| Tool | Status | Notes |
|------|--------|-------|
| `validate_intake` | ✅ Implemented | `bo_mcp_server/tools/validate_intake.py` |
| `create_campaign` | ✅ Implemented | `bo_mcp_server/tools/create_campaign.py` |
| `generate_suggestions` | ✅ Implemented | `bo_mcp_server/tools/generate_suggestions.py` |
| `submit_results` | ✅ Implemented | `bo_mcp_server/tools/submit_results.py` |
| `get_diagnostics` | ✅ Implemented | `bo_mcp_server/tools/get_diagnostics.py` - includes feature importance & model correlation |
| `get_suggestion_explanation` | ✅ Implemented | `bo_mcp_server/tools/get_suggestion_explanation.py` |
| `replay_suggestion` | 🔄 Planned | Deferred - requires storing RNG seeds and model snapshots |
| `pause_campaign` / `resume_campaign` | ✅ Implemented | `bo_mcp_server/tools/campaign_lifecycle.py` |
| `terminate_campaign` | ✅ Implemented | `bo_mcp_server/tools/campaign_lifecycle.py` |
| `add_campaign_member` | 🔄 Planned | Deferred - collaboration features |
| `upload_results_file` | ✅ Implemented | `bo_mcp_server/tools/upload_results_file.py` |

### B.2 MCP Resources Implementation Status

| Resource | Status | Notes |
|----------|--------|-------|
| `campaign` | ✅ Implemented | `bo_mcp_server/resources/campaign_resource.py` |
| `suggestion` | ✅ Implemented | `bo_mcp_server/resources/suggestion_resource.py` |
| `diagnostic_report` | ❌ Not as resource | Diagnostics returned via tool, not as separate resource |

### B.3 BO Engine Implementation Status

| Module | Status | Notes |
|--------|--------|-------|
| `models.py` | ✅ Implemented | ModelListGP with Matérn 5/2 kernel + ARD |
| `acquisition.py` | ✅ Implemented | qLogNEHVI, qLogNEI, qLogNParEGO, EIpuAcquisition |
| `constraints.py` | ✅ Implemented | Sum/mixture constraints via reparameterization |
| `suggestions.py` | ✅ Implemented | Initial design (Sobol) + BO + TuRBO + outcome constraints |
| `diagnostics.py` | ✅ Implemented | Hypervolume, Pareto front, model health |
| `transforms.py` | ✅ Implemented | Categorical encoding, bounds handling |
| `feature_importance.py` | ✅ Implemented | Inverse lengthscales + SHAP values |
| `turbo.py` | ✅ Implemented | TuRBO state management, trust region bounds |
| `method_selector.py` | ✅ Implemented | Automatic method selection with explanations |
| `types.py` | ✅ Implemented | OptimizationSpec, OutcomeConstraintSpec, ObservationData with cost |

### B.4 Domain Models Implementation Status

| Model | Status | Notes |
|-------|--------|-------|
| `Campaign` | ✅ Implemented | Includes version for optimistic locking |
| `CampaignSpec` | ✅ Implemented | Immutable configuration |
| `Suggestion` | ✅ Implemented | Includes full provenance |
| `Result` | ✅ Implemented | Links to suggestions |
| `User` | ✅ Implemented | API key authentication |

### B.5 Storage Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| SQLite persistence | ✅ Implemented | SQLAlchemy async with aiosqlite |
| Repositories | ✅ Implemented | Campaign, Suggestion, Result, CampaignSpec |
| Optimistic locking | ✅ Implemented | Version field on Campaign entity |
| Audit logging | ❌ Not started | Not yet implemented |
| Alembic migrations | ❌ Not started | Using auto-create for MVP |

### B.6 API Routes Implementation Status

| Route | Status | Notes |
|-------|--------|-------|
| `/campaigns` | ✅ Implemented | Create, list, get, validate |
| `/suggestions` | ✅ Implemented | Generate, list |
| `/results` | ✅ Implemented | Submit (manual + file upload) |
| `/diagnostics` | ✅ Implemented | Get diagnostics with feature importance |

### B.7 Frontend Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Pages** | | |
| `IntakePage.tsx` | ✅ Implemented | Campaign creation form |
| `CampaignListPage.tsx` | ✅ Implemented | List user's campaigns |
| `CampaignDetailPage.tsx` | ✅ Implemented | Full campaign view with diagnostics |
| `DiagnosticsPage.tsx` | ⚠️ Embedded | Diagnostics shown within CampaignDetailPage |
| **Pareto Components** | | |
| `ParetoChart.tsx` | ✅ Implemented | Plotly scatter matrix |
| `HypervolumeChart.tsx` | ✅ Implemented | Hypervolume progression |
| **Diagnostics Components** | | |
| `HealthIndicator.tsx` | ✅ Implemented | Health status badge |
| `ProgressIndicator.tsx` | ✅ Implemented | Progress status |
| `ModelInfoCard.tsx` | ✅ Implemented | Model information display |
| `WarningsAlert.tsx` | ✅ Implemented | Health warnings |
| `FeatureImportanceChart.tsx` | ✅ Implemented | Parameter importance bar chart |
| **Common Components** | | |
| `ApiKeyModal.tsx` | ✅ Implemented | API key entry |
| `LoadingSpinner.tsx` | ✅ Implemented | Loading states |
| `ErrorAlert.tsx` | ✅ Implemented | Error display |
| **Campaign Components** | | |
| `SuggestionProvenance.tsx` | ✅ Implemented | Suggestion details |

### B.8 MVP Acceptance Criteria Status

| Criterion | Status | Notes |
|-----------|--------|-------|
| **Campaign Lifecycle** | | |
| Create campaign from valid intake | ✅ Pass | |
| Reject invalid intake | ✅ Pass | Validation errors returned |
| **Multi-Objective Optimization** | | |
| Generate Pareto-aware suggestions | ✅ Pass | qLogNEHVI |
| Track Pareto progress | ✅ Pass | Hypervolume computed |
| Handle 3-4 objectives | ✅ Pass | ModelListGP supports any count |
| **Parameter Types** | | |
| Handle continuous parameters | ✅ Pass | |
| Handle discrete parameters | ✅ Pass | |
| Handle categorical parameters | ✅ Pass | One-hot encoding |
| **Constraints** | | |
| Enforce mixture constraints | ✅ Pass | BoTorch constraint handling |
| **Collaboration** | | |
| Multi-user access | ⚠️ Partial | API keys work, but no role-based access |
| Conflict detection | ✅ Pass | Optimistic locking implemented |
| User attribution | ⚠️ Partial | User tracked, but no audit log |
| **Result Ingestion** | | |
| Manual entry works | ✅ Pass | |
| File upload works | ✅ Pass | CSV/Excel via API |
| API push works | ✅ Pass | |
| **Diagnostics** | | |
| Pareto progress reporting | ✅ Pass | Hypervolume trend |
| Health warnings | ✅ Pass | "BO not working" detection |
| Feature importance | ✅ Pass | Lengthscales + SHAP |
| **Reproducibility** | | |
| Suggestion replay | ❌ Not started | Tool not implemented |
| Audit log complete | ❌ Not started | Audit logging not implemented |

### B.9 What to Work on Next

**High Priority (MVP Blockers)**:
1. **Complete monorepo migration** - See Appendix C below

**Medium Priority (v1.0 Polish)**:
1. Add audit logging for all actions
2. Implement Alembic migrations
3. Add comprehensive E2E tests

**Lower Priority (v1.1 Features)**:
1. `get_suggestion_explanation` tool
2. `replay_suggestion` tool
3. Campaign lifecycle tools (pause/resume/terminate)
4. `add_campaign_member` for collaboration
5. SSO/OAuth authentication

---

## Appendix C: Monorepo Migration (COMPLETED)

The codebase has been restructured from a single backend package to a modular monorepo with independently installable packages.

### C.1 Repository Structure

```
bo-mcp-ui/
├── packages/
│   ├── bo-domain/          # Pure domain models
│   │   └── src/domain/     # Campaign, Suggestion, Result, User, CampaignSpec
│   │
│   ├── bo-engine/          # Bayesian optimization engine
│   │   ├── src/bo_engine/  # BoTorch models, acquisition, diagnostics
│   │   └── tests/          # Unit tests
│   │
│   ├── bo-mcp-server/      # MCP server implementation
│   │   ├── src/bo_mcp_server/ # MCP tools and resources
│   │   └── tests/          # Unit and integration tests
│   │
│   └── bo-mcp-api/         # REST API proxy
│       └── src/api/        # FastAPI routes
│
├── apps/
│   └── frontend/           # React + Vite application
│
├── scripts/                # Utility scripts
│   ├── create_test_user.py
│   ├── toy_example.py
│   ├── mcp_standalone_example.py
│   └── run_mcp_server.py
│
├── pyproject.toml          # uv workspace root
├── .python-version         # Python 3.12 (required for shap compatibility)
└── README.md
```

### C.2 Package Dependency Graph

```
bo-domain         (shared types - no deps on other packages)
     ↑
bo-engine         (BO logic - depends on bo-domain)
     ↑
bo-mcp-server     (MCP server - depends on bo-domain, bo-engine)
     ↑
bo-mcp-api        (REST API - depends on bo-mcp-server)
     ↑
frontend          (React app - calls bo-mcp-api via HTTP)
```

### C.3 Migration Status

| Task | Status | Notes |
|------|--------|-------|
| Create folder structure | ✅ Done | packages/, apps/, scripts/ created |
| Copy source files | ✅ Done | All .py files in place |
| Create pyproject.toml files | ✅ Done | All 4 packages + workspace root |
| Create package README files | ✅ Done | All 4 packages |
| Fix imports | ✅ Done | Uses `bo_engine`, `bo_mcp_server`, `api` |
| Pin Python 3.12 | ✅ Done | `.python-version` file created |
| Update shap requirement | ✅ Done | `shap>=0.50` for Python 3.12 compatibility |
| Install with uv | ✅ Done | All packages installed successfully |
| Run tests | ✅ Done | 86 tests passing |
| Linting | ✅ Done | ruff check passes |
| Type checking | ✅ Done | pyright passes |
| Update scripts | ✅ Done | New scripts in `scripts/` directory |
| Update Docker config | ✅ Done | `Dockerfile.api` and `docker-compose.yml` updated |
| Clean up old backend/ | ✅ Done | Removed legacy `backend/` and `frontend/` folders |
| Merge bo-domain into bo-mcp-server | ✅ Done | Domain models now live in `bo-mcp-server/src/domain/` |

### C.4 Current Package Structure

```
bo-mcp/
├── packages/
│   ├── bo-engine/              # Standalone BO engine (no deps on other packages)
│   │   ├── src/bo_engine/
│   │   │   ├── types.py        # Internal types (OptimizationSpec, etc.)
│   │   │   ├── models.py       # GP model creation/fitting
│   │   │   ├── acquisition.py  # qLogNEHVI acquisition function
│   │   │   ├── constraints.py  # Sum/linear constraint handling
│   │   │   ├── suggestions.py  # Suggestion generation
│   │   │   ├── diagnostics.py  # Hypervolume, Pareto front, model health
│   │   │   └── transforms.py   # Parameter transformations
│   │   └── pyproject.toml
│   │
│   ├── bo-mcp-server/          # MCP server (depends on bo-engine)
│   │   ├── src/
│   │   │   └── bo_mcp_server/  # Domain models, storage, tools, and resources
│   │   └── pyproject.toml
│   │
│   └── bo-mcp-api/             # REST API (depends on bo-mcp-server)
│       ├── src/api/
│       └── pyproject.toml
│
└── apps/frontend/              # React + TypeScript + Vite
```

### C.5 Quick Start (New Structure)

```bash
# Clone and setup
git clone <repo-url>
cd bo-mcp-ui

# Create virtual environment with Python 3.11+
uv venv --python 3.11
uv sync

# Install workspace packages
uv pip install -e packages/bo-engine -e packages/bo-mcp-server -e packages/bo-mcp-api

# Run tests
uv run pytest

# Start the API server
uv run uvicorn api.main:app --port 8000

# Run the toy example
uv run python scripts/toy_example.py
```

### C.6 Package Imports

After installation, import packages directly:

```python
# Domain models (from bo-mcp-server)
from domain import Campaign, CampaignSpec, Suggestion, Result, User

# BO engine (standalone)
from bo_engine import generate_initial_design, create_model, compute_hypervolume
from bo_engine.types import OptimizationSpec, ParameterSpec, ObjectiveSpec

# MCP server
from bo_mcp_server import mcp, create_mcp_server
from bo_mcp_server.tools.create_campaign import create_campaign

# Storage
from storage import init_database, get_session, CampaignRepository

# API
from api import create_app
```

### C.7 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access:
# - Frontend: http://localhost:3000
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

---

## Appendix D: Completed Implementation Details (v1.0.1 - v1.3)

> Moved from IMPLEMENTATION_PLAN.md on 2026-01-15

### D.1 v1.0.1 - Single-Objective Support (Completed 2026-01-14)

**Features Implemented:**
- [x] **qLogNEI acquisition function** - Single-objective noisy expected improvement
- [x] **qLogEI acquisition function** - Single-objective expected improvement (noiseless)
- [x] **SingleTaskGP model** - Direct single-task GP for single-objective optimization
- [x] **Automatic method selection** - AUTO mode selects appropriate acquisition based on n_objectives
- [x] **Single-objective diagnostics** - Best value tracking, improvement history, stagnation detection

**Key Files:**
- `packages/bo-engine/src/bo_engine/acquisition.py` - qLogNEI/qLogEI implementations
- `packages/bo-engine/src/bo_engine/models.py` - SingleTaskGP support
- `packages/bo-engine/tests/test_single_objective.py` - 13 tests

### D.2 v1.1 - Advanced Features (Completed 2026-01-14)

**Features Implemented:**
- [x] **qLogNParEGO acquisition** - Alternative multi-objective using Chebyshev scalarization
- [x] **Input Warping (Kumaraswamy CDF)** - Non-stationary objective handling
- [x] **LOO Cross-Validation** - Model quality metrics (RMSE, MAE, R²)
- [x] **AcquisitionMethod enum** - Explicit selection (AUTO, QLOGNEI, QLOGEI, QLOGNEHVI, QLOGPAREGO)

**Key Files:**
- `packages/bo-engine/tests/test_acquisition_methods.py` - 11 tests
- `packages/bo-engine/tests/test_input_warping.py` - 11 tests
- `packages/bo-engine/tests/test_loo_cv.py` - 12 tests

### D.3 v1.2 - Transparency Features (Completed 2026-01-15)

**Features Implemented:**
- [x] **Method Selection API** - `select_methods()` returns transparent choices with explanations
- [x] **TuRBO Core** - `TurboState`, `update_turbo_state()`, `get_turbo_bounds()`
- [x] **MCP Integration** - `method_selection` field in `generate_suggestions` response
- [x] **Frontend Component** - `MethodSelectionInfo.tsx` Bootstrap card

**Key Files:**
- `packages/bo-engine/src/bo_engine/method_selector.py` - Method selection logic
- `packages/bo-engine/src/bo_engine/turbo.py` - TuRBO state management
- `packages/bo-engine/tests/test_method_selector.py` - 23 tests
- `packages/bo-engine/tests/test_turbo.py` - 17 tests
- `packages/bo-mcp-server/src/bo_mcp_server/tools/generate_suggestions.py` - Updated response
- `apps/frontend/src/components/campaign/MethodSelectionInfo.tsx` - Transparency UI
- `apps/frontend/src/types/index.ts` - MethodSelection TypeScript type

### D.4 Method Selection Logic

The `select_methods()` function analyzes problem characteristics:

| Characteristic | Detection | Selection |
|---------------|-----------|-----------|
| Single-objective | `n_objectives == 1` | SingleTaskGP + qLogNEI |
| Multi-objective | `n_objectives > 1` | ModelListGP + qLogNEHVI |
| High-dimensional | `n_parameters > 20` | Recommend TuRBO |
| Initial design | `n_observations == 0` | Sobol sequence |
| Categorical params | Has CATEGORICAL type | One-hot encoding note |

**Confidence Levels:**
- `high`: Enough data for reliable predictions
- `medium`: Less than 2×n_parameters observations
- `low`: Very sparse data or edge cases

### D.5 TuRBO Implementation

Trust Region Bayesian Optimization for high-dimensional problems:

```python
@dataclass
class TurboState:
    dim: int                          # Problem dimensionality
    batch_size: int                   # Suggestions per iteration
    length: float = 0.8               # Trust region length [0,1]
    length_min: float = 0.5**7        # ~0.0078 (restart trigger)
    length_max: float = 1.6           # Maximum expansion
    failure_counter: int = 0          # Consecutive non-improvements
    failure_tolerance: int | None     # ceil(max(4/batch, dim/batch))
    success_counter: int = 0          # Consecutive improvements
    success_tolerance: int = 10       # Expand after this many
    best_value: float = float("-inf") # For maximization
    restart_triggered: bool = False   # When length < length_min
```

**State Update Rules:**
- Expand (×2) after `success_tolerance` consecutive improvements
- Contract (÷2) after `failure_tolerance` consecutive failures
- Restart when `length < length_min`

### D.6 v1.3 - Scaling Features (Completed 2026-01-15)

**Features Implemented:**
- [x] **TuRBO Integration with suggestions.py** - Full TuRBO integration in `generate_next_batch()`
- [x] **TuRBO State Persistence** - JSON serialization in Campaign entity
- [x] **Outcome Constraint Modeling** - Learn feasibility from objective thresholds
- [x] **Cost-Aware BO (EIpu)** - Expected Improvement per Unit cost (partial integration)

**Key Files:**

*TuRBO Integration:*
- `packages/bo-engine/src/bo_engine/suggestions.py` - TuRBO integration in `generate_next_batch()`
- `packages/bo-engine/src/bo_engine/turbo.py` - `get_turbo_bounds()` kernel fix
- `packages/bo-engine/tests/test_turbo_integration.py` - 9 integration tests
- `packages/bo-mcp-server/src/domain/campaign.py` - `turbo_state` field
- `packages/bo-mcp-server/src/storage/models.py` - `turbo_state_json` column
- `packages/bo-mcp-server/src/bo_mcp_server/tools/generate_suggestions.py` - State persistence

*Outcome Constraints:*
- `packages/bo-engine/src/bo_engine/types.py` - `OutcomeConstraintSpec` dataclass
- `packages/bo-engine/src/bo_engine/suggestions.py` - `_build_outcome_constraint_models()`
- `packages/bo-engine/tests/test_outcome_constraints.py` - 8 tests
- `packages/bo-mcp-server/src/domain/campaign_spec.py` - `OutcomeConstraint` domain model

*Cost-Aware BO:*
- `packages/bo-engine/src/bo_engine/types.py` - `cost` field in `ObservationData`, `use_cost_aware` flag
- `packages/bo-engine/src/bo_engine/acquisition.py` - `EIpuAcquisition` class
- `packages/bo-engine/tests/test_cost_aware.py` - 8 passing, 3 skipped

*Demo Scripts:*
- `scripts/turbo_demo.py` - TuRBO demonstration
- `scripts/outcome_constraints_demo.py` - Outcome constraint demonstration
- `scripts/cost_aware_demo.py` - Cost-aware BO demonstration
- `scripts/v12_v13_features_demo.py` - Comprehensive v1.2/v1.3 demo

### D.7 API Changes in v1.2/v1.3

**`generate_next_batch()` signature change:**
```python
def generate_next_batch(
    spec: OptimizationSpec,
    observations: list[ObservationData],
    batch_size: int | None = None,
    iteration: int = 0,
    turbo_state: TurboState | None = None,  # NEW in v1.2
) -> tuple[list[SuggestionResult], TurboState | None]:  # NEW return type
```

**New types in v1.3:**
```python
@dataclass(frozen=True)
class OutcomeConstraintSpec:
    """Specification for an outcome constraint."""
    objective_name: str
    threshold: float
    greater_than: bool = True  # True: obj >= threshold, False: obj <= threshold

@dataclass
class ObservationData:
    parameter_values: dict[str, Any]
    objective_values: dict[str, float]
    cost: float | None = None  # NEW: explicit cost for this evaluation

@dataclass(frozen=True)
class OptimizationSpec:
    ...
    use_turbo: bool = False                                    # NEW
    outcome_constraints: list[OutcomeConstraintSpec] = field(default_factory=list)  # NEW
    use_cost_aware: bool = False                               # NEW
```

### D.8 Known Limitations

**EIpu Integration:**
- Full EIpu integration with BoTorch optimizer requires additional work
- When `use_cost_aware=True` and cost data is incomplete, falls back to standard acquisition
- Tests for full EIpu workflow are marked as skipped

**TuRBO Scope:**
- TuRBO is single-objective only (per original paper)
- Multi-objective problems fall back to standard L-BFGS-B optimization

### D.9 Test Coverage Summary

| Package | Tests | Status |
|---------|-------|--------|
| bo-engine | 162 | 159 passing, 3 skipped |
| bo-mcp-server | 73 | ✅ Passing |
| **Total** | **235** | 232 passing, 3 skipped |

**Skipped Tests (EIpu integration):**
- `test_generates_suggestions_with_cost`
- `test_acquisition_function_is_eipu`
- `test_cost_aware_with_outcome_constraint`

**Pre-existing Flaky Tests:**
- 2 multi-objective tests occasionally fail due to BoTorch numerical instability

**Quality Checks:**
- ruff format: ✅
- ruff check: ✅
- pyright: 0 errors (pre-existing type issues in GPyTorch/BoTorch kernel access)

---

## Appendix E: v2.0 Advanced Features (Completed 2026-01-15)

### E.1 Overview

v2.0 introduces three major advanced features:

1. **Multi-Fidelity BO (qMFKG)** - Cheap/expensive evaluation support
2. **Transfer Learning (RGPE)** - Leverage prior campaigns
3. **SAASBO** - Sparse GP for very high dimensions (50+ params)

All features are accessible via the standalone MCP server tools.

### E.2 Multi-Fidelity Bayesian Optimization (qMFKG)

**Purpose:** Use cheap, low-fidelity evaluations to guide optimization while reserving expensive, high-fidelity evaluations for promising regions.

**Use Cases:**
- Simulations with adjustable resolution
- Lab experiments with varying precision
- Manufacturing processes with different quality levels

**Key Files:**
- `packages/bo-engine/src/bo_engine/multifidelity.py` - Core implementation
- `scripts/multifidelity_mcp_demo.py` - MCP server demonstration

**API:**
```python
@dataclass(frozen=True)
class FidelityParameterSpec:
    """Specification for a fidelity parameter."""
    name: str           # Name of the fidelity parameter
    bounds: tuple[float, float]  # (min_fidelity, max_fidelity)
    target: float       # Target fidelity for final optimization
    cost_weight: float = 1.0  # Cost scaling factor
    fixed_cost: float = 5.0   # Fixed base cost
```

**Campaign Configuration:**
```python
{
    "name": "Multi-Fidelity Campaign",
    "parameters": [...],
    "objectives": [...],
    "fidelity_parameter": {
        "name": "fidelity",
        "bounds": [0.1, 1.0],
        "target": 1.0,
        "cost_weight": 1.0,
        "fixed_cost": 5.0
    },
    "acquisition_method": "qMFKG"
}
```

**Implementation Details:**
- Uses `SingleTaskMultiFidelityGP` from BoTorch
- `qMultiFidelityKnowledgeGradient` acquisition function
- `InverseCostWeightedUtility` for cost-aware suggestions
- Projects suggestions to target fidelity for final evaluation

### E.3 Transfer Learning (RGPE)

**Purpose:** Leverage knowledge from prior optimization campaigns to accelerate optimization on new but related tasks.

**Use Cases:**
- New product variants (similar to previously optimized products)
- Seasonal reformulations (building on prior years)
- Process transfers across facilities

**Key Files:**
- `packages/bo-engine/src/bo_engine/transfer_learning.py` - RGPE implementation
- `scripts/transfer_learning_mcp_demo.py` - MCP server demonstration

**API:**
```python
@dataclass(frozen=True)
class TransferLearningSpec:
    """Specification for transfer learning."""
    prior_campaign_ids: list[str]  # IDs of prior campaigns
    num_ranking_samples: int = 512  # Samples for rank computation

@dataclass(frozen=True)
class PriorTaskData:
    """Data from a prior optimization task."""
    task_id: str
    train_x: Tensor
    train_y: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Campaign Configuration:**
```python
{
    "name": "Transfer Learning Campaign",
    "parameters": [...],
    "objectives": [...],
    "transfer_learning": {
        "prior_campaign_ids": ["uuid-of-prior-campaign"],
        "num_ranking_samples": 512
    }
}
```

**Implementation Details:**
- RGPE (Rank-weighted GP Ensemble) from Feurer et al. (ICML 2018)
- Trains base models from prior campaigns + target model
- Computes weights via leave-one-out cross-validation ranking
- Weighted ensemble prediction combines prior + target knowledge

**RGPE Weights:**
- Higher weight → model is more trusted for predictions
- Target model weight increases as target data grows
- Prior models help most in early iterations with sparse data

### E.4 SAASBO (High-Dimensional Optimization)

**Purpose:** Optimize in very high dimensions (50+ parameters) where only a subset of parameters significantly affect the objective.

**Use Cases:**
- Chemical formulations with many ingredients
- Machine learning hyperparameter tuning
- Manufacturing processes with many settings

**Key Files:**
- `packages/bo-engine/src/bo_engine/saasbo.py` - SAAS implementation
- `scripts/saasbo_mcp_demo.py` - MCP server demonstration

**API:**
```python
@dataclass(frozen=True)
class SAASBOConfig:
    """Configuration for SAASBO optimization."""
    warmup_steps: int = 256   # NUTS warmup (recommended: 256-512)
    num_samples: int = 128    # Posterior samples (recommended: 128-256)
    thinning: int = 16        # Keep every Nth sample
    disable_progbar: bool = True
```

**Campaign Configuration:**
```python
{
    "name": "High-Dimensional Campaign",
    "parameters": [...],  # 50+ parameters
    "objectives": [...],
    "use_saasbo": True,
    "acquisition_method": "SAASBO"
}
```

**Implementation Details:**
- Uses `SaasFullyBayesianSingleTaskGP` from BoTorch
- SAAS prior induces sparsity in inverse lengthscales
- Fit via NUTS (No-U-Turn Sampler) for fully Bayesian inference
- Automatically identifies important parameters

**Parameter Importance:**
- SAASBO computes importance from lengthscales
- Small lengthscale = important (sensitive to changes)
- Large lengthscale = unimportant (SAAS prior pushes these large)
- `compute_saasbo_importance()` returns normalized importance scores

**Computational Considerations:**
- NUTS sampling is O(n³) in observations
- Recommended for < 200 observations
- Each iteration may take minutes for high-dim problems
- Use `estimate_saasbo_runtime()` for time estimates

### E.5 Method Selection Updates

The `select_methods()` function now recognizes v2.0 features:

| Characteristic | Detection | Selection |
|---------------|-----------|-----------|
| Fidelity parameter | `spec.fidelity_parameter is not None` | MFGP + qMFKG |
| Transfer learning | `spec.transfer_learning is not None` | RGPE ensemble |
| Very high-dimensional | `n_parameters >= 50` | SAASBO |
| SAASBO enabled | `spec.use_saasbo == True` | SaasFullyBayesianSingleTaskGP |

### E.6 New Types in v2.0

```python
# In bo_engine/types.py
@dataclass(frozen=True)
class FidelityParameterSpec:
    name: str
    bounds: tuple[float, float]
    target: float
    cost_weight: float = 1.0
    fixed_cost: float = 5.0

@dataclass(frozen=True)
class TransferLearningSpec:
    prior_campaign_ids: list[str]
    num_ranking_samples: int = 512

@dataclass(frozen=True)
class OptimizationSpec:
    ...
    fidelity_parameter: FidelityParameterSpec | None = None  # NEW v2.0
    transfer_learning: TransferLearningSpec | None = None    # NEW v2.0
    use_saasbo: bool = False                                 # NEW v2.0

class AcquisitionMethod(str, Enum):
    ...
    QMFKG = "qMFKG"     # NEW v2.0
    SAASBO = "SAASBO"   # NEW v2.0
```

### E.7 MCP Server Updates

**Domain Models Added:**
- `FidelityParameter` - Pydantic model for fidelity configuration
- `TransferLearningConfig` - Pydantic model for transfer learning

**CampaignSpec Extended:**
```python
class CampaignSpec(BaseModel):
    ...
    fidelity_parameter: FidelityParameter | None = None
    transfer_learning: TransferLearningConfig | None = None
    use_saasbo: bool = False
```

### E.8 Demo Scripts

| Script | Feature | Usage |
|--------|---------|-------|
| `scripts/multifidelity_mcp_demo.py` | Multi-fidelity BO | `uv run python scripts/multifidelity_mcp_demo.py` |
| `scripts/transfer_learning_mcp_demo.py` | Transfer Learning | `uv run python scripts/transfer_learning_mcp_demo.py` |
| `scripts/saasbo_mcp_demo.py` | SAASBO | `uv run python scripts/saasbo_mcp_demo.py` |

All demo scripts use the standalone MCP server (no REST API or backend required).

### E.9 Decision Matrix (Updated for v2.0)

| Problem Characteristic | Model | Acquisition | Strategy |
|----------------------|-------|-------------|----------|
| 1 objective, ≤20 params | SingleTaskGP | qLogNEI | L-BFGS-B |
| 1 objective, >20 params | SingleTaskGP | qLogNEI | TuRBO |
| 1 objective, ≥50 params | SaasFullyBayesianGP | qLogEI | SAASBO |
| 2+ objectives | ModelListGP | qLogNEHVI | L-BFGS-B |
| With fidelity param | MFGP | qMFKG | Cost-aware |
| With prior campaigns | RGPE | qLogNEI | Transfer |
| With cost budget | SingleTaskGP | EIpu | L-BFGS-B |
| Categorical params | MixedSingleTaskGP | qLogNEI | Mixed |
| <2×n_params observations | Any | Sobol | Initial design |

### E.10 References

**Multi-Fidelity BO:**
- Wu & Frazier (2016) - Multi-fidelity Knowledge Gradient
- BoTorch Tutorial: https://botorch.org/docs/tutorials/multi_fidelity_bo/

**Transfer Learning (RGPE):**
- Feurer, Letham, Bakshy (ICML 2018 AutoML Workshop) - RGPE
- BoTorch Tutorial: https://botorch.org/docs/tutorials/meta_learning_with_rgpe/

**SAASBO:**
- Eriksson & Jankowiak (UAI 2021) - SAASBO
- BoTorch Tutorial: https://botorch.org/docs/tutorials/saasbo/

---

*End of Design Document*
