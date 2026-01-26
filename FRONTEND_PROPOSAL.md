# Frontend Implementation Proposal: v1.0.1 + v1.1 Features

**Version**: 1.0
**Status**: Proposal
**Date**: 2026-01-14

This document outlines the frontend changes required to expose the new BO engine features (v1.0.1 single-objective support and v1.1 advanced features) in the React web UI.

---

## 1. Summary of Backend Changes

### v1.0.1 - Single-Objective Support
| Feature | Backend Implementation | Frontend Impact |
|---------|----------------------|-----------------|
| qLogNEI acquisition | `bo_engine/acquisition.py` | Campaign creation form |
| SingleTaskGP model | `bo_engine/models.py` | Diagnostics display |
| Single-objective diagnostics | `bo_engine/diagnostics.py` | New diagnostic charts |
| Best value tracking | `compute_best_value()` | Convergence visualization |

### v1.1 - Advanced Features
| Feature | Backend Implementation | Frontend Impact |
|---------|----------------------|-----------------|
| qLogNParEGO acquisition | `bo_engine/acquisition.py` | Acquisition selector |
| Input warping | `bo_engine/models.py` | Optional toggle in form |
| LOO Cross-Validation | `bo_engine/diagnostics.py` | Model quality cards |
| AcquisitionMethod enum | `bo_engine/types.py` | Dropdown selector |

---

## 2. Campaign Creation Form Updates

### 2.1 Acquisition Method Selector

**Location**: `apps/frontend/src/pages/IntakePage.tsx`

Add a new dropdown to select acquisition method:

```tsx
// New component: AcquisitionMethodSelector.tsx
interface AcquisitionMethodSelectorProps {
  value: AcquisitionMethod;
  onChange: (method: AcquisitionMethod) => void;
  nObjectives: number;
}

const acquisitionMethods = [
  { value: 'auto', label: 'Auto (Recommended)', description: 'Automatically selects based on objectives' },
  { value: 'qLogNEI', label: 'qLogNEI', description: 'Single-objective: Noisy Expected Improvement' },
  { value: 'qLogEI', label: 'qLogEI', description: 'Single-objective: Expected Improvement (noiseless)' },
  { value: 'qLogNEHVI', label: 'qLogNEHVI', description: 'Multi-objective: Hypervolume Improvement' },
  { value: 'qLogNParEGO', label: 'qLogNParEGO', description: 'Multi-objective: Chebyshev Scalarization' },
];

// Filter based on n_objectives
const availableMethods = nObjectives === 1
  ? acquisitionMethods.filter(m => ['auto', 'qLogNEI', 'qLogEI'].includes(m.value))
  : acquisitionMethods.filter(m => ['auto', 'qLogNEHVI', 'qLogNParEGO'].includes(m.value));
```

**UI Design**:
```
┌─────────────────────────────────────────────────────────┐
│ Advanced Settings                              [▼ Hide] │
├─────────────────────────────────────────────────────────┤
│ Acquisition Method                                      │
│ ┌─────────────────────────────────────────────────┐    │
│ │ Auto (Recommended)                           ▼ │    │
│ └─────────────────────────────────────────────────┘    │
│ ⓘ Automatically selects qLogNEI for single-objective   │
│   or qLogNEHVI for multi-objective                     │
│                                                         │
│ ☑ Enable Input Warping                                 │
│ ⓘ Kumaraswamy warping helps with non-stationary        │
│   objectives where response varies across regions      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 API Schema Updates

**File**: `apps/frontend/src/api/types.ts`

```typescript
// Add new types
export type AcquisitionMethod =
  | 'auto'
  | 'qLogNEI'
  | 'qLogEI'
  | 'qLogNEHVI'
  | 'qLogNParEGO';

export interface CampaignIntake {
  name: string;
  description?: string;
  parameters: ParameterSpec[];
  objectives: ObjectiveSpec[];
  constraints?: ConstraintSpec[];
  batch_size: number;
  // New v1.0.1/v1.1 fields
  acquisition_method?: AcquisitionMethod;  // Default: 'auto'
  use_input_warping?: boolean;             // Default: false
}
```

---

## 3. Single-Objective Diagnostics Display

### 3.1 Convergence Chart Component

**New Component**: `apps/frontend/src/components/diagnostics/ConvergenceChart.tsx`

For single-objective optimization, display:
- Best value history (running best over iterations)
- Improvement rate indicator
- Stagnation detection warning

```tsx
interface ConvergenceChartProps {
  improvementHistory: number[];
  objectiveName: string;
  minimize: boolean;
}

const ConvergenceChart: React.FC<ConvergenceChartProps> = ({
  improvementHistory,
  objectiveName,
  minimize,
}) => {
  const data = [{
    x: Array.from({ length: improvementHistory.length }, (_, i) => i + 1),
    y: improvementHistory,
    type: 'scatter',
    mode: 'lines+markers',
    name: `Best ${objectiveName}`,
    line: { color: '#007bff' },
  }];

  const layout = {
    title: `Optimization Progress: ${objectiveName}`,
    xaxis: { title: 'Iteration' },
    yaxis: { title: minimize ? 'Best Value (↓ better)' : 'Best Value (↑ better)' },
  };

  return <Plot data={data} layout={layout} />;
};
```

**UI Design**:
```
┌─────────────────────────────────────────────────────────┐
│ Optimization Progress                                   │
├─────────────────────────────────────────────────────────┤
│ Best value: 0.412                                       │
│ Improvement rate: 15.3%                                 │
│                                                         │
│   ▲                                                     │
│ 2.0├──●                                                 │
│    │   ╲                                                │
│ 1.5├────●──●                                            │
│    │        ╲                                           │
│ 1.0├─────────●──●──●                                    │
│    │              ╲                                     │
│ 0.5├───────────────●──●──●                              │
│    └───┬───┬───┬───┬───┬───┬───┬──►                     │
│        1   2   3   4   5   6   7   Iteration            │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Conditional Rendering

Update `DiagnosticsPage.tsx` to show different charts based on objective count:

```tsx
const DiagnosticsPage: React.FC<{ campaign: Campaign }> = ({ campaign }) => {
  const isSingleObjective = campaign.objectives.length === 1;

  return (
    <div className="diagnostics-container">
      {isSingleObjective ? (
        <>
          <ConvergenceChart
            improvementHistory={campaign.diagnostics.improvement_history}
            objectiveName={campaign.objectives[0].name}
            minimize={campaign.objectives[0].minimize}
          />
          <SingleObjectiveMetrics diagnostics={campaign.diagnostics} />
        </>
      ) : (
        <>
          <ParetoChart paretoFront={campaign.diagnostics.pareto_front} />
          <HypervolumeChart history={campaign.diagnostics.hypervolume_history} />
        </>
      )}
      <ModelQualityCard metrics={campaign.diagnostics.loo_cv_metrics} />
    </div>
  );
};
```

---

## 4. Model Quality (LOO-CV) Display

### 4.1 Model Quality Card Component

**New Component**: `apps/frontend/src/components/diagnostics/ModelQualityCard.tsx`

```tsx
interface LOOCVMetrics {
  rmse: number;
  mae: number;
  r_squared: number;
}

interface ModelQualityCardProps {
  metrics: Record<string, LOOCVMetrics>;
  objectiveNames: string[];
}

const ModelQualityCard: React.FC<ModelQualityCardProps> = ({
  metrics,
  objectiveNames,
}) => {
  const getQualityLevel = (r2: number): 'excellent' | 'good' | 'moderate' | 'poor' => {
    if (r2 > 0.9) return 'excellent';
    if (r2 > 0.7) return 'good';
    if (r2 > 0.5) return 'moderate';
    return 'poor';
  };

  return (
    <Card>
      <Card.Header>
        <h5>Model Quality (LOO Cross-Validation)</h5>
      </Card.Header>
      <Card.Body>
        <Table striped bordered>
          <thead>
            <tr>
              <th>Objective</th>
              <th>R²</th>
              <th>RMSE</th>
              <th>Quality</th>
            </tr>
          </thead>
          <tbody>
            {objectiveNames.map((name, i) => {
              const m = metrics[i];
              const quality = getQualityLevel(m.r_squared);
              return (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{m.r_squared.toFixed(3)}</td>
                  <td>{m.rmse.toFixed(4)}</td>
                  <td>
                    <Badge bg={qualityColors[quality]}>{quality}</Badge>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      </Card.Body>
    </Card>
  );
};
```

**UI Design**:
```
┌─────────────────────────────────────────────────────────┐
│ Model Quality (LOO Cross-Validation)                    │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────┬────────┬────────┬──────────────┐       │
│ │ Objective   │ R²     │ RMSE   │ Quality      │       │
│ ├─────────────┼────────┼────────┼──────────────┤       │
│ │ yield       │ 0.923  │ 0.0342 │ 🟢 Excellent │       │
│ │ cost        │ 0.781  │ 0.1234 │ 🟡 Good      │       │
│ └─────────────┴────────┴────────┴──────────────┘       │
│                                                         │
│ ⓘ R² > 0.9 = Excellent | 0.7-0.9 = Good |              │
│   0.5-0.7 = Moderate | < 0.5 = Poor                    │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Suggestion Card Updates

### 5.1 Acquisition Function Display

Update suggestion cards to show acquisition method:

**File**: `apps/frontend/src/components/campaign/SuggestionCard.tsx`

```tsx
const SuggestionCard: React.FC<{ suggestion: Suggestion }> = ({ suggestion }) => {
  return (
    <Card className="suggestion-card">
      <Card.Header>
        <span>Suggestion #{suggestion.batch_index + 1}</span>
        {suggestion.acquisition_function && (
          <Badge bg="info" className="ms-2">
            {suggestion.acquisition_function}
          </Badge>
        )}
      </Card.Header>
      <Card.Body>
        {/* Parameter values */}
        <div className="parameters">
          {Object.entries(suggestion.parameter_values).map(([name, value]) => (
            <div key={name} className="parameter-row">
              <span className="param-name">{name}:</span>
              <span className="param-value">{formatValue(value)}</span>
            </div>
          ))}
        </div>

        {/* Provenance info (collapsible) */}
        <Accordion>
          <Accordion.Item eventKey="0">
            <Accordion.Header>Why this suggestion?</Accordion.Header>
            <Accordion.Body>
              <p>{suggestion.explanation}</p>
              <small className="text-muted">
                Model: {suggestion.model_type} |
                Confidence: {suggestion.confidence_level} |
                Uncertainty: {suggestion.model_uncertainty?.toFixed(4)}
              </small>
            </Accordion.Body>
          </Accordion.Item>
        </Accordion>
      </Card.Body>
    </Card>
  );
};
```

---

## 6. API Endpoint Updates

The REST API needs to expose the new fields. Update the API schemas:

**File**: `packages/bo-mcp-api/src/api/schemas/intake.py`

```python
from enum import Enum

class AcquisitionMethod(str, Enum):
    AUTO = "auto"
    QLOGNEI = "qLogNEI"
    QLOGEI = "qLogEI"
    QLOGNEHVI = "qLogNEHVI"
    QLOGPAREGO = "qLogNParEGO"

class CampaignIntake(BaseModel):
    name: str
    description: str | None = None
    parameters: list[ParameterSpec]
    objectives: list[ObjectiveSpec]
    constraints: list[ConstraintSpec] | None = None
    batch_size: int = 3
    # New v1.0.1/v1.1 fields
    acquisition_method: AcquisitionMethod = AcquisitionMethod.AUTO
    use_input_warping: bool = False
```

**File**: `packages/bo-mcp-api/src/api/schemas/diagnostics.py`

```python
class LOOCVMetrics(BaseModel):
    rmse: float
    mae: float
    r_squared: float

class SingleObjectiveDiagnostics(BaseModel):
    best_value: float
    best_parameters: dict[str, float]
    improvement_history: list[float]
    improvement_rate: float
    health_status: str
    warnings: list[str]

class DiagnosticsResponse(BaseModel):
    # Existing multi-objective fields
    hypervolume: float | None = None
    hypervolume_history: list[float] | None = None
    pareto_front: list[dict] | None = None
    n_pareto_points: int | None = None

    # New single-objective fields (v1.0.1)
    best_value: float | None = None
    best_parameters: dict | None = None
    improvement_history: list[float] | None = None
    improvement_rate: float | None = None

    # Model quality (v1.1)
    loo_cv_metrics: dict[str, LOOCVMetrics] | None = None

    # Common fields
    n_results: int
    health_status: str
    warnings: list[str]
```

---

## 7. Implementation Phases

### Phase 1: Core Form Updates (1-2 days)
1. Add `AcquisitionMethodSelector` component
2. Add `use_input_warping` toggle to advanced settings
3. Update `CampaignIntake` TypeScript types
4. Wire up form submission with new fields

### Phase 2: Diagnostics Display (2-3 days)
1. Create `ConvergenceChart` for single-objective
2. Create `ModelQualityCard` for LOO-CV metrics
3. Add conditional rendering in `DiagnosticsPage`
4. Update API response types

### Phase 3: Suggestion Cards (1 day)
1. Update `SuggestionCard` with acquisition function badge
2. Add collapsible explanation section
3. Display confidence level and uncertainty

### Phase 4: Testing & Polish (1-2 days)
1. Add component tests (Vitest)
2. Test form validation for acquisition method
3. Responsive design adjustments
4. Accessibility review

---

## 8. File Checklist

### New Files to Create
- [ ] `apps/frontend/src/components/intake/AcquisitionMethodSelector.tsx`
- [ ] `apps/frontend/src/components/intake/InputWarpingToggle.tsx`
- [ ] `apps/frontend/src/components/diagnostics/ConvergenceChart.tsx`
- [ ] `apps/frontend/src/components/diagnostics/ModelQualityCard.tsx`
- [ ] `apps/frontend/src/components/diagnostics/SingleObjectiveMetrics.tsx`

### Files to Modify
- [ ] `apps/frontend/src/api/types.ts` - Add new types
- [ ] `apps/frontend/src/pages/IntakePage.tsx` - Add new form fields
- [ ] `apps/frontend/src/pages/CampaignDetailPage.tsx` - Conditional diagnostics
- [ ] `apps/frontend/src/components/campaign/SuggestionCard.tsx` - Show acquisition
- [ ] `packages/bo-mcp-api/src/api/schemas/intake.py` - New schema fields
- [ ] `packages/bo-mcp-api/src/api/schemas/diagnostics.py` - New response fields
- [ ] `packages/bo-mcp-api/src/api/routes/diagnostics.py` - Return LOO-CV metrics

---

## 9. Dependencies

No new npm packages required. Uses existing:
- `react-plotly.js` for charts
- `react-bootstrap` for UI components
- Existing TypeScript setup

---

## 10. Backward Compatibility

All new fields have defaults, ensuring backward compatibility:
- `acquisition_method` defaults to `'auto'`
- `use_input_warping` defaults to `false`
- Existing multi-objective campaigns continue to work unchanged
- Diagnostics response includes both single and multi-objective fields (null if not applicable)
