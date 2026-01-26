import { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Card, Row, Col, Badge, Button, Table, Form, Modal, Tabs, Tab } from 'react-bootstrap';
import {
  getCampaign,
  getCampaignSpec,
  getSuggestions,
  getResults,
  getDiagnostics,
  generateSuggestions,
  submitResults,
} from '../api/campaigns';
import type { Campaign, CampaignSpec, Suggestion, Result, DiagnosticReport } from '../types';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import ParetoChart from '../components/pareto/ParetoChart';
import HypervolumeChart from '../components/pareto/HypervolumeChart';
import HealthIndicator from '../components/diagnostics/HealthIndicator';
import WarningsAlert from '../components/diagnostics/WarningsAlert';
import ModelInfoCard from '../components/diagnostics/ModelInfoCard';
import ProgressIndicator from '../components/diagnostics/ProgressIndicator';
import FeatureImportanceChart from '../components/diagnostics/FeatureImportanceChart';
import SuggestionProvenanceDisplay from '../components/campaign/SuggestionProvenance';

function CampaignDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [spec, setSpec] = useState<CampaignSpec | null>(null);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [results, setResults] = useState<Result[]>([]);
  const [diagnostics, setDiagnostics] = useState<DiagnosticReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [showResultModal, setShowResultModal] = useState(false);
  const [selectedSuggestion, setSelectedSuggestion] = useState<Suggestion | null>(null);
  const [resultValues, setResultValues] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);

  const loadData = useCallback(async () => {
    if (!id) return;

    try {
      setLoading(true);
      setError(null);

      const campaignData = await getCampaign(id);
      setCampaign(campaignData);

      if (campaignData.spec_id) {
        const specData = await getCampaignSpec(campaignData.spec_id);
        setSpec(specData);
      }

      const [suggestionsData, resultsData, diagnosticsData] = await Promise.all([
        getSuggestions(id),
        getResults(id),
        getDiagnostics(id),
      ]);

      setSuggestions(suggestionsData);
      setResults(resultsData);
      setDiagnostics(diagnosticsData);
    } catch (err) {
      setError('Failed to load campaign data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleGenerateSuggestions = async () => {
    if (!id) return;

    try {
      setGenerating(true);
      setError(null);
      const result = await generateSuggestions(id);
      if (result.success) {
        await loadData();
      } else {
        setError(result.errors?.join(', ') || 'Failed to generate suggestions');
      }
    } catch (err) {
      setError('Failed to generate suggestions');
      console.error(err);
    } finally {
      setGenerating(false);
    }
  };

  const openResultModal = (suggestion: Suggestion) => {
    setSelectedSuggestion(suggestion);
    setResultValues({});
    setShowResultModal(true);
  };

  const handleSubmitResult = async () => {
    if (!id || !selectedSuggestion || !spec) return;

    try {
      setSubmitting(true);
      setError(null);

      const objectiveValues: Record<string, number> = {};
      for (const obj of spec.objectives) {
        const value = parseFloat(resultValues[obj.name] || '0');
        if (isNaN(value)) {
          setError(`Invalid value for ${obj.name}`);
          setSubmitting(false);
          return;
        }
        objectiveValues[obj.name] = value;
      }

      const result = await submitResults(id, {
        results: [
          {
            parameter_values: selectedSuggestion.parameter_values,
            objective_values: objectiveValues,
            suggestion_id: selectedSuggestion.id,
          },
        ],
        source: 'gui',
      });

      if (result.success) {
        setShowResultModal(false);
        await loadData();
      } else {
        setError(result.errors?.join(', ') || 'Failed to submit result');
      }
    } catch (err) {
      setError('Failed to submit result');
      console.error(err);
    } finally {
      setSubmitting(false);
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, string> = {
      created: 'secondary',
      active: 'primary',
      paused: 'warning',
      completed: 'success',
      failed: 'danger',
    };
    return <Badge bg={variants[status] || 'secondary'}>{status}</Badge>;
  };

  const isSuggestionExecuted = (suggestion: Suggestion) => {
    return results.some((r) => r.suggestion_id === suggestion.id);
  };

  if (loading) {
    return <LoadingSpinner message="Loading campaign..." />;
  }

  if (error && !campaign) {
    return <ErrorAlert title="Error" message={error} />;
  }

  if (!campaign || !spec) {
    return <ErrorAlert title="Not Found" message="Campaign not found" />;
  }

  return (
    <div>
      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {/* Header */}
      <div className="d-flex justify-content-between align-items-start mb-4">
        <div>
          <h2>{spec.name}</h2>
          <p className="text-muted mb-0">{spec.description}</p>
        </div>
        <div className="text-end">
          {getStatusBadge(campaign.status)}
          <div className="mt-2">
            <small className="text-muted">
              Iteration {campaign.iteration} | {results.length} results
            </small>
          </div>
        </div>
      </div>

      {/* Warnings and Alerts */}
      {diagnostics && diagnostics.warnings && diagnostics.warnings.length > 0 && (
        <WarningsAlert
          warnings={diagnostics.warnings}
          healthStatus={diagnostics.health_status}
        />
      )}

      {/* Diagnostics Summary */}
      {diagnostics && (
        <Card className="mb-4">
          <Card.Body>
            <Row>
              <Col md={2}>
                <HealthIndicator status={diagnostics.health_status || 'healthy'} />
              </Col>
              <Col md={2}>
                <ProgressIndicator status={diagnostics.progress_status} />
              </Col>
              <Col md={2}>
                <div className="text-center">
                  <h4 className="mb-0">{diagnostics.n_results ?? 0}</h4>
                  <small className="text-muted">Total Results</small>
                </div>
              </Col>
              <Col md={2}>
                <div className="text-center">
                  <h4 className="mb-0">{diagnostics.n_pareto_points ?? 0}</h4>
                  <small className="text-muted">Pareto Points</small>
                </div>
              </Col>
              <Col md={2}>
                <div className="text-center">
                  <h4 className="mb-0">{(diagnostics.hypervolume ?? 0).toFixed(2)}</h4>
                  <small className="text-muted">Hypervolume</small>
                </div>
              </Col>
              <Col md={2}>
                <div className="text-center">
                  <h4 className="mb-0">{diagnostics.iteration ?? 0}</h4>
                  <small className="text-muted">Iteration</small>
                </div>
              </Col>
            </Row>
          </Card.Body>
        </Card>
      )}

      {/* Main Content Tabs */}
      <Tabs defaultActiveKey="suggestions" className="mb-4">
        <Tab eventKey="suggestions" title="Suggestions">
          <Card>
            <Card.Header className="d-flex justify-content-between align-items-center">
              <span>Current Suggestions</span>
              <Button
                variant="primary"
                size="sm"
                onClick={handleGenerateSuggestions}
                disabled={generating}
              >
                {generating ? 'Generating...' : 'Generate New Suggestions'}
              </Button>
            </Card.Header>
            <Card.Body>
              {suggestions.length === 0 ? (
                <p className="text-muted text-center py-4">
                  No suggestions yet. Click "Generate New Suggestions" to start.
                </p>
              ) : (
                <div>
                  {suggestions.map((suggestion, index) => (
                    <Card key={suggestion.id} className="mb-3">
                      <Card.Header className="d-flex justify-content-between align-items-center">
                        <span>
                          <strong>Suggestion #{index + 1}</strong>
                          <span className="ms-3">
                            {isSuggestionExecuted(suggestion) ? (
                              <Badge bg="success">Executed</Badge>
                            ) : (
                              <Badge bg="warning">Pending</Badge>
                            )}
                          </span>
                          {suggestion.provenance?.confidence_level && (
                            <Badge
                              bg={
                                suggestion.provenance.confidence_level === 'high'
                                  ? 'success'
                                  : suggestion.provenance.confidence_level === 'medium'
                                  ? 'warning'
                                  : 'danger'
                              }
                              className="ms-2"
                            >
                              {suggestion.provenance.confidence_level} confidence
                            </Badge>
                          )}
                        </span>
                        {!isSuggestionExecuted(suggestion) && (
                          <Button
                            variant="outline-primary"
                            size="sm"
                            onClick={() => openResultModal(suggestion)}
                          >
                            Submit Result
                          </Button>
                        )}
                      </Card.Header>
                      <Card.Body>
                        <Table size="sm" className="mb-0">
                          <thead>
                            <tr>
                              {spec.parameters.map((p) => (
                                <th key={p.name}>{p.name}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              {spec.parameters.map((p) => (
                                <td key={p.name}>
                                  <code className="parameter-badge">
                                    {typeof suggestion.parameter_values[p.name] === 'number'
                                      ? (suggestion.parameter_values[p.name] as number).toFixed(4)
                                      : suggestion.parameter_values[p.name]}
                                  </code>
                                </td>
                              ))}
                            </tr>
                          </tbody>
                        </Table>
                        {suggestion.provenance && (
                          <SuggestionProvenanceDisplay provenance={suggestion.provenance} />
                        )}
                      </Card.Body>
                    </Card>
                  ))}
                </div>
              )}
            </Card.Body>
          </Card>
        </Tab>

        <Tab eventKey="results" title="Results">
          <Card>
            <Card.Header>Experimental Results</Card.Header>
            <Card.Body>
              {results.length === 0 ? (
                <p className="text-muted text-center py-4">No results submitted yet.</p>
              ) : (
                <Table responsive hover>
                  <thead>
                    <tr>
                      <th>#</th>
                      {spec.parameters.map((p) => (
                        <th key={p.name}>{p.name}</th>
                      ))}
                      {spec.objectives.map((o) => (
                        <th key={o.name}>
                          {o.name} {o.unit && `(${o.unit})`}
                        </th>
                      ))}
                      <th>Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((result, index) => (
                      <tr key={result.id}>
                        <td>{index + 1}</td>
                        {spec.parameters.map((p) => (
                          <td key={p.name}>
                            <code className="parameter-badge">
                              {typeof result.parameter_values[p.name] === 'number'
                                ? (result.parameter_values[p.name] as number).toFixed(2)
                                : result.parameter_values[p.name]}
                            </code>
                          </td>
                        ))}
                        {spec.objectives.map((o) => (
                          <td key={o.name}>
                            <strong>{result.objective_values[o.name]?.toFixed(2)}</strong>
                          </td>
                        ))}
                        <td>
                          <Badge bg="secondary">{result.source}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>
        </Tab>

        <Tab eventKey="pareto" title="Pareto Front">
          <Row>
            <Col md={8}>
              <Card>
                <Card.Header>Pareto Front Visualization</Card.Header>
                <Card.Body>
                  {(diagnostics?.pareto_front && diagnostics.pareto_front.length > 0) || results.length > 0 ? (
                    <ParetoChart
                      paretoFront={diagnostics?.pareto_front || []}
                      objectives={spec.objectives}
                      allResults={results}
                    />
                  ) : (
                    <p className="text-muted text-center py-4">
                      Not enough data to display Pareto front. Submit more results.
                    </p>
                  )}
                </Card.Body>
              </Card>
            </Col>
            <Col md={4}>
              <Card>
                <Card.Header>Hypervolume Progress</Card.Header>
                <Card.Body>
                  <HypervolumeChart
                    hypervolume={diagnostics?.hypervolume ?? 0}
                    iteration={campaign.iteration}
                  />
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </Tab>

        <Tab eventKey="config" title="Configuration">
          <Card>
            <Card.Header>Campaign Configuration</Card.Header>
            <Card.Body>
              <Row>
                <Col md={6}>
                  <h6>Parameters</h6>
                  <Table size="sm">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Range/Values</th>
                      </tr>
                    </thead>
                    <tbody>
                      {spec.parameters.map((p) => (
                        <tr key={p.name}>
                          <td>{p.name}</td>
                          <td>
                            <Badge bg="info">{p.type}</Badge>
                          </td>
                          <td>
                            {p.type === 'categorical'
                              ? p.categories?.join(', ')
                              : `[${p.bounds?.[0]}, ${p.bounds?.[1]}]`}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </Col>
                <Col md={6}>
                  <h6>Objectives</h6>
                  <Table size="sm">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Direction</th>
                        <th>Unit</th>
                      </tr>
                    </thead>
                    <tbody>
                      {spec.objectives.map((o) => (
                        <tr key={o.name}>
                          <td>{o.name}</td>
                          <td>
                            <Badge bg={o.direction === 'maximize' ? 'success' : 'warning'}>
                              {o.direction}
                            </Badge>
                          </td>
                          <td>{o.unit || '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </Col>
              </Row>
              <hr />
              <Row>
                <Col md={6}>
                  <p className="mb-1">
                    <strong>Batch Size:</strong> {spec.batch_size}
                  </p>
                  {spec.created_at && (
                    <p className="mb-0">
                      <strong>Created:</strong> {new Date(spec.created_at).toLocaleString()}
                    </p>
                  )}
                </Col>
                <Col md={6}>
                  <ModelInfoCard modelInfo={diagnostics?.model_info} />
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Tab>

        <Tab eventKey="advanced" title="Advanced">
          <Card className="mb-4">
            <Card.Header>
              <strong>Confidence & Model Quality Metrics</strong>
            </Card.Header>
            <Card.Body>
              <p className="text-muted mb-4">
                These metrics help assess whether the Bayesian Optimization is working effectively
                and how much to trust the suggestions.
              </p>

              <Row>
                <Col md={6}>
                  <h6>Suggestion Confidence Levels</h6>
                  <Table size="sm">
                    <thead>
                      <tr>
                        <th>Level</th>
                        <th>Model Uncertainty</th>
                        <th>Interpretation</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><Badge bg="success">High</Badge></td>
                        <td>&lt; 0.1</td>
                        <td>Model is confident about this region. Suggestion likely exploits known good areas.</td>
                      </tr>
                      <tr>
                        <td><Badge bg="warning">Medium</Badge></td>
                        <td>0.1 - 0.3</td>
                        <td>Moderate uncertainty. Balances exploration and exploitation.</td>
                      </tr>
                      <tr>
                        <td><Badge bg="danger">Low</Badge></td>
                        <td>&gt; 0.3</td>
                        <td>High uncertainty. Suggestion explores unknown regions of the parameter space.</td>
                      </tr>
                    </tbody>
                  </Table>

                  <h6 className="mt-4">Health Status Criteria</h6>
                  <Table size="sm">
                    <thead>
                      <tr>
                        <th>Status</th>
                        <th>Criteria</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><Badge bg="success">Healthy</Badge></td>
                        <td>Optimization is progressing normally. Hypervolume increasing.</td>
                      </tr>
                      <tr>
                        <td><Badge bg="warning">Warning</Badge></td>
                        <td>3+ iterations without improvement, or model correlation &lt; 0.3</td>
                      </tr>
                      <tr>
                        <td><Badge bg="danger">Critical</Badge></td>
                        <td>5+ iterations without improvement, or model predictions uncorrelated with results</td>
                      </tr>
                    </tbody>
                  </Table>
                </Col>

                <Col md={6}>
                  <h6>Current Diagnostics</h6>
                  {diagnostics && (
                    <Table size="sm">
                      <tbody>
                        <tr>
                          <td><strong>Health Status</strong></td>
                          <td>
                            <Badge bg={
                              diagnostics.health_status === 'healthy' ? 'success' :
                              diagnostics.health_status === 'warning' ? 'warning' : 'danger'
                            }>
                              {diagnostics.health_status || 'unknown'}
                            </Badge>
                          </td>
                        </tr>
                        <tr>
                          <td><strong>Progress Status</strong></td>
                          <td>
                            <Badge bg={
                              diagnostics.progress_status === 'improving' ? 'success' :
                              diagnostics.progress_status === 'stagnant' ? 'warning' : 'danger'
                            }>
                              {diagnostics.progress_status || 'unknown'}
                            </Badge>
                          </td>
                        </tr>
                        <tr>
                          <td><strong>Iteration</strong></td>
                          <td>{diagnostics.iteration}</td>
                        </tr>
                        <tr>
                          <td><strong>Total Results</strong></td>
                          <td>{diagnostics.n_results}</td>
                        </tr>
                        <tr>
                          <td><strong>Pareto Points</strong></td>
                          <td>{diagnostics.n_pareto_points ?? 0}</td>
                        </tr>
                        <tr>
                          <td><strong>Hypervolume</strong></td>
                          <td>{(diagnostics.hypervolume ?? 0).toFixed(4)}</td>
                        </tr>
                        <tr>
                          <td><strong>Pending Suggestions</strong></td>
                          <td>{diagnostics.n_pending_suggestions ?? 0}</td>
                        </tr>
                      </tbody>
                    </Table>
                  )}

                  <h6 className="mt-4">Objective Ranges</h6>
                  {diagnostics?.objective_ranges && Object.keys(diagnostics.objective_ranges).length > 0 ? (
                    <Table size="sm">
                      <thead>
                        <tr>
                          <th>Objective</th>
                          <th>Min</th>
                          <th>Max</th>
                          <th>Direction</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(diagnostics.objective_ranges).map(([name, range]) => (
                          <tr key={name}>
                            <td>{name}</td>
                            <td>{range.min.toFixed(2)}</td>
                            <td>{range.max.toFixed(2)}</td>
                            <td>
                              <Badge bg={range.direction === 'maximize' ? 'success' : 'warning'}>
                                {range.direction}
                              </Badge>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  ) : (
                    <p className="text-muted">No results yet to calculate ranges.</p>
                  )}
                </Col>
              </Row>

              {/* Feature Importance */}
              {diagnostics?.feature_importance && (
                <Row className="mt-4">
                  <Col md={12}>
                    <FeatureImportanceChart featureImportance={diagnostics.feature_importance} />
                  </Col>
                </Row>
              )}

              <hr />

              <h6>Understanding the Metrics</h6>
              <Row>
                <Col md={6}>
                  <Card className="bg-light">
                    <Card.Body>
                      <h6>Hypervolume</h6>
                      <p className="small mb-0">
                        Measures the volume of objective space dominated by the Pareto front.
                        <strong> Higher is better.</strong> An increasing hypervolume indicates
                        the optimization is finding better trade-offs between objectives.
                      </p>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={6}>
                  <Card className="bg-light">
                    <Card.Body>
                      <h6>Model Uncertainty</h6>
                      <p className="small mb-0">
                        Predicted variance from the Gaussian Process model.
                        <strong> Lower means more confidence.</strong> High uncertainty regions
                        are explored to improve the model's understanding of the objective landscape.
                      </p>
                    </Card.Body>
                  </Card>
                </Col>
              </Row>
              <Row className="mt-3">
                <Col md={6}>
                  <Card className="bg-light">
                    <Card.Body>
                      <h6>Acquisition Value (Expected Improvement)</h6>
                      <p className="small mb-0">
                        The qLogNEHVI acquisition function estimates how much a suggestion
                        will expand the Pareto front. <strong>Higher values suggest more promising points.</strong>
                      </p>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={6}>
                  <Card className="bg-light">
                    <Card.Body>
                      <h6>Pareto Optimality</h6>
                      <p className="small mb-0">
                        A point is Pareto optimal if no other point is better in all objectives.
                        <strong> More Pareto points = more diverse trade-off options.</strong>
                        Dominated points are worse than at least one Pareto point in every objective.
                      </p>
                    </Card.Body>
                  </Card>
                </Col>
              </Row>

              {diagnostics?.model_info && (
                <>
                  <hr />
                  <h6>Optimization Algorithm Details</h6>
                  <Table size="sm">
                    <tbody>
                      <tr>
                        <td><strong>Surrogate Model</strong></td>
                        <td>{diagnostics.model_info.type}</td>
                      </tr>
                      <tr>
                        <td><strong>Acquisition Function</strong></td>
                        <td>{diagnostics.model_info.acquisition_function}</td>
                      </tr>
                      <tr>
                        <td><strong>Batch Strategy</strong></td>
                        <td>{diagnostics.model_info.batch_strategy}</td>
                      </tr>
                      <tr>
                        <td><strong>Kernel</strong></td>
                        <td>{diagnostics.model_info.kernel}</td>
                      </tr>
                    </tbody>
                  </Table>
                  <p className="small text-muted">
                    The Gaussian Process model learns a probabilistic mapping from parameters to objectives.
                    The Matérn 5/2 kernel captures smooth relationships with automatic relevance determination
                    to identify which parameters most influence each objective.
                  </p>
                </>
              )}
            </Card.Body>
          </Card>
        </Tab>
      </Tabs>

      {/* Result Submission Modal */}
      <Modal show={showResultModal} onHide={() => setShowResultModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Submit Result</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedSuggestion && spec && (
            <>
              <h6>Parameters</h6>
              <Table size="sm" className="mb-4">
                <tbody>
                  {spec.parameters.map((p) => (
                    <tr key={p.name}>
                      <td>{p.name}</td>
                      <td>
                        <code>
                          {typeof selectedSuggestion.parameter_values[p.name] === 'number'
                            ? (selectedSuggestion.parameter_values[p.name] as number).toFixed(2)
                            : selectedSuggestion.parameter_values[p.name]}
                        </code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <h6>Objective Values</h6>
              {spec.objectives.map((obj) => (
                <Form.Group key={obj.name} className="mb-3">
                  <Form.Label>
                    {obj.name} {obj.unit && `(${obj.unit})`} - {obj.direction}
                  </Form.Label>
                  <Form.Control
                    type="number"
                    step="any"
                    placeholder={`Enter ${obj.name} value`}
                    value={resultValues[obj.name] || ''}
                    onChange={(e) =>
                      setResultValues({ ...resultValues, [obj.name]: e.target.value })
                    }
                  />
                </Form.Group>
              ))}
            </>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowResultModal(false)}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleSubmitResult} disabled={submitting}>
            {submitting ? 'Submitting...' : 'Submit Result'}
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
}

export default CampaignDetailPage;
