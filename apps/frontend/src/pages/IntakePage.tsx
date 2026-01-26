import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Form, Button, Alert, Row, Col } from 'react-bootstrap';
import { createCampaign, validateIntake } from '../api/campaigns';
import type { InputParameter, Objective, Constraint, ParameterType, OptimizationDirection } from '../types';
import ErrorAlert from '../components/common/ErrorAlert';

function IntakePage() {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [batchSize, setBatchSize] = useState(3);
  const [parameters, setParameters] = useState<InputParameter[]>([
    { name: '', type: 'continuous', bounds: [0, 1] },
  ]);
  const [objectives, setObjectives] = useState<Objective[]>([
    { name: '', direction: 'maximize' },
  ]);
  const [constraints, setConstraints] = useState<Constraint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);

  const addParameter = () => {
    setParameters([...parameters, { name: '', type: 'continuous', bounds: [0, 1] }]);
  };

  const removeParameter = (index: number) => {
    setParameters(parameters.filter((_, i) => i !== index));
  };

  const updateParameter = (index: number, field: keyof InputParameter, value: unknown) => {
    const updated = [...parameters];
    if (field === 'type') {
      const paramType = value as ParameterType;
      updated[index] = {
        name: updated[index].name,
        type: paramType,
        ...(paramType === 'continuous' ? { bounds: [0, 1] as [number, number] } : {}),
        ...(paramType === 'discrete' ? { bounds: [0, 10] as [number, number] } : {}),
        ...(paramType === 'categorical' ? { categories: ['A', 'B'] } : {}),
      };
    } else if (field === 'name') {
      updated[index] = { ...updated[index], name: value as string };
    } else if (field === 'bounds') {
      updated[index] = { ...updated[index], bounds: value as [number, number] };
    } else if (field === 'categories') {
      updated[index] = { ...updated[index], categories: value as string[] };
    }
    setParameters(updated);
  };

  const addObjective = () => {
    setObjectives([...objectives, { name: '', direction: 'maximize' }]);
  };

  const removeObjective = (index: number) => {
    setObjectives(objectives.filter((_, i) => i !== index));
  };

  const updateObjective = (index: number, field: keyof Objective, value: unknown) => {
    const updated = [...objectives];
    if (field === 'name') {
      updated[index] = { ...updated[index], name: value as string };
    } else if (field === 'direction') {
      updated[index] = { ...updated[index], direction: value as OptimizationDirection };
    } else if (field === 'unit') {
      updated[index] = { ...updated[index], unit: value as string };
    }
    setObjectives(updated);
  };

  const addSumConstraint = () => {
    setConstraints([...constraints, { type: 'sum_equals', parameters: [], value: 1 }]);
  };

  const removeConstraint = (index: number) => {
    setConstraints(constraints.filter((_, i) => i !== index));
  };

  const updateConstraint = (index: number, field: keyof Constraint, value: unknown) => {
    const updated = [...constraints];
    if (field === 'type') {
      updated[index] = { ...updated[index], type: value as Constraint['type'] };
    } else if (field === 'parameters') {
      updated[index] = { ...updated[index], parameters: value as string[] };
    } else if (field === 'value') {
      updated[index] = { ...updated[index], value: value as number };
    }
    setConstraints(updated);
  };

  const handleValidate = async () => {
    setValidationErrors([]);
    setWarnings([]);
    setError(null);

    const intake = {
      name,
      description,
      parameters: parameters.filter((p) => p.name),
      objectives: objectives.filter((o) => o.name),
      constraints: constraints.length > 0 ? constraints : undefined,
      batch_size: batchSize,
    };

    try {
      const result = await validateIntake(intake);
      if (!result.valid) {
        setValidationErrors(result.errors || []);
      }
      if (result.warnings && result.warnings.length > 0) {
        setWarnings(result.warnings);
      }
      return result.valid;
    } catch (err) {
      setError('Failed to validate intake');
      console.error(err);
      return false;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const isValid = await handleValidate();
    if (!isValid) {
      setLoading(false);
      return;
    }

    const intake = {
      name,
      description,
      parameters: parameters.filter((p) => p.name),
      objectives: objectives.filter((o) => o.name),
      constraints: constraints.length > 0 ? constraints : undefined,
      batch_size: batchSize,
    };

    try {
      const result = await createCampaign(intake);
      if (result.success && result.campaign_id) {
        navigate(`/campaign/${result.campaign_id}`);
      } else {
        setValidationErrors(result.errors || ['Failed to create campaign']);
      }
    } catch (err) {
      setError('Failed to create campaign');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 className="mb-4">Create New Campaign</h2>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {validationErrors.length > 0 && (
        <Alert variant="danger">
          <Alert.Heading>Validation Errors</Alert.Heading>
          <ul className="mb-0">
            {validationErrors.map((err, i) => (
              <li key={i}>{err}</li>
            ))}
          </ul>
        </Alert>
      )}

      {warnings.length > 0 && (
        <Alert variant="warning">
          <Alert.Heading>Warnings</Alert.Heading>
          <ul className="mb-0">
            {warnings.map((warn, i) => (
              <li key={i}>{warn}</li>
            ))}
          </ul>
        </Alert>
      )}

      <Form onSubmit={handleSubmit}>
        {/* Basic Info */}
        <Card className="mb-4">
          <Card.Header>Campaign Information</Card.Header>
          <Card.Body>
            <Form.Group className="mb-3">
              <Form.Label>Campaign Name *</Form.Label>
              <Form.Control
                type="text"
                placeholder="e.g., Chemical Yield Optimization"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </Form.Group>
            <Form.Group className="mb-3">
              <Form.Label>Description</Form.Label>
              <Form.Control
                as="textarea"
                rows={2}
                placeholder="Describe your optimization goals"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </Form.Group>
            <Form.Group>
              <Form.Label>Batch Size</Form.Label>
              <Form.Control
                type="number"
                min={1}
                max={10}
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
              />
              <Form.Text className="text-muted">
                Number of suggestions to generate per iteration
              </Form.Text>
            </Form.Group>
          </Card.Body>
        </Card>

        {/* Parameters */}
        <Card className="mb-4">
          <Card.Header className="d-flex justify-content-between align-items-center">
            <span>Input Parameters</span>
            <Button variant="outline-primary" size="sm" onClick={addParameter}>
              + Add Parameter
            </Button>
          </Card.Header>
          <Card.Body>
            {parameters.map((param, index) => (
              <Row key={index} className="mb-3 align-items-end">
                <Col md={3}>
                  <Form.Group>
                    <Form.Label>Name</Form.Label>
                    <Form.Control
                      type="text"
                      placeholder="e.g., temperature"
                      value={param.name}
                      onChange={(e) => updateParameter(index, 'name', e.target.value)}
                    />
                  </Form.Group>
                </Col>
                <Col md={2}>
                  <Form.Group>
                    <Form.Label>Type</Form.Label>
                    <Form.Select
                      value={param.type}
                      onChange={(e) => updateParameter(index, 'type', e.target.value)}
                    >
                      <option value="continuous">Continuous</option>
                      <option value="discrete">Discrete</option>
                      <option value="categorical">Categorical</option>
                    </Form.Select>
                  </Form.Group>
                </Col>
                <Col md={5}>
                  {param.type === 'continuous' || param.type === 'discrete' ? (
                    <Row>
                      <Col>
                        <Form.Group>
                          <Form.Label>Min</Form.Label>
                          <Form.Control
                            type="number"
                            step={param.type === 'continuous' ? 'any' : '1'}
                            value={param.bounds?.[0] ?? 0}
                            onChange={(e) =>
                              updateParameter(index, 'bounds', [
                                parseFloat(e.target.value),
                                param.bounds?.[1] ?? 1,
                              ])
                            }
                          />
                        </Form.Group>
                      </Col>
                      <Col>
                        <Form.Group>
                          <Form.Label>Max</Form.Label>
                          <Form.Control
                            type="number"
                            step={param.type === 'continuous' ? 'any' : '1'}
                            value={param.bounds?.[1] ?? 1}
                            onChange={(e) =>
                              updateParameter(index, 'bounds', [
                                param.bounds?.[0] ?? 0,
                                parseFloat(e.target.value),
                              ])
                            }
                          />
                        </Form.Group>
                      </Col>
                    </Row>
                  ) : (
                    <Form.Group>
                      <Form.Label>Categories (comma-separated)</Form.Label>
                      <Form.Control
                        type="text"
                        placeholder="e.g., Pt, Pd, Rh"
                        value={param.categories?.join(', ') ?? ''}
                        onChange={(e) =>
                          updateParameter(
                            index,
                            'categories',
                            e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                          )
                        }
                      />
                    </Form.Group>
                  )}
                </Col>
                <Col md={2}>
                  {parameters.length > 1 && (
                    <Button
                      variant="outline-danger"
                      size="sm"
                      onClick={() => removeParameter(index)}
                    >
                      Remove
                    </Button>
                  )}
                </Col>
              </Row>
            ))}
          </Card.Body>
        </Card>

        {/* Objectives */}
        <Card className="mb-4">
          <Card.Header className="d-flex justify-content-between align-items-center">
            <span>Objectives (KPIs to Optimize)</span>
            <Button variant="outline-primary" size="sm" onClick={addObjective}>
              + Add Objective
            </Button>
          </Card.Header>
          <Card.Body>
            {objectives.map((obj, index) => (
              <Row key={index} className="mb-3 align-items-end">
                <Col md={4}>
                  <Form.Group>
                    <Form.Label>Name</Form.Label>
                    <Form.Control
                      type="text"
                      placeholder="e.g., yield"
                      value={obj.name}
                      onChange={(e) => updateObjective(index, 'name', e.target.value)}
                    />
                  </Form.Group>
                </Col>
                <Col md={3}>
                  <Form.Group>
                    <Form.Label>Direction</Form.Label>
                    <Form.Select
                      value={obj.direction}
                      onChange={(e) => updateObjective(index, 'direction', e.target.value)}
                    >
                      <option value="maximize">Maximize</option>
                      <option value="minimize">Minimize</option>
                    </Form.Select>
                  </Form.Group>
                </Col>
                <Col md={3}>
                  <Form.Group>
                    <Form.Label>Unit (optional)</Form.Label>
                    <Form.Control
                      type="text"
                      placeholder="e.g., %"
                      value={obj.unit ?? ''}
                      onChange={(e) => updateObjective(index, 'unit', e.target.value)}
                    />
                  </Form.Group>
                </Col>
                <Col md={2}>
                  {objectives.length > 1 && (
                    <Button
                      variant="outline-danger"
                      size="sm"
                      onClick={() => removeObjective(index)}
                    >
                      Remove
                    </Button>
                  )}
                </Col>
              </Row>
            ))}
          </Card.Body>
        </Card>

        {/* Constraints */}
        <Card className="mb-4">
          <Card.Header className="d-flex justify-content-between align-items-center">
            <span>Constraints (Optional)</span>
            <Button variant="outline-primary" size="sm" onClick={addSumConstraint}>
              + Add Sum Constraint
            </Button>
          </Card.Header>
          <Card.Body>
            {constraints.length === 0 ? (
              <p className="text-muted mb-0">
                No constraints defined. Add a sum constraint for mixture parameters.
              </p>
            ) : (
              constraints.map((constraint, index) => (
                <Row key={index} className="mb-3 align-items-end">
                  <Col md={3}>
                    <Form.Group>
                      <Form.Label>Type</Form.Label>
                      <Form.Select
                        value={constraint.type}
                        onChange={(e) => updateConstraint(index, 'type', e.target.value)}
                      >
                        <option value="sum_equals">Sum Equals</option>
                        <option value="sum_leq">Sum Less Than</option>
                        <option value="sum_geq">Sum Greater Than</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                  <Col md={4}>
                    <Form.Group>
                      <Form.Label>Parameters (comma-separated)</Form.Label>
                      <Form.Control
                        type="text"
                        placeholder="e.g., fraction_A, fraction_B"
                        value={constraint.parameters.join(', ')}
                        onChange={(e) =>
                          updateConstraint(
                            index,
                            'parameters',
                            e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                          )
                        }
                      />
                    </Form.Group>
                  </Col>
                  <Col md={2}>
                    <Form.Group>
                      <Form.Label>Value</Form.Label>
                      <Form.Control
                        type="number"
                        step="any"
                        value={constraint.value}
                        onChange={(e) =>
                          updateConstraint(index, 'value', parseFloat(e.target.value))
                        }
                      />
                    </Form.Group>
                  </Col>
                  <Col md={2}>
                    <Button
                      variant="outline-danger"
                      size="sm"
                      onClick={() => removeConstraint(index)}
                    >
                      Remove
                    </Button>
                  </Col>
                </Row>
              ))
            )}
          </Card.Body>
        </Card>

        <div className="d-flex gap-2">
          <Button variant="outline-secondary" onClick={handleValidate} disabled={loading}>
            Validate
          </Button>
          <Button variant="primary" type="submit" disabled={loading}>
            {loading ? 'Creating...' : 'Create Campaign'}
          </Button>
        </div>
      </Form>
    </div>
  );
}

export default IntakePage;
