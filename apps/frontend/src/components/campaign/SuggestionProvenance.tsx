import { Card, Badge, Table, Collapse } from 'react-bootstrap';
import { useState } from 'react';
import type { SuggestionProvenance as ProvenanceType } from '../../types';

interface SuggestionProvenanceProps {
  provenance: ProvenanceType;
}

function SuggestionProvenanceDisplay({ provenance }: SuggestionProvenanceProps) {
  const [expanded, setExpanded] = useState(false);

  const getConfidenceBadge = (level?: string) => {
    switch (level) {
      case 'high':
        return <Badge bg="success">High Confidence</Badge>;
      case 'medium':
        return <Badge bg="warning">Medium Confidence</Badge>;
      case 'low':
        return <Badge bg="danger">Low Confidence</Badge>;
      default:
        return <Badge bg="secondary">Unknown</Badge>;
    }
  };

  const getMethodBadge = (method: string) => {
    switch (method) {
      case 'bo':
        return <Badge bg="primary">Bayesian Optimization</Badge>;
      case 'initial_design':
        return <Badge bg="info">Initial Design</Badge>;
      case 'manual':
        return <Badge bg="secondary">Manual</Badge>;
      default:
        return <Badge bg="secondary">{method}</Badge>;
    }
  };

  return (
    <Card className="mt-2 mb-2">
      <Card.Header
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: 'pointer' }}
        className="d-flex justify-content-between align-items-center py-2"
      >
        <span>
          <small className="text-muted me-2">Method:</small>
          {getMethodBadge(provenance.generation_method)}
          <span className="ms-3">
            {getConfidenceBadge(provenance.confidence_level)}
          </span>
        </span>
        <small className="text-muted">
          {expanded ? 'Click to collapse' : 'Click for details'}
        </small>
      </Card.Header>

      <Collapse in={expanded}>
        <div>
          <Card.Body>
            {provenance.explanation && (
              <p className="mb-3">
                <strong>Why this suggestion:</strong><br />
                {provenance.explanation}
              </p>
            )}

            <Table size="sm" className="mb-0">
              <tbody>
                {provenance.acquisition_function && (
                  <tr>
                    <td><strong>Acquisition Function</strong></td>
                    <td>{provenance.acquisition_function}</td>
                  </tr>
                )}
                {provenance.model_type && (
                  <tr>
                    <td><strong>Model Type</strong></td>
                    <td>{provenance.model_type}</td>
                  </tr>
                )}
                {provenance.acquisition_value != null && (
                  <tr>
                    <td><strong>Expected Improvement</strong></td>
                    <td>{provenance.acquisition_value.toFixed(4)}</td>
                  </tr>
                )}
                {provenance.model_uncertainty != null && (
                  <tr>
                    <td><strong>Model Uncertainty</strong></td>
                    <td>{provenance.model_uncertainty.toFixed(4)}</td>
                  </tr>
                )}
                <tr>
                  <td><strong>Iteration</strong></td>
                  <td>{provenance.iteration}</td>
                </tr>
                <tr>
                  <td><strong>Batch Position</strong></td>
                  <td>{provenance.batch_index + 1}</td>
                </tr>
                {provenance.random_seed !== undefined && (
                  <tr>
                    <td><strong>Random Seed</strong></td>
                    <td><code>{provenance.random_seed}</code></td>
                  </tr>
                )}
                {provenance.model_version !== undefined && (
                  <tr>
                    <td><strong>Model Version</strong></td>
                    <td>{provenance.model_version}</td>
                  </tr>
                )}
              </tbody>
            </Table>
          </Card.Body>
        </div>
      </Collapse>
    </Card>
  );
}

export default SuggestionProvenanceDisplay;
