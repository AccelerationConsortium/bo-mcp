import { Card, Badge, Table, Collapse, Alert } from 'react-bootstrap';
import { useState } from 'react';
import type { MethodSelection } from '../../types';

interface MethodSelectionInfoProps {
  selection: MethodSelection;
}

function MethodSelectionInfo({ selection }: MethodSelectionInfoProps) {
  const [expanded, setExpanded] = useState(false);

  const getConfidenceBadge = (level: string) => {
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

  return (
    <Card className="mt-3 mb-3">
      <Card.Header
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: 'pointer' }}
        className="d-flex justify-content-between align-items-center"
      >
        <span>
          <strong>Algorithm Selection</strong>
          <span className="ms-2">{getConfidenceBadge(selection.confidence)}</span>
        </span>
        <i className={`bi bi-chevron-${expanded ? 'up' : 'down'}`} />
      </Card.Header>

      <Collapse in={expanded}>
        <div>
          <Card.Body>
            {/* Explanation */}
            <div className="mb-3">
              <p style={{ whiteSpace: 'pre-line' }}>{selection.explanation}</p>
            </div>

            {/* Method summary table */}
            <Table size="sm" bordered className="mb-3">
              <tbody>
                <tr>
                  <td style={{ width: '30%' }}><strong>Model</strong></td>
                  <td><code>{selection.model_type}</code></td>
                </tr>
                <tr>
                  <td><strong>Acquisition</strong></td>
                  <td><code>{selection.acquisition_function}</code></td>
                </tr>
                <tr>
                  <td><strong>Strategy</strong></td>
                  <td>{selection.optimization_strategy}</td>
                </tr>
                <tr>
                  <td><strong>Transforms</strong></td>
                  <td>{selection.input_transforms.join(', ')}</td>
                </tr>
              </tbody>
            </Table>

            {/* Warnings */}
            {selection.warnings.length > 0 && (
              <Alert variant="warning" className="mb-3">
                <strong>Notes:</strong>
                <ul className="mb-0 mt-1">
                  {selection.warnings.map((warning, i) => (
                    <li key={i}>{warning}</li>
                  ))}
                </ul>
              </Alert>
            )}

            {/* Alternatives */}
            {selection.alternatives.length > 0 && (
              <div>
                <strong>Alternative approaches:</strong>
                <ul className="mt-1 mb-0">
                  {selection.alternatives.map((alt, i) => (
                    <li key={i}>
                      <code>{alt.model || alt.acquisition}</code>: {alt.reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Card.Body>
        </div>
      </Collapse>
    </Card>
  );
}

export default MethodSelectionInfo;
