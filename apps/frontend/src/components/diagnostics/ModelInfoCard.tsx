import { Card, Table } from 'react-bootstrap';

interface ModelInfoCardProps {
  modelInfo?: {
    type: string;
    acquisition_function: string;
    batch_strategy: string;
    kernel: string;
  };
}

function ModelInfoCard({ modelInfo }: ModelInfoCardProps) {
  if (!modelInfo) {
    return null;
  }

  return (
    <Card className="mb-3">
      <Card.Header>
        <strong>Optimization Settings</strong>
      </Card.Header>
      <Card.Body>
        <Table size="sm" className="mb-0">
          <tbody>
            <tr>
              <td><strong>Model Type</strong></td>
              <td>{modelInfo.type}</td>
            </tr>
            <tr>
              <td><strong>Acquisition Function</strong></td>
              <td>{modelInfo.acquisition_function}</td>
            </tr>
            <tr>
              <td><strong>Batch Strategy</strong></td>
              <td>{modelInfo.batch_strategy}</td>
            </tr>
            <tr>
              <td><strong>Kernel</strong></td>
              <td>{modelInfo.kernel}</td>
            </tr>
          </tbody>
        </Table>
      </Card.Body>
    </Card>
  );
}

export default ModelInfoCard;
