import { Card } from 'react-bootstrap';

interface HypervolumeChartProps {
  hypervolume: number;
  iteration: number;
}

function HypervolumeChart({ hypervolume, iteration }: HypervolumeChartProps) {
  return (
    <div className="text-center">
      <div className="mb-3">
        <h2 className="text-primary mb-0">{hypervolume.toFixed(2)}</h2>
        <small className="text-muted">Current Hypervolume</small>
      </div>
      <Card bg="light" className="border-0">
        <Card.Body className="py-2">
          <small className="text-muted">
            After <strong>{iteration}</strong> iteration{iteration !== 1 ? 's' : ''}
          </small>
        </Card.Body>
      </Card>
      <p className="text-muted mt-3 small">
        Hypervolume measures the volume of objective space dominated by the Pareto front.
        Higher values indicate better overall optimization progress.
      </p>
    </div>
  );
}

export default HypervolumeChart;
