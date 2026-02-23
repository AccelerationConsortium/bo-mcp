import { Card } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import type { FeatureImportance } from '../../types';

interface FeatureImportanceChartProps {
  featureImportance?: FeatureImportance | null;
}

function FeatureImportanceChart({ featureImportance }: FeatureImportanceChartProps) {
  if (!featureImportance?.lengthscale?.aggregate) {
    return null;
  }

  const aggregate = featureImportance.lengthscale.aggregate;

  // Sort by importance (descending)
  const sorted = Object.entries(aggregate).sort(([, a], [, b]) => b - a);
  const params = sorted.map(([name]) => name);
  const values = sorted.map(([, value]) => value);

  return (
    <Card className="mb-3">
      <Card.Header>
        <strong>Parameter Importance</strong>
        <small className="text-muted ms-2">(from GP length scales)</small>
      </Card.Header>
      <Card.Body>
        <Plot
          data={[
            {
              type: 'bar',
              orientation: 'h',
              y: params,
              x: values,
              marker: {
                color: values.map((v) =>
                  v > 0.3 ? '#198754' : v > 0.15 ? '#0d6efd' : '#6c757d'
                ),
              },
              text: values.map((v) => `${(v * 100).toFixed(1)}%`),
              textposition: 'outside',
              hovertemplate: '%{y}: %{x:.1%}<extra></extra>',
            },
          ]}
          layout={{
            margin: { t: 10, r: 60, b: 40, l: 100 },
            xaxis: {
              title: { text: 'Relative Importance' },
              tickformat: '.0%',
              range: [0, Math.max(...values) * 1.2],
            },
            yaxis: {
              autorange: 'reversed',
            },
            height: Math.max(150, params.length * 30 + 60),
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
        <small className="text-muted">
          Higher importance = parameter has stronger effect on objectives
        </small>
      </Card.Body>
    </Card>
  );
}

export default FeatureImportanceChart;
