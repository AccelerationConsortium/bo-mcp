import Plot from 'react-plotly.js';
import type { Objective, Result } from '../../types';
import type { Layout } from 'plotly.js';

interface ParetoChartProps {
  paretoFront: Record<string, number>[];
  objectives: Objective[];
  allResults?: Result[];
}

function ParetoChart({ paretoFront, objectives, allResults }: ParetoChartProps) {
  if (objectives.length < 2) {
    return <p className="text-muted text-center">Need at least 2 objectives for visualization</p>;
  }

  if (paretoFront.length === 0 && (!allResults || allResults.length === 0)) {
    return <p className="text-muted text-center">No data to visualize yet</p>;
  }

  // For 2 objectives, use a scatter plot
  if (objectives.length === 2) {
    const obj1 = objectives[0];
    const obj2 = objectives[1];

    // Collect all objective values for axis scaling
    const allObj1Values: number[] = [];
    const allObj2Values: number[] = [];

    // Add Pareto front values
    paretoFront.forEach((p) => {
      if (p[obj1.name] != null) allObj1Values.push(p[obj1.name]);
      if (p[obj2.name] != null) allObj2Values.push(p[obj2.name]);
    });

    // Add all results values
    if (allResults) {
      allResults.forEach((r) => {
        if (r.objective_values[obj1.name] != null) allObj1Values.push(r.objective_values[obj1.name]);
        if (r.objective_values[obj2.name] != null) allObj2Values.push(r.objective_values[obj2.name]);
      });
    }

    // Calculate axis ranges with padding
    const getAxisRange = (values: number[]): [number, number] => {
      if (values.length === 0) return [0, 1];
      const min = Math.min(...values);
      const max = Math.max(...values);
      const padding = (max - min) * 0.15 || 1; // 15% padding, minimum 1
      return [min - padding, max + padding];
    };

    const xRange = getAxisRange(allObj1Values);
    const yRange = getAxisRange(allObj2Values);

    // Build traces
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const traces: any[] = [];

    // Add dominated points (all results that are not in Pareto front)
    if (allResults && allResults.length > 0) {
      const paretoSet = new Set(
        paretoFront.map((p) => `${p[obj1.name]},${p[obj2.name]}`)
      );

      const dominatedPoints = allResults.filter(
        (r) => !paretoSet.has(`${r.objective_values[obj1.name]},${r.objective_values[obj2.name]}`)
      );

      if (dominatedPoints.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'markers',
          name: 'Dominated',
          x: dominatedPoints.map((r) => r.objective_values[obj1.name]),
          y: dominatedPoints.map((r) => r.objective_values[obj2.name]),
          marker: {
            size: 10,
            color: '#adb5bd',
            symbol: 'circle',
            line: {
              color: '#6c757d',
              width: 1,
            },
          },
          text: dominatedPoints.map(
            (r) =>
              `${obj1.name}: ${r.objective_values[obj1.name]?.toFixed(2)}<br>${obj2.name}: ${r.objective_values[obj2.name]?.toFixed(2)}<br>(Dominated)`
          ),
          hoverinfo: 'text',
        });
      }
    }

    // Add Pareto front points
    if (paretoFront.length > 0) {
      traces.push({
        type: 'scatter',
        mode: 'markers+lines',
        name: 'Pareto Front',
        x: paretoFront.map((p) => p[obj1.name]),
        y: paretoFront.map((p) => p[obj2.name]),
        marker: {
          size: 14,
          color: '#0d6efd',
          symbol: 'star',
          line: {
            color: '#fff',
            width: 2,
          },
        },
        line: {
          color: '#0d6efd',
          width: 2,
          dash: 'dot',
        },
        text: paretoFront.map(
          (p) =>
            `${obj1.name}: ${p[obj1.name]?.toFixed(2)}<br>${obj2.name}: ${p[obj2.name]?.toFixed(2)}<br><b>Pareto Optimal</b>`
        ),
        hoverinfo: 'text',
      });
    }

    return (
      <Plot
        data={traces}
        layout={{
          title: { text: 'Pareto Front' },
          xaxis: {
            title: { text: `${obj1.name} (${obj1.direction})${obj1.unit ? ` [${obj1.unit}]` : ''}` },
            range: xRange,
          },
          yaxis: {
            title: { text: `${obj2.name} (${obj2.direction})${obj2.unit ? ` [${obj2.unit}]` : ''}` },
            range: yRange,
          },
          hovermode: 'closest',
          margin: { t: 50, r: 20, b: 60, l: 60 },
          showlegend: true,
          legend: {
            x: 0,
            y: 1,
            bgcolor: 'rgba(255,255,255,0.8)',
          },
        } as Partial<Layout>}
        style={{ width: '100%', height: '400px' }}
        config={{ responsive: true }}
      />
    );
  }

  // For 3+ objectives, use parallel coordinates
  const dimensions = objectives.map((obj) => {
    const values = paretoFront.map((p) => p[obj.name]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const padding = (max - min) * 0.1 || 1;

    return {
      label: `${obj.name}${obj.unit ? ` (${obj.unit})` : ''}`,
      values: values,
      range: [min - padding, max + padding],
    };
  });

  // Use type assertion for parcoords which has different structure
  const parcoordsData = {
    type: 'parcoords' as const,
    line: {
      color: paretoFront.map((_, i) => i),
      colorscale: 'Blues',
    },
    dimensions,
  };

  return (
    <Plot
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      data={[parcoordsData as any]}
      layout={{
        title: { text: 'Pareto Front (Parallel Coordinates)' },
        margin: { t: 50, r: 20, b: 20, l: 20 },
      } as Partial<Layout>}
      style={{ width: '100%', height: '400px' }}
      config={{ responsive: true }}
    />
  );
}

export default ParetoChart;
