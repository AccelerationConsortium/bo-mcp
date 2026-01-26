interface HealthIndicatorProps {
  status: 'healthy' | 'warning' | 'critical';
}

function HealthIndicator({ status }: HealthIndicatorProps) {
  const statusConfig = {
    healthy: {
      color: '#28a745',
      bgColor: '#d4edda',
      text: 'Healthy',
      description: 'Optimization is progressing well',
    },
    warning: {
      color: '#ffc107',
      bgColor: '#fff3cd',
      text: 'Warning',
      description: 'Progress may be slowing down',
    },
    critical: {
      color: '#dc3545',
      bgColor: '#f8d7da',
      text: 'Critical',
      description: 'Optimization may not be effective',
    },
  };

  const config = statusConfig[status];

  return (
    <div
      className="text-center p-3 rounded"
      style={{ backgroundColor: config.bgColor }}
    >
      <div
        className="health-indicator mx-auto mb-2"
        style={{
          backgroundColor: config.color,
          width: '20px',
          height: '20px',
          borderRadius: '50%',
        }}
      />
      <h5 style={{ color: config.color }} className="mb-1">
        {config.text}
      </h5>
      <small className="text-muted">{config.description}</small>
    </div>
  );
}

export default HealthIndicator;
