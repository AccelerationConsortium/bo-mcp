import { Alert } from 'react-bootstrap';

interface WarningsAlertProps {
  warnings: string[];
  healthStatus?: 'healthy' | 'warning' | 'critical';
}

function WarningsAlert({ warnings, healthStatus }: WarningsAlertProps) {
  if (!warnings || warnings.length === 0) {
    return null;
  }

  const variant = healthStatus === 'critical' ? 'danger' :
                  healthStatus === 'warning' ? 'warning' : 'info';

  return (
    <Alert variant={variant} className="mb-3">
      <Alert.Heading>
        {healthStatus === 'critical' ? 'Critical Issues Detected' :
         healthStatus === 'warning' ? 'Warnings' : 'Information'}
      </Alert.Heading>
      <ul className="mb-0">
        {warnings.map((warning, index) => (
          <li key={index}>{warning}</li>
        ))}
      </ul>
    </Alert>
  );
}

export default WarningsAlert;
