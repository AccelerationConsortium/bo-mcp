import { Badge } from 'react-bootstrap';

interface ProgressIndicatorProps {
  status?: 'improving' | 'stagnant' | 'regressing';
}

function ProgressIndicator({ status }: ProgressIndicatorProps) {
  if (!status) {
    return <Badge bg="secondary">Unknown</Badge>;
  }

  const config = {
    improving: {
      bg: 'success',
      icon: '\u2191', // Up arrow
      label: 'Improving',
    },
    stagnant: {
      bg: 'warning',
      icon: '\u2192', // Right arrow
      label: 'Stagnant',
    },
    regressing: {
      bg: 'danger',
      icon: '\u2193', // Down arrow
      label: 'Regressing',
    },
  };

  const { bg, icon, label } = config[status];

  return (
    <div className="text-center">
      <div style={{ fontSize: '1.5rem' }}>{icon}</div>
      <Badge bg={bg}>{label}</Badge>
      <div className="text-muted small mt-1">Progress</div>
    </div>
  );
}

export default ProgressIndicator;
