import { Spinner } from 'react-bootstrap';

interface LoadingSpinnerProps {
  message?: string;
}

function LoadingSpinner({ message = 'Loading...' }: LoadingSpinnerProps) {
  return (
    <div className="text-center py-5">
      <Spinner animation="border" role="status" variant="primary" />
      <p className="mt-3 text-muted">{message}</p>
    </div>
  );
}

export default LoadingSpinner;
