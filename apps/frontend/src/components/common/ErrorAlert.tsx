import { Alert } from 'react-bootstrap';

interface ErrorAlertProps {
  title?: string;
  message: string;
  errors?: string[];
  onDismiss?: () => void;
}

function ErrorAlert({ title = 'Error', message, errors, onDismiss }: ErrorAlertProps) {
  return (
    <Alert variant="danger" dismissible={!!onDismiss} onClose={onDismiss}>
      <Alert.Heading>{title}</Alert.Heading>
      <p className="mb-0">{message}</p>
      {errors && errors.length > 0 && (
        <ul className="mb-0 mt-2">
          {errors.map((error, index) => (
            <li key={index}>{error}</li>
          ))}
        </ul>
      )}
    </Alert>
  );
}

export default ErrorAlert;
