import { useState, useEffect } from 'react';
import { Modal, Button, Form, Alert } from 'react-bootstrap';
import { setApiKey, getApiKey, clearApiKey } from '../../api/client';

interface ApiKeyModalProps {
  show: boolean;
  onHide: () => void;
  onSave: () => void;
}

function ApiKeyModal({ show, onHide, onSave }: ApiKeyModalProps) {
  const [apiKey, setApiKeyValue] = useState('');
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (show) {
      const existing = getApiKey();
      if (existing) {
        setApiKeyValue(existing);
      }
      setSaved(false);
    }
  }, [show]);

  const handleSave = () => {
    if (apiKey.trim()) {
      setApiKey(apiKey.trim());
      setSaved(true);
      setTimeout(() => {
        onSave();
      }, 500);
    }
  };

  const handleClear = () => {
    clearApiKey();
    setApiKeyValue('');
    setSaved(false);
  };

  return (
    <Modal show={show} onHide={onHide} centered>
      <Modal.Header closeButton>
        <Modal.Title>API Key Configuration</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <p className="text-muted mb-3">
          Enter your API key to authenticate with the BO-MCP backend.
          For development, use: <code>dev-api-key-12345</code>
        </p>
        <Form.Group>
          <Form.Label>API Key</Form.Label>
          <Form.Control
            type="password"
            placeholder="Enter API key"
            value={apiKey}
            onChange={(e) => setApiKeyValue(e.target.value)}
          />
        </Form.Group>
        {saved && (
          <Alert variant="success" className="mt-3 mb-0">
            API key saved successfully!
          </Alert>
        )}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="outline-danger" onClick={handleClear}>
          Clear
        </Button>
        <Button variant="secondary" onClick={onHide}>
          Cancel
        </Button>
        <Button variant="primary" onClick={handleSave}>
          Save
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

export default ApiKeyModal;
