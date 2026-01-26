import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Card, Row, Col, Badge, Button } from 'react-bootstrap';
import { getCampaigns } from '../api/campaigns';
import type { Campaign } from '../types';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';

function CampaignListPage() {
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadCampaigns();
  }, []);

  const loadCampaigns = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getCampaigns();
      setCampaigns(data);
    } catch (err) {
      setError('Failed to load campaigns. Please check your API key and try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, string> = {
      created: 'secondary',
      active: 'primary',
      paused: 'warning',
      completed: 'success',
      failed: 'danger',
    };
    return <Badge bg={variants[status] || 'secondary'}>{status}</Badge>;
  };

  if (loading) {
    return <LoadingSpinner message="Loading campaigns..." />;
  }

  if (error) {
    return (
      <ErrorAlert
        title="Failed to Load Campaigns"
        message={error}
        onDismiss={() => setError(null)}
      />
    );
  }

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2>Optimization Campaigns</h2>
        <Link to="/new">
          <Button variant="primary">New Campaign</Button>
        </Link>
      </div>

      {campaigns.length === 0 ? (
        <Card className="text-center py-5">
          <Card.Body>
            <h5 className="text-muted">No campaigns yet</h5>
            <p className="text-muted mb-3">
              Create your first optimization campaign to get started.
            </p>
            <Link to="/new">
              <Button variant="primary">Create Campaign</Button>
            </Link>
          </Card.Body>
        </Card>
      ) : (
        <Row xs={1} md={2} lg={3} className="g-4">
          {campaigns.map((campaign) => (
            <Col key={campaign.id}>
              <Link to={`/campaign/${campaign.id}`} className="text-decoration-none">
                <Card className="h-100">
                  <Card.Body>
                    <div className="d-flex justify-content-between align-items-start mb-2">
                      <Card.Title className="text-dark">
                        {campaign.name || 'Unnamed Campaign'}
                      </Card.Title>
                      {getStatusBadge(campaign.status)}
                    </div>
                    <Card.Text className="text-muted small">
                      {campaign.description || 'No description'}
                    </Card.Text>
                    <div className="d-flex justify-content-between text-muted small">
                      <span>Iteration: {campaign.iteration}</span>
                      <span>
                        Created: {new Date(campaign.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </Card.Body>
                </Card>
              </Link>
            </Col>
          ))}
        </Row>
      )}
    </div>
  );
}

export default CampaignListPage;
