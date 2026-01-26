import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { Container, Nav, Navbar } from 'react-bootstrap';
import { useState, useEffect } from 'react';
import IntakePage from './pages/IntakePage';
import CampaignListPage from './pages/CampaignListPage';
import CampaignDetailPage from './pages/CampaignDetailPage';
import ApiKeyModal from './components/common/ApiKeyModal';
import { getApiKey } from './api/client';

function App() {
  const location = useLocation();
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [hasApiKey, setHasApiKey] = useState(false);

  useEffect(() => {
    const key = getApiKey();
    setHasApiKey(!!key);
    if (!key) {
      setShowApiKeyModal(true);
    }
  }, []);

  const handleApiKeySet = () => {
    setHasApiKey(true);
    setShowApiKeyModal(false);
  };

  return (
    <>
      <Navbar bg="dark" variant="dark" expand="lg" className="mb-4">
        <Container>
          <Navbar.Brand as={Link} to="/">
            BO-MCP-UI
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link
                as={Link}
                to="/"
                active={location.pathname === '/'}
              >
                Campaigns
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/new"
                active={location.pathname === '/new'}
              >
                New Campaign
              </Nav.Link>
            </Nav>
            <Nav>
              <Nav.Link onClick={() => setShowApiKeyModal(true)}>
                {hasApiKey ? 'API Key Set' : 'Set API Key'}
              </Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container>
        <Routes>
          <Route path="/" element={<CampaignListPage />} />
          <Route path="/new" element={<IntakePage />} />
          <Route path="/campaign/:id" element={<CampaignDetailPage />} />
        </Routes>
      </Container>

      <ApiKeyModal
        show={showApiKeyModal}
        onHide={() => setShowApiKeyModal(false)}
        onSave={handleApiKeySet}
      />
    </>
  );
}

export default App;
