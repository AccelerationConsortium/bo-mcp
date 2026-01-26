import axios from 'axios';

const API_BASE_URL = '/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add API key to all requests
apiClient.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('apiKey');
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey;
  }
  return config;
});

// Handle errors globally
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      console.error('Unauthorized - check API key');
    }
    return Promise.reject(error);
  }
);

export default apiClient;

// Helper to set API key
export const setApiKey = (key: string) => {
  localStorage.setItem('apiKey', key);
};

// Helper to get current API key
export const getApiKey = (): string | null => {
  return localStorage.getItem('apiKey');
};

// Helper to clear API key
export const clearApiKey = () => {
  localStorage.removeItem('apiKey');
};
