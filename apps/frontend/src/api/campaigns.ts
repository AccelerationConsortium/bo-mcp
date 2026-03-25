import apiClient from './client';
import type {
  Campaign,
  CampaignSpec,
  CreateCampaignResponse,
  DiagnosticReport,
  GenerateSuggestionsResponse,
  IntakeRequest,
  Result,
  SubmitResultsRequest,
  SubmitResultsResponse,
  Suggestion,
} from '../types';

// Create a new campaign
export const createCampaign = async (intake: IntakeRequest['intake']): Promise<CreateCampaignResponse> => {
  const response = await apiClient.post<CreateCampaignResponse>('/campaigns', { intake });
  return response.data;
};

// Get all campaigns
export const getCampaigns = async (): Promise<Campaign[]> => {
  const response = await apiClient.get<{ campaigns: Campaign[]; total: number }>('/campaigns');
  return response.data.campaigns;
};

// Get single campaign by ID
export const getCampaign = async (campaignId: string): Promise<Campaign> => {
  const response = await apiClient.get<Campaign>(`/campaigns/${campaignId}`);
  return response.data;
};

// Get campaign spec
export const getCampaignSpec = async (specId: string): Promise<CampaignSpec> => {
  const response = await apiClient.get<CampaignSpec>(`/campaigns/spec/${specId}`);
  return response.data;
};

// Generate suggestions for a campaign
export const generateSuggestions = async (campaignId: string): Promise<GenerateSuggestionsResponse> => {
  const response = await apiClient.post<GenerateSuggestionsResponse>(
    `/suggestions/${campaignId}/generate`
  );
  return response.data;
};

// Get suggestions for a campaign
export const getSuggestions = async (campaignId: string): Promise<Suggestion[]> => {
  const response = await apiClient.get<Suggestion[]>(`/suggestions/${campaignId}`);
  return response.data;
};

// Submit results for a campaign
export const submitResults = async (
  campaignId: string,
  data: SubmitResultsRequest
): Promise<SubmitResultsResponse> => {
  const response = await apiClient.post<SubmitResultsResponse>(
    `/results/${campaignId}`,
    data
  );
  return response.data;
};

// Get results for a campaign
export const getResults = async (campaignId: string): Promise<Result[]> => {
  const response = await apiClient.get<Result[]>(`/results/${campaignId}`);
  return response.data;
};

// Get diagnostics for a campaign
export const getDiagnostics = async (campaignId: string): Promise<DiagnosticReport> => {
  const response = await apiClient.get<DiagnosticReport>(`/diagnostics/${campaignId}`);
  return response.data;
};
