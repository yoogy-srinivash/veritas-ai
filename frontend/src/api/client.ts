import axios from 'axios';
import type { HealthResponse, DemoPreset, VerificationResponse, ExplanationResponse, MetricsData } from '../types';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

export const getHealth = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
};

export const getPresets = async (): Promise<DemoPreset[]> => {
  const response = await api.get<DemoPreset[]>('/presets');
  return response.data;
};

export const getMetrics = async (): Promise<MetricsData> => {
  const response = await api.get<MetricsData>('/metrics');
  return response.data;
};

export const pollExplanation = async (jobId: string): Promise<ExplanationResponse> => {
  const response = await api.get<ExplanationResponse>(`/explanations/${jobId}`);
  return response.data;
};

export const verifyContent = async (formData: FormData): Promise<VerificationResponse> => {
  const response = await api.post<VerificationResponse>('/verify', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export default api;
