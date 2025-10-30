import api from './api';
import type { AnalysisStartResponse, AnalysisStatus, AnalysisResult } from '../types/api';

export interface AnalysisRequest {
  session_id: string;
  correlation_id: string;
  currency_pair?: string;
  base_currency?: string;
  quote_currency?: string;
  amount: number;
  risk_tolerance: string;
  urgency: string;
  timeframe?: string;
  timeframe_text?: string;
  timeframe_days?: number;
}

export const analysisService = {
  startAnalysis: async (request: AnalysisRequest): Promise<AnalysisStartResponse> => {
    const response = await api.post<AnalysisStartResponse>('/api/analysis/start', request);
    return response.data;
  },

  getStatus: async (correlationId: string): Promise<AnalysisStatus> => {
    const response = await api.get<AnalysisStatus>(`/api/analysis/status/${correlationId}`);
    return response.data;
  },

  getResult: async (correlationId: string): Promise<AnalysisResult> => {
    const response = await api.get<AnalysisResult>(`/api/analysis/result/${correlationId}`);
    return response.data;
  },

  listAnalyses: async (params?: { status?: string; currency_pair?: string; limit?: number; offset?: number }) => {
    const response = await api.get('/api/analysis/list', { params });
    return response.data;
  },

  deleteAnalysis: async (correlationId: string): Promise<void> => {
    await api.delete(`/api/analysis/${correlationId}`);
  },

  // SSE stream connection for real-time updates
  streamStatus: (correlationId: string, onUpdate: (data: AnalysisStatus) => void, onError?: (error: any) => void) => {
    const eventSource = new EventSource(`${api.defaults.baseURL}/api/analysis/stream/${correlationId}`);

    eventSource.addEventListener('status', (event) => {
      const data = JSON.parse(event.data);
      onUpdate(data);

      if (data.status === 'completed' || data.status === 'error') {
        eventSource.close();
      }
    });

    eventSource.addEventListener('error', (event) => {
      console.error('SSE Error:', event);
      if (onError) onError(event);
      eventSource.close();
    });

    return () => eventSource.close();
  },
};
