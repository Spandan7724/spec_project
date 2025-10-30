import api from './api';

export const visualizationService = {
  getConfidenceBreakdown: async (correlationId: string) => {
    const response = await api.get(`/api/viz/confidence/${correlationId}`);
    return response.data;
  },

  getRiskBreakdown: async (correlationId: string) => {
    const response = await api.get(`/api/viz/risk-breakdown/${correlationId}`);
    return response.data;
  },

  getCostBreakdown: async (correlationId: string) => {
    const response = await api.get(`/api/viz/cost-breakdown/${correlationId}`);
    return response.data;
  },

  getTimelineData: async (correlationId: string) => {
    const response = await api.get(`/api/viz/timeline-data/${correlationId}`);
    return response.data;
  },

  getPredictionChart: async (correlationId: string) => {
    const response = await api.get(`/api/viz/prediction-chart/${correlationId}`);
    return response.data;
  },

  getEvidence: async (correlationId: string) => {
    const response = await api.get(`/api/viz/evidence/${correlationId}`);
    return response.data;
  },
};
