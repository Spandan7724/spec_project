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

  // New visualization endpoints
  getHistoricalPrices: async (correlationId: string) => {
    const response = await api.get(`/api/viz/historical-prices/${correlationId}`);
    return response.data;
  },

  getTechnicalIndicators: async (correlationId: string) => {
    const response = await api.get(`/api/viz/technical-indicators/${correlationId}`);
    return response.data;
  },

  getSentimentTimeline: async (correlationId: string) => {
    const response = await api.get(`/api/viz/sentiment-timeline/${correlationId}`);
    return response.data;
  },

  getEventsTimeline: async (correlationId: string) => {
    const response = await api.get(`/api/viz/events-timeline/${correlationId}`);
    return response.data;
  },

  getSHAPExplanations: async (correlationId: string) => {
    const response = await api.get(`/api/viz/shap-explanations/${correlationId}`);
    return response.data;
  },

  getPredictionQuantiles: async (correlationId: string) => {
    const response = await api.get(`/api/viz/prediction-quantiles/${correlationId}`);
    return response.data;
  },

  getMarketRegime: async (correlationId: string) => {
    const response = await api.get(`/api/viz/market-regime/${correlationId}`);
    return response.data;
  },
};
