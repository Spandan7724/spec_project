import api from './api';
import type { Model, TrainingJobResponse, TrainingStatus } from '../types/api';

export interface TrainModelRequest {
  currency_pair: string;
  model_type: 'lightgbm' | 'lstm';
  horizons?: number[];
  version?: string;
  history_days?: number;
  gbm_rounds?: number;
  gbm_patience?: number;
  gbm_learning_rate?: number;
  gbm_num_leaves?: number;
  lstm_epochs?: number;
  lstm_hidden_dim?: number;
  lstm_seq_len?: number;
  lstm_lr?: number;
  lstm_interval?: string;
}

export const modelsService = {
  trainModel: async (request: TrainModelRequest): Promise<TrainingJobResponse> => {
    const response = await api.post<TrainingJobResponse>('/api/models/train', request);
    return response.data;
  },

  getTrainingStatus: async (jobId: string): Promise<TrainingStatus> => {
    const response = await api.get<TrainingStatus>(`/api/models/train/status/${jobId}`);
    return response.data;
  },

  listModels: async (params?: { currency_pair?: string; model_type?: string }) => {
    const response = await api.get('/api/models/', { params });
    return response.data;
  },

  getModelDetails: async (modelId: string): Promise<Model> => {
    const response = await api.get<Model>(`/api/models/${modelId}`);
    return response.data;
  },

  deleteModel: async (modelId: string): Promise<void> => {
    await api.delete(`/api/models/${modelId}`);
  },

  getRegistryInfo: async () => {
    const response = await api.get('/api/models/registry/info');
    return response.data;
  },
};
