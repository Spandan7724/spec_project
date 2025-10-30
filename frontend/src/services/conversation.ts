import api from './api';
import type { ConversationMessage } from '../types/api';

export const conversationService = {
  sendMessage: async (userInput: string, sessionId?: string): Promise<ConversationMessage> => {
    const response = await api.post<ConversationMessage>('/api/conversation/message', {
      user_input: userInput,
      session_id: sessionId,
    });
    return response.data;
  },

  resetSession: async (sessionId: string): Promise<void> => {
    await api.post(`/api/conversation/reset/${sessionId}`);
  },

  getSession: async (sessionId: string): Promise<any> => {
    const response = await api.get(`/api/conversation/session/${sessionId}`);
    return response.data;
  },

  getActiveSessions: async (): Promise<any> => {
    const response = await api.get('/api/conversation/sessions/active');
    return response.data;
  },
};
