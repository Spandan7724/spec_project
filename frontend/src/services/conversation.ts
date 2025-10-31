import api from './api';
import type { ConversationMessage } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

export const conversationService = {
  sendMessage: async (userInput: string, sessionId?: string): Promise<ConversationMessage> => {
    const response = await api.post<ConversationMessage>('/api/conversation/message', {
      user_input: userInput,
      session_id: sessionId,
    });
    return response.data;
  },

  /**
   * Send a message with streaming support using Server-Sent Events (SSE).
   * Calls onChunk for each chunk received, and resolves with the final complete message.
   */
  sendMessageStream: async (
    userInput: string,
    sessionId: string | undefined,
    onChunk: (chunk: Partial<ConversationMessage>) => void
  ): Promise<ConversationMessage> => {
    const response = await fetch(`${API_BASE_URL}/api/conversation/message/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_input: userInput,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let lastCompleteMessage: ConversationMessage | null = null;

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        // Decode the chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages (lines ending with \n\n)
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) {
            continue;
          }

          const data = line.slice(6); // Remove 'data: ' prefix

          if (data.trim() === '[DONE]') {
            // Stream complete
            continue;
          }

          try {
            const parsed = JSON.parse(data);

            // Check for errors
            if (parsed.error) {
              throw new Error(parsed.error);
            }

            // Call onChunk callback
            onChunk(parsed);

            // Store last complete message
            if (parsed.is_complete) {
              lastCompleteMessage = parsed as ConversationMessage;
            }
          } catch (e) {
            console.error('Error parsing SSE data:', e, data);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    if (!lastCompleteMessage) {
      throw new Error('No complete message received from stream');
    }

    return lastCompleteMessage;
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
