import { createContext, useContext, ReactNode } from 'react';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { STORAGE_KEYS } from '../lib/storage';
import type { AnalysisResult } from '../types/api';

// Types for stored data
export interface ChatMessageMetadata {
  isAnalyzing?: boolean;
  hasResults?: boolean;
  correlationId?: string;
  resultData?: AnalysisResult;
  progress?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  metadata?: ChatMessageMetadata;
}

export interface ChatSession {
  sessionId: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

export interface AnalysisHistoryItem {
  correlationId: string;
  currencyPair: string;
  action: string;
  confidence: number;
  status: string;
  createdAt: number;
  result?: AnalysisResult;
}

export interface UserPreferences {
  defaultCurrencyPair?: string;
  defaultRiskLevel?: string;
  defaultUrgency?: string;
  defaultTimeframe?: string;
}

interface SessionContextType {
  // Chat sessions
  chatSessions: Record<string, ChatSession>;
  addChatMessage: (sessionId: string, message: ChatMessage) => void;
  createChatSession: (sessionId: string) => void;
  clearChatSession: (sessionId: string) => void;
  getChatSession: (sessionId: string) => ChatSession | undefined;

  // Active session
  activeSessionId: string | null;
  setActiveSessionId: (sessionId: string | null) => void;

  // Analysis history
  analysisHistory: AnalysisHistoryItem[];
  addAnalysisToHistory: (item: AnalysisHistoryItem) => void;
  updateAnalysisInHistory: (correlationId: string, updates: Partial<AnalysisHistoryItem>) => void;
  clearAnalysisHistory: () => void;

  // User preferences
  preferences: UserPreferences;
  updatePreferences: (updates: Partial<UserPreferences>) => void;

  // Clear all data
  clearAllData: () => void;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [chatSessions, setChatSessions, removeChatSessions] = useLocalStorage<Record<string, ChatSession>>(
    STORAGE_KEYS.CHAT_SESSIONS,
    {}
  );

  const [activeSessionId, setActiveSessionId, removeActiveSession] = useLocalStorage<string | null>(
    STORAGE_KEYS.ACTIVE_SESSION,
    null
  );

  const [analysisHistory, setAnalysisHistory, removeAnalysisHistory] = useLocalStorage<AnalysisHistoryItem[]>(
    STORAGE_KEYS.ANALYSIS_HISTORY,
    []
  );

  const [preferences, setPreferences, removePreferences] = useLocalStorage<UserPreferences>(
    STORAGE_KEYS.USER_PREFERENCES,
    {}
  );

  // Chat session functions
  const createChatSession = (sessionId: string) => {
    const session: ChatSession = {
      sessionId,
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setChatSessions((prev) => ({ ...prev, [sessionId]: session }));
  };

  const addChatMessage = (sessionId: string, message: ChatMessage) => {
    setChatSessions((prev) => {
      const session = prev[sessionId];
      if (!session) {
        // Create session if it doesn't exist
        return {
          ...prev,
          [sessionId]: {
            sessionId,
            messages: [message],
            createdAt: Date.now(),
            updatedAt: Date.now(),
          },
        };
      }

      let nextMessages = session.messages;
      if (message.metadata?.hasResults && message.metadata.correlationId) {
        const correlationId = message.metadata.correlationId;
        nextMessages = nextMessages.filter(
          (msg) =>
            !(
              msg.metadata?.correlationId === correlationId &&
              msg.metadata?.isAnalyzing
            )
        );
      }
      return {
        ...prev,
        [sessionId]: {
          ...session,
          messages: [...nextMessages, message],
          updatedAt: Date.now(),
        },
      };
    });
  };

  const clearChatSession = (sessionId: string) => {
    setChatSessions((prev) => {
      const newSessions = { ...prev };
      delete newSessions[sessionId];
      return newSessions;
    });
    if (activeSessionId === sessionId) {
      setActiveSessionId(null);
    }
  };

  const getChatSession = (sessionId: string): ChatSession | undefined => {
    return chatSessions[sessionId];
  };

  // Analysis history functions
  const addAnalysisToHistory = (item: AnalysisHistoryItem) => {
    setAnalysisHistory((prev) => {
      // Check if item already exists
      const exists = prev.some((h) => h.correlationId === item.correlationId);
      if (exists) {
        return prev.map((h) => (h.correlationId === item.correlationId ? item : h));
      }
      // Add new item at the beginning (most recent first)
      return [item, ...prev].slice(0, 50); // Keep max 50 items
    });
  };

  const updateAnalysisInHistory = (correlationId: string, updates: Partial<AnalysisHistoryItem>) => {
    setAnalysisHistory((prev) =>
      prev.map((item) =>
        item.correlationId === correlationId ? { ...item, ...updates } : item
      )
    );
  };

  const clearAnalysisHistory = () => {
    setAnalysisHistory([]);
  };

  // User preferences functions
  const updatePreferences = (updates: Partial<UserPreferences>) => {
    setPreferences((prev) => ({ ...prev, ...updates }));
  };

  // Clear all data
  const clearAllData = () => {
    removeChatSessions();
    removeActiveSession();
    removeAnalysisHistory();
    removePreferences();
  };

  const value: SessionContextType = {
    chatSessions,
    addChatMessage,
    createChatSession,
    clearChatSession,
    getChatSession,
    activeSessionId,
    setActiveSessionId,
    analysisHistory,
    addAnalysisToHistory,
    updateAnalysisInHistory,
    clearAnalysisHistory,
    preferences,
    updatePreferences,
    clearAllData,
  };

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession() {
  const context = useContext(SessionContext);
  if (context === undefined) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
}
