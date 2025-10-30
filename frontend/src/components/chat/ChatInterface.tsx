import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { conversationService } from '../../services/conversation';
import { analysisService } from '../../services/analysis';
import MessageBubble from './MessageBubble';
import { Send, RefreshCw } from 'lucide-react';
import { useSession, type ChatMessage } from '../../contexts/SessionContext';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export default function ChatInterface() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [requiresInput, setRequiresInput] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { activeSessionId, setActiveSessionId, getChatSession, addChatMessage } = useSession();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Load persisted messages on mount
  useEffect(() => {
    if (activeSessionId) {
      const session = getChatSession(activeSessionId);
      if (session) {
        const loadedMessages = session.messages.map((msg) => ({
          role: msg.role,
          content: msg.content,
          timestamp: new Date(msg.timestamp),
        }));
        setMessages(loadedMessages);
        setSessionId(activeSessionId);
      }
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await conversationService.sendMessage(input.trim(), sessionId || undefined);

      const newSessionId = response.session_id;
      if (!sessionId) {
        setSessionId(newSessionId);
        setActiveSessionId(newSessionId);
      }

      // Persist user message
      addChatMessage(newSessionId, {
        role: 'user',
        content: userMessage.content,
        timestamp: userMessage.timestamp.getTime(),
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.message,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setRequiresInput(response.requires_input);

      // Persist assistant message
      addChatMessage(newSessionId, {
        role: 'assistant',
        content: assistantMessage.content,
        timestamp: assistantMessage.timestamp.getTime(),
      });

      // If the user confirmed, kick off analysis and navigate to results
      if (response.state === 'processing' && response.parameters) {
        try {
          const correlationId = `analysis-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
          const request = {
            session_id: newSessionId,
            correlation_id: correlationId,
            currency_pair: response.parameters.currency_pair,
            base_currency: response.parameters.base_currency,
            quote_currency: response.parameters.quote_currency,
            amount: Number(response.parameters.amount) || 0,
            risk_tolerance: response.parameters.risk_tolerance || 'moderate',
            urgency: response.parameters.urgency || 'normal',
            timeframe: response.parameters.timeframe,
            timeframe_days: response.parameters.timeframe_days,
          } as const;

          await analysisService.startAnalysis(request as any);
          navigate(`/results/${correlationId}`);
        } catch (err) {
          console.error('Failed to start analysis from chat:', err);
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);

      // Persist error message
      if (sessionId) {
        addChatMessage(sessionId, {
          role: 'assistant',
          content: errorMessage.content,
          timestamp: errorMessage.timestamp.getTime(),
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    if (sessionId) {
      try {
        await conversationService.resetSession(sessionId);
      } catch (error) {
        console.error('Error resetting session:', error);
      }
    }
    setMessages([]);
    setSessionId(null);
    setActiveSessionId(null);
    setRequiresInput(true);
    setInput('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-200px)] max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-3xl font-bold">Chat Interface</h1>
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-accent transition-colors"
          title="Reset conversation"
        >
          <RefreshCw size={16} />
          Reset
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto border rounded-lg p-4 mb-4 bg-card">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <p className="text-lg mb-2">Welcome to Currency Assistant!</p>
              <p className="text-sm">Tell me about your currency conversion needs.</p>
              <p className="text-sm mt-4">For example:</p>
              <p className="text-sm italic">"I want to convert 5000 USD to EUR"</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message, index) => (
              <MessageBubble key={index} message={message} />
            ))}
            {isLoading && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="animate-pulse">●</div>
                <div className="animate-pulse delay-100">●</div>
                <div className="animate-pulse delay-200">●</div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={requiresInput ? "Type your message..." : "Analysis in progress..."}
          disabled={isLoading || !requiresInput}
          className="flex-1 px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || isLoading || !requiresInput}
          className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          <Send size={20} />
          Send
        </button>
      </div>

      {sessionId && (
        <p className="text-xs text-muted-foreground mt-2">
          Session ID: {sessionId}
        </p>
      )}
    </div>
  );
}
