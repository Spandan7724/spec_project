import { useState, useEffect, useRef } from 'react';
import { RefreshCw, Send } from 'lucide-react';
import { conversationService } from '../../services/conversation';
import { analysisService } from '../../services/analysis';
import { addChatMessage, getChatHistory } from '../../utils/storage';
import MessageBubble from './MessageBubble';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isAnalyzing?: boolean;
  hasResults?: boolean;
  correlationId?: string;
  resultData?: any;
  progress?: number;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState<string>('');
  const [requiresInput, setRequiresInput] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);

  useEffect(() => {
    const history = getChatHistory();
    if (history.length > 0) {
      const lastSession = history[history.length - 1];
      if (lastSession.sessionId) {
        setSessionId(lastSession.sessionId);
        setActiveSessionId(lastSession.sessionId);
        setMessages(
          lastSession.messages.map((m: any) => ({
            ...m,
            timestamp: new Date(m.timestamp),
          }))
        );
      }
    }
  }, []);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setRequiresInput(false);

    const newSessionId = sessionId || `session-${Date.now()}`;
    if (!sessionId) {
      setSessionId(newSessionId);
      setActiveSessionId(newSessionId);
    }

    addChatMessage(newSessionId, {
      role: 'user',
      content: input,
      timestamp: userMessage.timestamp.getTime(),
    });

    try {
      const response = await conversationService.sendMessageStream(
        input,
        sessionId || undefined,
        (chunk) => {
          if (chunk.message) {
            setStreamingMessage(chunk.message);
          }
        }
      );

      if (!sessionId) {
        setSessionId(response.session_id);
        setActiveSessionId(response.session_id);
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.message,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setStreamingMessage('');

      addChatMessage(response.session_id, {
        role: 'assistant',
        content: response.message,
        timestamp: assistantMessage.timestamp.getTime(),
      });

      setRequiresInput(response.requires_input);

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

          const analyzingMessage: Message = {
            role: 'assistant',
            content: 'Analysis started. Gathering market data...',
            timestamp: new Date(),
            isAnalyzing: true,
            correlationId: correlationId,
            progress: 0,
          };

          setMessages((prev) => [...prev, analyzingMessage]);

          addChatMessage(newSessionId, {
            role: 'assistant',
            content: analyzingMessage.content,
            timestamp: analyzingMessage.timestamp.getTime(),
          });

          startAnalysisProgressStream(newSessionId, correlationId);
        } catch (err) {
          console.error('Failed to start analysis from chat:', err);
          const errorMsg: Message = {
            role: 'assistant',
            content: 'Sorry, I encountered an error starting the analysis. Please try again.',
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errorMsg]);
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
        },
      ]);
      setRequiresInput(true);
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

  const startAnalysisProgressStream = (sessionId: string, correlationId: string) => {
    const cleanup = analysisService.streamStatus(
      correlationId,
      (statusUpdate) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.correlationId === correlationId && msg.isAnalyzing
              ? {
                  ...msg,
                  content: statusUpdate.message,
                  progress: statusUpdate.progress,
                }
              : msg
          )
        );

        if (statusUpdate.status === 'completed') {
          fetchAnalysisResults(sessionId, correlationId);
        }
      },
      (error) => {
        console.error('Analysis stream error:', error);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.correlationId === correlationId && msg.isAnalyzing
              ? {
                  ...msg,
                  content: 'Analysis encountered an error. Please try again.',
                  isAnalyzing: false,
                }
              : msg
          )
        );
      }
    );

    return cleanup;
  };

  const fetchAnalysisResults = async (sessionId: string, correlationId: string) => {
    try {
      const session = await conversationService.getSession(sessionId);

      if (!session.has_results) {
        console.warn('Analysis completed but no results in session');
        return;
      }

      const resultsMessage: Message = {
        role: 'assistant',
        content: session.result_summary || 'Analysis complete!',
        timestamp: new Date(),
        hasResults: true,
        correlationId: correlationId,
        resultData: session.recommendation,
      };

      setMessages((prev) =>
        prev.map((msg) =>
          msg.correlationId === correlationId && msg.isAnalyzing ? resultsMessage : msg
        )
      );

      addChatMessage(sessionId, {
        role: 'assistant',
        content: resultsMessage.content,
        timestamp: resultsMessage.timestamp.getTime(),
      });

      setRequiresInput(true);
    } catch (error) {
      console.error('Error fetching analysis results:', error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.correlationId === correlationId && msg.isAnalyzing
            ? {
                ...msg,
                content: 'Analysis completed but failed to load results. Please refresh.',
                isAnalyzing: false,
              }
            : msg
        )
      );
    }
  };

  return (
    <div className="flex flex-col h-[calc(100dvh-env(safe-area-inset-bottom)-env(safe-area-inset-top))]">
      {/* Header */}
      <div className="h-14 shrink-0 border-b border-border/40 px-4 flex items-center justify-between bg-background">
        <h1 className="text-base font-semibold">Currency Assistant</h1>
        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md hover:bg-muted transition-colors"
        >
          <RefreshCw size={12} />
          Reset
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 pt-4">
        <div className="mx-auto max-w-3xl pb-32">
          {messages.length === 0 ? (
            <div className="h-full min-h-[60vh] flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-muted mb-4">
                  <Send size={20} className="text-muted-foreground" />
                </div>
                <h2 className="text-lg font-medium mb-2">Welcome to Currency Assistant</h2>
                <p className="text-sm text-muted-foreground mb-6">
                  I'll help you make informed decisions about currency conversions.
                </p>
                <div className="text-left p-4 bg-muted/30 rounded-lg space-y-2">
                  <p className="text-xs font-medium text-muted-foreground mb-2">Try asking:</p>
                  <p className="text-sm text-muted-foreground">"I want to convert 5000 USD to EUR"</p>
                  <p className="text-sm text-muted-foreground">"Should I exchange GBP to JPY now?"</p>
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <MessageBubble key={index} message={message} />
              ))}
              {streamingMessage && (
                <MessageBubble
                  message={{
                    role: 'assistant',
                    content: streamingMessage,
                    timestamp: new Date(),
                  }}
                />
              )}
              {isLoading && !streamingMessage && (
                <div className="flex items-center gap-2 text-muted-foreground text-sm mb-4">
                  <div className="animate-pulse">●</div>
                  <div className="animate-pulse delay-100">●</div>
                  <div className="animate-pulse delay-200">●</div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </div>

      {/* Input */}
      <div className="shrink-0 border-t border-border/40 bg-background px-4 py-3">
        <div className="mx-auto max-w-3xl">
          <div className="relative rounded-lg border border-border/40 bg-background focus-within:border-primary/40 focus-within:ring-2 focus-within:ring-primary/10 transition-all">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder={requiresInput ? 'Message Currency Assistant...' : 'Analysis in progress...'}
              disabled={isLoading || !requiresInput}
              rows={1}
              className="w-full bg-transparent px-4 pt-3 pb-10 resize-none outline-none text-sm disabled:opacity-50 disabled:cursor-not-allowed placeholder:text-muted-foreground/60"
            />
            <div className="absolute bottom-2 right-2">
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading || !requiresInput}
                className="size-7 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
              >
                <Send size={14} />
              </button>
            </div>
          </div>
          {sessionId && (
            <p className="text-[10px] text-muted-foreground/60 mt-1.5 text-center">
              Session: {sessionId}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
