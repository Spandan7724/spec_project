import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Loader2, RefreshCw, Send, Sparkles, Home, MessageSquare, BarChart3, Cpu, History, Moon, Sun, Menu } from 'lucide-react';
import { conversationService } from '../../services/conversation';
import { analysisService, type AnalysisRequest } from '../../services/analysis';
import MessageBubble from './MessageBubble';
import { useSession, type ChatMessage, type ChatSession } from '../../contexts/SessionContext';
import { useTheme } from '../../contexts/ThemeContext';
import type { AnalysisResult } from '../../types/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isAnalyzing?: boolean;
  hasResults?: boolean;
  correlationId?: string;
  resultData?: AnalysisResult;
  progress?: number;
}

const MAX_TEXTAREA_HEIGHT = 220;

const quickPrompts = [
  'What is the outlook for EUR to USD this week?',
  'Should I convert 10,000 GBP to JPY today?',
  'Compare CAD to AUD conversion over the next month.',
];

function formatRelativeTime(date?: Date) {
  if (!date) {
    return 'No messages yet';
  }

  const diff = Date.now() - date.getTime();
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;

  if (diff < minute) return 'just now';
  if (diff < hour) return `${Math.max(1, Math.round(diff / minute))}m ago`;
  if (diff < day) return `${Math.max(1, Math.round(diff / hour))}h ago`;
  return `${Math.max(1, Math.round(diff / day))}d ago`;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState<string>('');
  const [requiresInput, setRequiresInput] = useState(true);
  const [showMenu, setShowMenu] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const loadedSessionIdRef = useRef<string | null>(null);
  const { theme, toggleTheme } = useTheme();

  const {
    chatSessions,
    activeSessionId: persistedActiveSessionId,
    setActiveSessionId,
    getChatSession,
    addChatMessage,
    clearChatSession,
  } = useSession();

  const defaultSessionId = useMemo(() => {
    if (persistedActiveSessionId) {
      return persistedActiveSessionId;
    }

    const sessions = Object.values(chatSessions) as ChatSession[];
    if (sessions.length === 0) {
      return null;
    }

    const latest = sessions.reduce((prev, current) =>
      current.updatedAt > prev.updatedAt ? current : prev
    );

    return latest.sessionId;
  }, [chatSessions, persistedActiveSessionId]);

  const lastMessage = messages.length > 0 ? messages[messages.length - 1] : undefined;
  const isInputDisabled = isLoading || !requiresInput;

  useEffect(() => {
    if (!textareaRef.current) return;
    const textarea = textareaRef.current;
    textarea.style.height = 'auto';
    const nextHeight = Math.min(textarea.scrollHeight, MAX_TEXTAREA_HEIGHT);
    textarea.style.height = `${nextHeight}px`;
  }, [input]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage]);

  useEffect(() => {
    if (!defaultSessionId) {
      return;
    }

    if (loadedSessionIdRef.current === defaultSessionId) {
      return;
    }

    const session = getChatSession(defaultSessionId);
    if (!session) {
      return;
    }

    const hydratedMessages: Message[] = session.messages.map((msg: ChatMessage) => ({
      role: msg.role,
      content: msg.content,
      timestamp: new Date(msg.timestamp),
      isAnalyzing: msg.metadata?.isAnalyzing,
      hasResults: msg.metadata?.hasResults,
      correlationId: msg.metadata?.correlationId,
      resultData: msg.metadata?.resultData,
      progress: msg.metadata?.progress,
    }));

    setSessionId(defaultSessionId);
    setMessages(hydratedMessages);
    loadedSessionIdRef.current = defaultSessionId;

    if (persistedActiveSessionId !== defaultSessionId) {
      setActiveSessionId(defaultSessionId);
    }
  }, [defaultSessionId, getChatSession, persistedActiveSessionId, setActiveSessionId]);

  const handlePromptSelect = useCallback(
    (prompt: string) => {
      if (isInputDisabled) return;
      setInput(prompt);
      requestAnimationFrame(() => {
        textareaRef.current?.focus();
      });
    },
    [isInputDisabled]
  );

  const handleSend = async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setRequiresInput(false);
    setStreamingMessage('');

    try {
      const response = await conversationService.sendMessageStream(
        trimmedInput,
        sessionId || undefined,
        (chunk) => {
          if (chunk.message) {
            setStreamingMessage(chunk.message);
          }
        }
      );

      const resolvedSessionId = response.session_id || sessionId || `session-${Date.now()}`;

      if (!sessionId || sessionId !== resolvedSessionId) {
        setSessionId(resolvedSessionId);
      }

      if (persistedActiveSessionId !== resolvedSessionId) {
        setActiveSessionId(resolvedSessionId);
      }

      loadedSessionIdRef.current = resolvedSessionId;

      addChatMessage(resolvedSessionId, {
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
      setStreamingMessage('');

      addChatMessage(resolvedSessionId, {
        role: 'assistant',
        content: response.message,
        timestamp: assistantMessage.timestamp.getTime(),
      });

      setRequiresInput(response.requires_input);

      if (response.state === 'processing' && response.parameters) {
        try {
          const correlationId = `analysis-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;

          const analysisRequest: AnalysisRequest = {
            session_id: resolvedSessionId,
            correlation_id: correlationId,
            currency_pair: response.parameters.currency_pair ?? undefined,
            base_currency: response.parameters.base_currency ?? undefined,
            quote_currency: response.parameters.quote_currency ?? undefined,
            amount: Number(response.parameters.amount) || 0,
            risk_tolerance: response.parameters.risk_tolerance || 'moderate',
            urgency: response.parameters.urgency || 'normal',
            timeframe: response.parameters.timeframe ?? undefined,
            timeframe_days: response.parameters.timeframe_days,
          };

          await analysisService.startAnalysis(analysisRequest);

          const analyzingMessage: Message = {
            role: 'assistant',
            content: 'Analysis started. Gathering market data...',
            timestamp: new Date(),
            isAnalyzing: true,
            correlationId,
            progress: 0,
          };

          setMessages((prev) => [...prev, analyzingMessage]);

          addChatMessage(resolvedSessionId, {
            role: 'assistant',
            content: analyzingMessage.content,
            timestamp: analyzingMessage.timestamp.getTime(),
            metadata: {
              isAnalyzing: true,
              correlationId,
              progress: 0,
            },
          });

          startAnalysisProgressStream(resolvedSessionId, correlationId);
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
      setStreamingMessage('');
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
        },
      ]);

      if (sessionId) {
        addChatMessage(sessionId, {
          role: 'user',
          content: userMessage.content,
          timestamp: userMessage.timestamp.getTime(),
        });
      }

      setRequiresInput(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = useCallback(async () => {
    if (sessionId) {
      try {
        await conversationService.resetSession(sessionId);
      } catch (error) {
        console.error('Error resetting session:', error);
      }
      clearChatSession(sessionId);
    }

    setMessages([]);
    setSessionId(null);
    setRequiresInput(true);
    setInput('');
    setStreamingMessage('');
    setIsLoading(false);
    loadedSessionIdRef.current = null;
    setActiveSessionId(null);
  }, [sessionId, clearChatSession, setActiveSessionId]);

  const startAnalysisProgressStream = (activeSession: string, correlationId: string) => {
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
          fetchAnalysisResults(activeSession, correlationId);
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

  const fetchAnalysisResults = async (activeSession: string, correlationId: string) => {
    try {
      const session = await conversationService.getSession(activeSession);

      if (!session.has_results) {
        console.warn('Analysis completed but no results in session');
        return;
      }

      const recommendation = (session.recommendation ?? undefined) as AnalysisResult | undefined;

      const resultsMessage: Message = {
        role: 'assistant',
        content: session.result_summary || 'Analysis complete!',
        timestamp: new Date(),
        hasResults: true,
        correlationId,
        resultData: recommendation,
      };

      setMessages((prev) =>
        prev.map((msg) =>
          msg.correlationId === correlationId && msg.isAnalyzing ? resultsMessage : msg
        )
      );

      addChatMessage(activeSession, {
        role: 'assistant',
        content: resultsMessage.content,
        timestamp: resultsMessage.timestamp.getTime(),
        metadata: {
          hasResults: true,
          correlationId,
          resultData: recommendation,
        },
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
    <div className="relative flex h-[calc(100dvh-env(safe-area-inset-bottom)-env(safe-area-inset-top))] flex-col bg-gradient-to-b from-muted/10 via-background to-background">
      <div className="pointer-events-none absolute inset-x-0 top-0 z-0 h-32 bg-gradient-to-b from-primary/10 via-background/40 to-transparent" aria-hidden />

      <header className="relative z-10 border-b border-border/40 bg-background/95 backdrop-blur">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="text-xl font-bold hover:text-primary transition-colors">
              Currency Assistant
            </Link>

            <div className="flex items-center gap-6">
              <div className="hidden md:flex gap-6">
                <Link
                  to="/"
                  className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Home size={20} />
                  <span>Home</span>
                </Link>

                <Link
                  to="/chat"
                  className="flex items-center gap-2 text-primary transition-colors"
                >
                  <MessageSquare size={20} />
                  <span>Chat</span>
                </Link>

                <Link
                  to="/analysis"
                  className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <BarChart3 size={20} />
                  <span>Analysis</span>
                </Link>

                <Link
                  to="/models"
                  className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Cpu size={20} />
                  <span>Models</span>
                </Link>

                <Link
                  to="/history"
                  className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <History size={20} />
                  <span>History</span>
                </Link>
              </div>

              <div className="flex items-center gap-3">
                {sessionId && (
                  <span className="hidden rounded-full border border-border/40 bg-muted/40 px-2.5 py-1 text-[10px] font-medium text-muted-foreground sm:inline-flex">
                    ...{sessionId.slice(-6)}
                  </span>
                )}
                <button
                  onClick={handleReset}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-border/50 px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:border-primary hover:text-primary"
                >
                  <RefreshCw size={14} />
                  <span className="hidden sm:inline">Reset</span>
                </button>
                <button
                  onClick={toggleTheme}
                  className="p-2 rounded-lg hover:bg-accent text-muted-foreground transition-colors"
                  aria-label="Toggle theme"
                >
                  {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
                </button>
                <button
                  onClick={() => setShowMenu(!showMenu)}
                  className="md:hidden p-2 rounded-lg hover:bg-accent text-muted-foreground transition-colors"
                  aria-label="Toggle menu"
                >
                  <Menu size={20} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile menu */}
      {showMenu && (
        <div className="md:hidden relative z-10 border-b border-border/40 bg-background/95 backdrop-blur">
          <nav className="container mx-auto px-4 py-3 flex flex-col gap-2">
            <Link
              to="/"
              onClick={() => setShowMenu(false)}
              className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors py-2"
            >
              <Home size={20} />
              <span>Home</span>
            </Link>
            <Link
              to="/chat"
              onClick={() => setShowMenu(false)}
              className="flex items-center gap-2 text-primary transition-colors py-2"
            >
              <MessageSquare size={20} />
              <span>Chat</span>
            </Link>
            <Link
              to="/analysis"
              onClick={() => setShowMenu(false)}
              className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors py-2"
            >
              <BarChart3 size={20} />
              <span>Analysis</span>
            </Link>
            <Link
              to="/models"
              onClick={() => setShowMenu(false)}
              className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors py-2"
            >
              <Cpu size={20} />
              <span>Models</span>
            </Link>
            <Link
              to="/history"
              onClick={() => setShowMenu(false)}
              className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors py-2"
            >
              <History size={20} />
              <span>History</span>
            </Link>
          </nav>
        </div>
      )}

      <main className="relative z-10 flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto px-4 scrollbar-hide">
          <div className="mx-auto flex h-full max-w-3xl flex-col pb-20 pt-6">
            {messages.length > 0 && (
              <div className="mb-6 flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
                <span className="inline-flex items-center gap-1 rounded-full border border-border/40 bg-muted/30 px-3 py-1 font-medium">
                  <Sparkles size={12} className="text-primary" />
                  {formatRelativeTime(lastMessage?.timestamp)}
                </span>
                <span>{messages.length === 1 ? '1 message' : `${messages.length} messages`}</span>
              </div>
            )}



            {messages.length === 0 && !streamingMessage ? (
              <div className="mt-12 rounded-2xl border border-dashed border-border/40 bg-background/90 p-10 text-center shadow-sm">
                <div className="mx-auto mb-6 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                  <Sparkles size={20} />
                </div>
                <h2 className="text-lg font-semibold">Welcome to Forex</h2>
                <p className="mt-3 text-sm text-muted-foreground">
                  Ask investment-grade questions and I'll bring structured chat experience to currency planning.
                </p>
                <div className="mt-6 space-y-2 text-left text-sm text-muted-foreground/90">
                  {quickPrompts.slice(0, 2).map((prompt) => (
                    <div key={prompt} className="rounded-xl border border-border/30 bg-muted/30 px-4 py-3">
                      {prompt}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex flex-1 flex-col gap-4">
                {messages.map((message, index) => (
                  <MessageBubble key={`${message.timestamp.toISOString()}-${index}`} message={message} />
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
                  <div className="flex items-center gap-2 self-start text-xs text-muted-foreground">
                    <span className="size-2 rounded-full bg-primary animate-pulse" />
                    Generating response...
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="relative z-10 border-t border-border/40 bg-background/95 px-4 py-2 backdrop-blur">
        <div className="mx-auto w-full max-w-3xl">
          {!requiresInput && (
            <div className="mb-2 flex items-center gap-2 rounded-lg border border-border/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
              <Loader2 size={14} className="animate-spin text-primary" />
              <span>Analysis in progress. I'll notify you when the dashboard is ready.</span>
            </div>
          )}
          <div className="relative flex items-center gap-2 rounded-xl border border-border/40 bg-background shadow-sm transition-all focus-within:border-primary/40 focus-within:shadow-md focus-within:shadow-primary/5">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder={requiresInput ? 'Ask about currency conversions...' : 'Analysis in progress...'}
              disabled={isInputDisabled}
              rows={1}
              className="flex-1 resize-none bg-transparent px-3.5 py-2.5 text-sm outline-none placeholder:text-muted-foreground/60 disabled:cursor-not-allowed disabled:opacity-60"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isInputDisabled}
              className="mr-2 inline-flex size-7 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-all hover:bg-primary/90 hover:scale-105 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100"
            >
              <Send size={14} />
            </button>
          </div>
          {requiresInput && (
            <div className="mt-2 flex flex-wrap gap-1.5 justify-center">
              {quickPrompts.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => handlePromptSelect(prompt)}
                  className="rounded-lg border border-border/30 bg-muted/20 px-2 py-0.5 text-[11px] font-medium text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-primary"
                >
                  {prompt}
                </button>
              ))}
            </div>
          )}
        </div>
      </footer>
    </div>
  );
}
