import { Loader2, TrendingUp, AlertTriangle, ExternalLink } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  // Chat continuity: analysis-related fields
  isAnalyzing?: boolean;
  hasResults?: boolean;
  correlationId?: string;
  resultData?: any;
  progress?: number;
}

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const navigate = useNavigate();
  const isUser = message.role === 'user';

  // Analyzing message with progress
  if (message.isAnalyzing) {
    return (
      <div className="flex gap-3 mb-4">
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
          <Loader2 size={16} className="animate-spin" />
        </div>
        <div className="flex-1">
          <div className="bg-muted/50 rounded-lg p-3">
            <p className="text-sm mb-2">{message.content}</p>
            {message.progress !== undefined && (
              <div className="w-full bg-border/40 rounded-full h-1.5">
                <div
                  className="bg-primary h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${message.progress}%` }}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Results message with inline card
  if (message.hasResults && message.resultData) {
    const result = message.resultData;

    return (
      <div className="flex gap-3 mb-4">
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
          <TrendingUp size={16} className="text-green-600" />
        </div>
        <div className="flex-1">
          <div className="bg-muted/50 rounded-lg p-3">
            {/* Summary text */}
            <div className="text-sm mb-3 space-y-1">
              {message.content.split('\n').map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </div>

            {/* Inline result card */}
            <div className="border border-border/40 rounded-lg p-3 bg-background space-y-2.5">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Recommendation</p>
                  <p className="font-semibold capitalize">
                    {result.action?.replace(/_/g, ' ') || 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Confidence</p>
                  <p className="font-semibold">
                    {result.confidence ? Math.round(result.confidence * 100) : 0}%
                  </p>
                </div>
              </div>

              {result.timeline && (
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Timeline</p>
                  <p className="text-sm">{result.timeline}</p>
                </div>
              )}

              {result.rationale && result.rationale.length > 0 && (
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Key Points</p>
                  <ul className="text-sm space-y-1">
                    {result.rationale.slice(0, 3).map((point: string, i: number) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-primary mt-0.5">•</span>
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {result.warnings && result.warnings.length > 0 && (
                <div className="flex items-start gap-2 p-2 bg-amber-50 dark:bg-amber-950 rounded">
                  <AlertTriangle size={16} className="text-amber-600 flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-amber-600">
                    {result.warnings[0]}
                  </p>
                </div>
              )}

              <button
                onClick={() => navigate(`/results/${message.correlationId}`)}
                className="w-full px-2.5 py-1.5 text-xs border border-border/40 rounded-md hover:bg-muted/50 transition-colors flex items-center justify-center gap-1.5"
              >
                <ExternalLink size={12} />
                View Full Dashboard
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Regular message
  return (
    <div className={`flex gap-3 mb-4 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        isUser ? 'bg-primary' : 'bg-muted'
      }`}>
        <span className="text-xs font-medium">
          {isUser ? 'U' : 'A'}
        </span>
      </div>
      <div className={`flex-1 ${isUser ? 'flex justify-end' : ''}`}>
        <div className={`rounded-lg px-3 py-2 max-w-[80%] ${
          isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'
        }`}>
          <p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>
          <p className="text-xs mt-1.5 opacity-60">
            {message.timestamp.toLocaleTimeString()}
          </p>
        </div>
      </div>
    </div>
  );
}
