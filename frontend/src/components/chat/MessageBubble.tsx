import { useMemo } from 'react';
import { Loader2, TrendingUp, AlertTriangle, ExternalLink } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { cn } from '../../lib/utils';
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

interface MessageBubbleProps {
  message: Message;
}

const formatTimestamp = (date: Date) =>
  new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  }).format(date);

export default function MessageBubble({ message }: MessageBubbleProps) {
  const navigate = useNavigate();
  const isUser = message.role === 'user';

  const timestampLabel = useMemo(() => formatTimestamp(message.timestamp), [message.timestamp]);

  if (message.isAnalyzing) {
    const clampedProgress = Math.min(100, Math.max(0, message.progress ?? 0));

    return (
      <div className="flex w-full justify-start">
        <div className="flex w-full max-w-3xl flex-col gap-2">
          <div className="rounded-2xl border border-primary/30 bg-primary/5 p-4 shadow-sm">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-primary">
              <Loader2 size={14} className="animate-spin" />
              Analyzing
            </div>
            <p className="mt-3 text-sm text-muted-foreground">{message.content}</p>
            <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-primary/10">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{ width: `${clampedProgress}%` }}
              />
            </div>
          </div>
          <span className="text-xs text-muted-foreground">{timestampLabel}</span>
        </div>
      </div>
    );
  }

  if (message.hasResults && message.resultData) {
    const result = message.resultData;
    const confidence = result.confidence ? Math.round(result.confidence * 100) : undefined;

    return (
      <div className="flex w-full justify-start">
        <div className="flex w-full max-w-3xl flex-col gap-2">
          <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-4 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-semibold uppercase tracking-wide text-emerald-600">
              <span>Currency Insights</span>
              <span className="text-muted-foreground">{timestampLabel}</span>
            </div>

            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <div>
                <p className="text-xs text-muted-foreground">Recommended Action</p>
                <p className="mt-1 text-base font-semibold capitalize">
                  {result.action ? result.action.replace(/_/g, ' ') : 'Not available'}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Confidence</p>
                <p className="mt-1 text-base font-semibold">
                  {typeof confidence === 'number' ? `${confidence}%` : 'Not rated'}
                </p>
              </div>
            </div>

            {result.timeline && (
              <div className="mt-4 rounded-xl border border-emerald-500/30 bg-background/80 p-3 text-sm">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Timeline</p>
                <p className="mt-1 text-muted-foreground">{result.timeline}</p>
              </div>
            )}

            {Array.isArray(result.rationale) && result.rationale.length > 0 && (
              <div className="mt-4 space-y-2">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Key Drivers</p>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  {result.rationale.slice(0, 3).map((point: string, index: number) => (
                    <li key={index} className="flex gap-2">
                      <span className="mt-1.5 block h-1.5 w-1.5 rounded-full bg-emerald-500" />
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {Array.isArray(result.warnings) && result.warnings.length > 0 && (
              <div className="mt-4 flex items-start gap-2 rounded-xl border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-600">
                <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
                <span>{result.warnings[0]}</span>
              </div>
            )}

            <button
              onClick={() => navigate(`/results/${message.correlationId}`)}
              className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-lg border border-border/40 bg-background px-3 py-2 text-xs font-semibold text-foreground transition-colors hover:border-primary/40 hover:text-primary"
            >
              <ExternalLink size={14} />
              View Full Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  const wrapperClasses = cn('flex w-full', isUser ? 'justify-end' : 'justify-start');
  const surfaceClasses = cn(
    'w-full max-w-2xl rounded-2xl border px-4 py-3 text-sm leading-relaxed shadow-sm backdrop-blur-sm',
    isUser
      ? 'border-primary/30 bg-primary/5 text-foreground'
      : 'border-border/40 bg-background/95'
  );
  const timestampClasses = cn(
    'mt-2 text-xs text-muted-foreground',
    isUser ? 'text-right' : 'text-left'
  );

  return (
    <div className={wrapperClasses}>
      <div className="flex max-w-2xl flex-col">
        <div className={surfaceClasses}>
          <p className="whitespace-pre-wrap">
            {message.content}
          </p>
        </div>
        <span className={timestampClasses}>{timestampLabel}</span>
      </div>
    </div>
  );
}
