import { Link } from 'react-router-dom';
import type { AnalysisHistoryItem } from '../../contexts/SessionContext';
import { ArrowRight, Clock, TrendingUp, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

interface HistoryCardProps {
  item: AnalysisHistoryItem;
}

const StatusBadge = ({ status }: { status: string }) => {
  const config = {
    completed: { color: 'bg-green-100 text-green-700', icon: CheckCircle2, label: 'Completed' },
    pending: { color: 'bg-yellow-100 text-yellow-700', icon: Loader2, label: 'Pending' },
    error: { color: 'bg-red-100 text-red-700', icon: AlertCircle, label: 'Error' },
    processing: { color: 'bg-blue-100 text-blue-700', icon: Loader2, label: 'Processing' },
  };

  const { color, icon: Icon, label } = config[status as keyof typeof config] || config.pending;

  return (
    <span className={`px-2 py-1 text-xs rounded flex items-center gap-1 ${color}`}>
      <Icon size={12} className={status === 'pending' || status === 'processing' ? 'animate-spin' : ''} />
      {label}
    </span>
  );
};

export default function HistoryCard({ item }: HistoryCardProps) {
  const formattedDate = new Date(item.createdAt).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <Link
      to={`/results/${item.correlationId}`}
      className="block border rounded-lg p-4 hover:bg-accent/50 transition-colors group"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <h3 className="font-semibold text-lg">{item.currencyPair}</h3>
            <StatusBadge status={item.status} />
            <span className="px-2 py-1 text-xs rounded bg-primary/10 text-primary">
              {item.action}
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm mb-3">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Clock size={14} />
              <span>{formattedDate}</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingUp size={14} />
              <span>
                Confidence: <span className="font-medium">{(item.confidence * 100).toFixed(1)}%</span>
              </span>
            </div>
            <div className="text-xs text-muted-foreground font-mono">
              {item.correlationId.slice(0, 20)}...
            </div>
          </div>

          {item.result && item.result.rationale && item.result.rationale.length > 0 && (
            <p className="text-sm text-muted-foreground">
              {item.result.rationale[0].substring(0, 150)}
              {item.result.rationale[0].length > 150 && '...'}
            </p>
          )}
        </div>

        <div className="flex items-center text-primary group-hover:translate-x-1 transition-transform">
          <ArrowRight size={20} />
        </div>
      </div>
    </Link>
  );
}
