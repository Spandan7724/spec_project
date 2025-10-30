import { useSession } from '../contexts/SessionContext';
import HistoryList from '../components/history/HistoryList';
import { Trash2 } from 'lucide-react';

export default function History() {
  const { analysisHistory, clearAnalysisHistory } = useSession();

  const handleClearAll = () => {
    if (confirm('Are you sure you want to clear all analysis history? This cannot be undone.')) {
      clearAnalysisHistory();
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Analysis History</h1>
          <p className="text-muted-foreground mt-1">
            View and manage your past currency analyses
          </p>
        </div>

        {analysisHistory.length > 0 && (
          <button
            onClick={handleClearAll}
            className="flex items-center gap-2 px-4 py-2 text-sm border border-destructive text-destructive hover:bg-destructive hover:text-destructive-foreground rounded-lg transition-colors"
          >
            <Trash2 size={16} />
            Clear All
          </button>
        )}
      </div>

      <HistoryList history={analysisHistory} />
    </div>
  );
}
