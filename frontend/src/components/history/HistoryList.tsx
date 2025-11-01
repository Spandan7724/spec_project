import { useState } from 'react';
import HistoryCard from './HistoryCard';
import type { AnalysisHistoryItem } from '../../contexts/SessionContext';
import { History as HistoryIcon } from 'lucide-react';

interface HistoryListProps {
  history: AnalysisHistoryItem[];
}

export default function HistoryList({ history }: HistoryListProps) {
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');

  // Filter history
  const filteredHistory = history.filter((item) => {
    const matchesStatus = filterStatus === 'all' || item.status === filterStatus;
    const matchesSearch =
      searchTerm === '' ||
      item.currencyPair.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.action.toLowerCase().includes(searchTerm.toLowerCase());

    return matchesStatus && matchesSearch;
  });

  if (history.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 border rounded-lg">
        <HistoryIcon size={64} className="text-muted-foreground opacity-50 mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Analysis History</h3>
        <p className="text-muted-foreground text-sm">
          Your past analyses will appear here
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4 md:space-y-6">
      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 md:gap-4 flex-wrap">
        <input
          type="text"
          placeholder="Search by currency pair or action..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="flex-1 min-w-[200px] px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary mobile-tap min-h-[44px] text-sm"
        />
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          className="px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary mobile-tap min-h-[44px] text-sm"
        >
          <option value="all">All Status</option>
          <option value="completed">Completed</option>
          <option value="pending">Pending</option>
          <option value="error">Error</option>
        </select>
      </div>

      {/* Results count */}
      <p className="text-xs md:text-sm text-muted-foreground">
        Showing {filteredHistory.length} of {history.length} analyses
      </p>

      {/* History cards */}
      {filteredHistory.length === 0 ? (
        <div className="text-center py-12 border rounded-lg">
          <p className="text-xs md:text-sm text-muted-foreground">No analyses match your filters</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 md:gap-4">
          {filteredHistory.map((item) => (
            <HistoryCard key={item.correlationId} item={item} />
          ))}
        </div>
      )}
    </div>
  );
}
