import { Link, useParams } from 'react-router-dom';
import ResultsDashboard from '../components/results/ResultsDashboard';

export default function Results() {
  const { correlationId } = useParams<{ correlationId: string }>();

  if (!correlationId) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 px-6 py-12 text-center">
        <h1 className="text-2xl font-semibold">No analysis selected</h1>
        <p className="max-w-md text-sm text-muted-foreground">
          Choose an analysis session from the chat or history page to view its dashboard.
        </p>
        <div className="flex gap-3">
          <Link
            to="/chat"
            className="inline-flex items-center gap-2 rounded-md border border-border px-4 py-2 text-sm font-medium text-foreground hover:border-primary hover:text-primary"
          >
            Back to Chat
          </Link>
          <Link
            to="/history"
            className="inline-flex items-center gap-2 rounded-md border border-border px-4 py-2 text-sm font-medium text-foreground hover:border-primary hover:text-primary"
          >
            View History
          </Link>
        </div>
      </div>
    );
  }

  return <ResultsDashboard correlationId={correlationId} />;
}
