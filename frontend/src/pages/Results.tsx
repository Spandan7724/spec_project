import { useParams } from 'react-router-dom';
import ResultsDashboard from '../components/results/ResultsDashboard';

export default function Results() {
  const { correlationId } = useParams<{ correlationId: string }>();

  if (!correlationId) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No correlation ID provided</p>
      </div>
    );
  }

  return <ResultsDashboard correlationId={correlationId} />;
}
