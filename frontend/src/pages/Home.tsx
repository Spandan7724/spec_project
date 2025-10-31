import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold">Currency Assistant</h1>
        <p className="text-muted-foreground mt-2">
          AI-powered currency conversion recommendations
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link
          to="/chat"
          className="group p-6 border rounded-lg transition-colors hover:bg-accent hover:border-accent/60"
        >
          <h2 className="text-2xl font-semibold mb-2 transition-colors group-hover:text-accent-foreground">
            Chat Interface
          </h2>
          <p className="text-muted-foreground transition-colors group-hover:text-accent-foreground/90">
            Have a conversation with our AI to get personalized recommendations
          </p>
        </Link>

        <Link
          to="/analysis"
          className="group p-6 border rounded-lg transition-colors hover:bg-accent hover:border-accent/60"
        >
          <h2 className="text-2xl font-semibold mb-2 transition-colors group-hover:text-accent-foreground">
            Quick Analysis
          </h2>
          <p className="text-muted-foreground transition-colors group-hover:text-accent-foreground/90">
            Fill out a simple form for fast currency conversion analysis
          </p>
        </Link>

        <Link
          to="/models"
          className="group p-6 border rounded-lg transition-colors hover:bg-accent hover:border-accent/60"
        >
          <h2 className="text-2xl font-semibold mb-2 transition-colors group-hover:text-accent-foreground">
            Model Training
          </h2>
          <p className="text-muted-foreground transition-colors group-hover:text-accent-foreground/90">
            Train and manage prediction models for currency pairs
          </p>
        </Link>
      </div>
    </div>
  );
}
