import { Link } from 'react-router-dom';
import { useSession } from '../contexts/SessionContext';
import { Tabs } from '../components/ui/tabs';
import { SpinningCoin } from '../components/ui/SpinningCoin';
import SessionCard from '../components/history/SessionCard';
import HistoryCard from '../components/history/HistoryCard';
import {
  MessageSquare,
  TrendingUp,
  FileText,
  Sparkles,
  Brain,
  Shield,
  LineChart,
  Lightbulb,
  Layers,
  Database,
  Cpu,
  Target,
  User,
  ArrowRight,
  CheckCircle2
} from 'lucide-react';

export default function Home() {
  const { chatSessions, analysisHistory } = useSession();

  // Get recent sessions (sorted by updatedAt, most recent first)
  const recentSessions = Object.values(chatSessions)
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, 5);

  // Get recent analyses (already sorted in context, most recent first)
  const recentAnalyses = analysisHistory.slice(0, 5);

  const OverviewTab = (
    <div className="space-y-8">
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

        <Link
          to="/history"
          className="group p-6 border rounded-lg transition-colors hover:bg-accent hover:border-accent/60"
        >
          <h2 className="text-2xl font-semibold mb-2 transition-colors group-hover:text-accent-foreground">
            Analysis History
          </h2>
          <p className="text-muted-foreground transition-colors group-hover:text-accent-foreground/90">
            View all past analyses and their results
          </p>
        </Link>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 border rounded-lg bg-card">
          <div className="flex items-center gap-3">
            <MessageSquare className="text-primary" size={24} />
            <div>
              <p className="text-sm text-muted-foreground">Active Sessions</p>
              <p className="text-2xl font-bold">{Object.keys(chatSessions).length}</p>
            </div>
          </div>
        </div>
        <div className="p-4 border rounded-lg bg-card">
          <div className="flex items-center gap-3">
            <TrendingUp className="text-primary" size={24} />
            <div>
              <p className="text-sm text-muted-foreground">Total Analyses</p>
              <p className="text-2xl font-bold">{analysisHistory.length}</p>
            </div>
          </div>
        </div>
        <div className="p-4 border rounded-lg bg-card">
          <div className="flex items-center gap-3">
            <FileText className="text-primary" size={24} />
            <div>
              <p className="text-sm text-muted-foreground">Completed</p>
              <p className="text-2xl font-bold">
                {analysisHistory.filter((a) => a.status === 'completed').length}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const HistoryTab = (
    <div className="space-y-8">
      {/* Recent Chat Sessions */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold">Recent Conversations</h2>
          <Link to="/chat" className="text-sm text-primary hover:underline">
            Start New Chat →
          </Link>
        </div>
        {recentSessions.length > 0 ? (
          <div className="space-y-3">
            {recentSessions.map((session) => (
              <SessionCard key={session.sessionId} session={session} />
            ))}
          </div>
        ) : (
          <div className="p-8 border rounded-lg text-center text-muted-foreground">
            <MessageSquare size={48} className="mx-auto mb-3 opacity-50" />
            <p>No conversations yet. Start a chat to get personalized recommendations!</p>
          </div>
        )}
      </div>

      {/* Recent Analyses */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold">Recent Analyses</h2>
          <Link to="/history" className="text-sm text-primary hover:underline">
            View All History →
          </Link>
        </div>
        {recentAnalyses.length > 0 ? (
          <div className="space-y-3">
            {recentAnalyses.map((item) => (
              <HistoryCard key={item.correlationId} item={item} />
            ))}
          </div>
        ) : (
          <div className="p-8 border rounded-lg text-center text-muted-foreground">
            <TrendingUp size={48} className="mx-auto mb-3 opacity-50" />
            <p>No analyses yet. Run a quick analysis to see your first results!</p>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="space-y-12">
      {/* Hero Banner */}
      <div className="text-center space-y-6 py-8">
        <div className="space-y-4">
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            Forex Currency Assistant
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground">
            Make smarter currency exchange decisions
          </p>
          <p className="text-base md:text-lg text-muted-foreground max-w-3xl mx-auto">
            AI-powered decision support that tells you exactly when and how to convert your money.
            Stop guessing—get data-driven recommendations backed by real-time market data, news sentiment, and ML price predictions.
          </p>
        </div>
        <div className="flex flex-wrap gap-4 justify-center">
          <Link
            to="/chat"
            className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
          >
            Start Chatting
            <ArrowRight size={18} />
          </Link>
          <Link
            to="/analysis"
            className="inline-flex items-center gap-2 px-6 py-3 border border-border rounded-lg font-medium hover:bg-accent transition-colors"
          >
            Quick Analysis
          </Link>
        </div>
      </div>

      {/* Spinning Coin Section */}
      <div className="flex flex-col items-center justify-center -mt-4">
        <SpinningCoin
          size={400}
          className="w-[350px] h-[350px] sm:w-[400px] sm:h-[400px]"
        />
        <p className="text-sm text-muted-foreground mt-2 italic">
          Real-time currency analysis
        </p>
      </div>

      {/* Key Features */}
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-3xl font-bold mb-2">Key Features</h2>
          <p className="text-muted-foreground">
            Powerful capabilities to help you make informed currency decisions
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Feature 1 */}
          <div className="p-6 border rounded-lg bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Target className="text-primary" size={24} />
              </div>
              <h3 className="text-lg font-semibold">Intelligent Timing</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Get clear actions: convert now, stage over time with specific tranches, or wait.
              Personalized to your risk tolerance, amount, and timeframe.
            </p>
          </div>

          {/* Feature 2 */}
          <div className="p-6 border rounded-lg bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Database className="text-primary" size={24} />
              </div>
              <h3 className="text-lg font-semibold">Multi-Source Intelligence</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Real-time exchange rates, financial news sentiment from trusted sources,
              and economic calendar tracking (GDP, interest rates, policy announcements).
            </p>
          </div>

          {/* Feature 3 */}
          <div className="p-6 border rounded-lg bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Brain className="text-primary" size={24} />
              </div>
              <h3 className="text-lg font-semibold">Dual ML Predictions</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              LightGBM for daily/weekly forecasts and LSTM for intraday predictions.
              Get uncertainty bounds and confidence scores for every prediction.
            </p>
          </div>

          {/* Feature 4 */}
          <div className="p-6 border rounded-lg bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Shield className="text-primary" size={24} />
              </div>
              <h3 className="text-lg font-semibold">Risk-Aware Engine</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Balances profit potential against volatility, costs, and event proximity.
              Three risk profiles: conservative, moderate, and aggressive.
            </p>
          </div>

          {/* Feature 5 */}
          <div className="p-6 border rounded-lg bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Lightbulb className="text-primary" size={24} />
              </div>
              <h3 className="text-lg font-semibold">Explainable AI</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              SHAP explanations show which factors drive predictions.
              Full evidence trail with visualizations, news, events, and technical signals.
            </p>
          </div>

          {/* Feature 6 */}
          <div className="p-6 border rounded-lg bg-card hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Sparkles className="text-primary" size={24} />
              </div>
              <h3 className="text-lg font-semibold">Flexible Access</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Modern web UI with interactive charts, terminal UI for power users,
              and REST API for integration with your own systems.
            </p>
          </div>
        </div>
      </div>

      {/* How It Works */}
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-3xl font-bold mb-2">How It Works</h2>
          <p className="text-muted-foreground">
            Multi-agent architecture powered by LangGraph
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Layer 1 */}
          <div className="relative p-6 border rounded-lg bg-card">
            <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
              1
            </div>
            <div className="flex items-center gap-3 mb-3">
              <Database className="text-primary" size={24} />
              <h3 className="text-lg font-semibold">Data Collection</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-2">
              Parallel agents fetch real-time market data and analyze news sentiment
            </p>
            <div className="text-xs text-muted-foreground space-y-1">
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>Exchange rates & indicators</span>
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>News sentiment analysis</span>
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>Economic event tracking</span>
              </div>
            </div>
          </div>

          {/* Layer 2 */}
          <div className="relative p-6 border rounded-lg bg-card">
            <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
              2
            </div>
            <div className="flex items-center gap-3 mb-3">
              <Brain className="text-primary" size={24} />
              <h3 className="text-lg font-semibold">ML Prediction</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-2">
              Generate price forecasts with confidence intervals
            </p>
            <div className="text-xs text-muted-foreground space-y-1">
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>LightGBM (daily/weekly)</span>
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>LSTM (intraday)</span>
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>Uncertainty quantification</span>
              </div>
            </div>
          </div>

          {/* Layer 3 */}
          <div className="relative p-6 border rounded-lg bg-card">
            <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
              3
            </div>
            <div className="flex items-center gap-3 mb-3">
              <Cpu className="text-primary" size={24} />
              <h3 className="text-lg font-semibold">Decision Engine</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-2">
              Score actions based on multiple factors
            </p>
            <div className="text-xs text-muted-foreground space-y-1">
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>Profit potential</span>
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>Risk assessment</span>
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>Cost optimization</span>
              </div>
            </div>
          </div>

          {/* Layer 4 */}
          <div className="relative p-6 border rounded-lg bg-card">
            <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
              4
            </div>
            <div className="flex items-center gap-3 mb-3">
              <User className="text-primary" size={24} />
              <h3 className="text-lg font-semibold">Supervisor</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-2">
              Natural language understanding and response formatting
            </p>
            <div className="text-xs text-muted-foreground space-y-1">
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>Intent extraction</span>
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>Conversation management</span>
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle2 size={12} />
                <span>Clear recommendations</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Getting Started */}
      <div className="space-y-6 bg-accent/30 rounded-lg p-8">
        <div className="text-center">
          <h2 className="text-3xl font-bold mb-2">Getting Started</h2>
          <p className="text-muted-foreground">
            Start making smarter currency decisions in three simple steps
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
          {/* Step 1 */}
          <div className="text-center space-y-3">
            <div className="w-16 h-16 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-2xl mx-auto">
              1
            </div>
            <h3 className="text-xl font-semibold">Start a Conversation</h3>
            <p className="text-sm text-muted-foreground">
              Just ask in plain English: "I need to convert 5000 USD to JPY this month"
            </p>
          </div>

          {/* Step 2 */}
          <div className="text-center space-y-3">
            <div className="w-16 h-16 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-2xl mx-auto">
              2
            </div>
            <h3 className="text-xl font-semibold">Get Your Recommendation</h3>
            <p className="text-sm text-muted-foreground">
              See whether to convert now, stage over time, or wait—with full reasoning and evidence
            </p>
          </div>

          {/* Step 3 */}
          <div className="text-center space-y-3">
            <div className="w-16 h-16 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-2xl mx-auto">
              3
            </div>
            <h3 className="text-xl font-semibold">Explore the Evidence</h3>
            <p className="text-sm text-muted-foreground">
              View charts, news, events, and technical indicators that support the decision
            </p>
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-border my-8"></div>

      {/* Existing Tabs Section */}
      <div className="space-y-6">
        <div>
          <h2 className="text-3xl font-bold">Your Dashboard</h2>
          <p className="text-muted-foreground mt-2">
            Access your conversations, analyses, and system overview
          </p>
        </div>

        <Tabs
          tabs={[
            { id: 'overview', label: 'Overview', content: OverviewTab },
            { id: 'history', label: 'History', content: HistoryTab },
          ]}
          defaultTab="overview"
        />
      </div>
    </div>
  );
}
