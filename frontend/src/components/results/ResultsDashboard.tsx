import { useState, useEffect } from 'react';
import { analysisService } from '../../services/analysis';
import { visualizationService } from '../../services/visualization';
import type { AnalysisResult, AnalysisStatus } from '../../types/api';
import { Loader2, AlertCircle, CheckCircle2, TrendingUp, Clock, DollarSign, Download, FileText, FileSpreadsheet } from 'lucide-react';
import { exportAnalysisPDF, exportAnalysisCSV, exportJSON } from '../../lib/export';
import { useSession } from '../../contexts/SessionContext';
import { toast } from 'sonner';
import ConfidenceChart from '../visualizations/ConfidenceChart';
import RiskChart from '../visualizations/RiskChart';
import CostChart from '../visualizations/CostChart';
import TimelineChart from '../visualizations/TimelineChart';
import PredictionChart from '../visualizations/PredictionChart';
import EvidenceViewer from '../evidence/EvidenceViewer';

interface ResultsDashboardProps {
  correlationId: string;
}

export default function ResultsDashboard({ correlationId }: ResultsDashboardProps) {
  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [vizData, setVizData] = useState<any>({
    confidence: [],
    risk: [],
    cost: [],
    timeline: [],
    prediction: [],
    evidence: {},
  });
  const { addAnalysisToHistory, updateAnalysisInHistory } = useSession();

  useEffect(() => {
    // Setup SSE stream for real-time updates
    const cleanup = analysisService.streamStatus(
      correlationId,
      (statusUpdate) => {
        setStatus(statusUpdate);

        // When completed, fetch full results
        if (statusUpdate.status === 'completed') {
          fetchResult();
        }
      },
      (error) => {
        console.error('Stream error:', error);
        setError('Connection error. Please refresh the page.');
      }
    );

    return cleanup;
  }, [correlationId]);

  const fetchResult = async () => {
    try {
      const data = await analysisService.getResult(correlationId);
      setResult(data);

      // Add to history
      addAnalysisToHistory({
        correlationId,
        currencyPair: data.metadata?.currency_pair || 'Unknown',
        action: data.action,
        confidence: data.confidence,
        status: 'completed',
        createdAt: Date.now(),
        result: data,
      });

      // Fetch visualization data in parallel
      Promise.allSettled([
        visualizationService.getConfidenceBreakdown(correlationId),
        visualizationService.getRiskBreakdown(correlationId),
        visualizationService.getCostBreakdown(correlationId),
        visualizationService.getTimelineData(correlationId),
        visualizationService.getPredictionChart(correlationId),
        visualizationService.getEvidence(correlationId),
      ]).then(([confidence, risk, cost, timeline, prediction, evidence]) => {
        const ev = evidence.status === 'fulfilled' ? evidence.value : {};
        const mappedEvidence = {
          news: Array.isArray(ev?.news?.articles) ? ev.news.articles : [],
          events: Array.isArray(ev?.calendar?.upcoming_events) ? ev.calendar.upcoming_events : [],
          market_data: ev?.market || {},
        };

        setVizData({
          confidence: confidence.status === 'fulfilled' ? confidence.value : [],
          risk: risk.status === 'fulfilled' ? risk.value : [],
          cost: cost.status === 'fulfilled' ? cost.value : [],
          timeline: timeline.status === 'fulfilled' ? timeline.value : [],
          prediction: prediction.status === 'fulfilled' ? prediction.value : [],
          evidence: mappedEvidence,
        });
      });
    } catch (err) {
      console.error('Error fetching result:', err);
      setError('Failed to load results');
    }
  };

  const handleExportPDF = async () => {
    if (!result) return;
    try {
      await exportAnalysisPDF(result);
      toast.success('PDF exported successfully');
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Failed to export PDF');
    }
  };

  const handleExportCSV = async () => {
    if (!result) return;
    try {
      await exportAnalysisCSV(result);
      toast.success('CSV exported successfully');
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Failed to export CSV');
    }
  };

  const handleExportJSON = () => {
    if (!result) return;
    try {
      exportJSON(result, `analysis-${correlationId}.json`);
      toast.success('JSON exported successfully');
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Failed to export JSON');
    }
  };

  // Loading state
  if (status?.status === 'pending' || status?.status === 'processing') {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="border rounded-lg p-8 text-center">
          <Loader2 className="animate-spin mx-auto mb-4" size={48} />
          <h2 className="text-2xl font-bold mb-2">Analyzing...</h2>
          <p className="text-muted-foreground mb-4">{status.message}</p>

          <div className="max-w-md mx-auto">
            <div className="w-full bg-secondary rounded-full h-2 mb-2">
              <div
                className="bg-primary h-2 rounded-full transition-all duration-300"
                style={{ width: `${status.progress}%` }}
              />
            </div>
            <p className="text-sm text-muted-foreground">{status.progress}% complete</p>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (status?.status === 'error' || error) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="border border-destructive rounded-lg p-8 text-center">
          <AlertCircle className="mx-auto mb-4 text-destructive" size={48} />
          <h2 className="text-2xl font-bold mb-2">Analysis Failed</h2>
          <p className="text-muted-foreground">{status?.message || error}</p>
        </div>
      </div>
    );
  }

  // Results state
  if (!result) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="border rounded-lg p-8 text-center">
          <Loader2 className="animate-spin mx-auto mb-4" size={48} />
          <p className="text-muted-foreground">Loading results...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold">Analysis Results</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Correlation ID: {correlationId}
          </p>
        </div>

        {/* Export buttons */}
        <div className="flex gap-2">
          <button
            onClick={handleExportPDF}
            className="flex items-center gap-2 px-3 py-2 text-sm border rounded-lg hover:bg-accent transition-colors"
            title="Export as PDF"
          >
            <FileText size={16} />
            PDF
          </button>
          <button
            onClick={handleExportCSV}
            className="flex items-center gap-2 px-3 py-2 text-sm border rounded-lg hover:bg-accent transition-colors"
            title="Export as CSV"
          >
            <FileSpreadsheet size={16} />
            CSV
          </button>
          <button
            onClick={handleExportJSON}
            className="flex items-center gap-2 px-3 py-2 text-sm border rounded-lg hover:bg-accent transition-colors"
            title="Export as JSON"
          >
            <Download size={16} />
            JSON
          </button>
        </div>
      </div>

      {/* Main Recommendation Card */}
      <div className="border rounded-lg p-6 bg-card">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="text-green-500" size={32} />
            <div>
              <h2 className="text-2xl font-bold capitalize">{result.action?.replace(/_/g, ' ')}</h2>
              <p className="text-muted-foreground">Recommended Action</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold">{Math.round(result.confidence * 100)}%</p>
            <p className="text-sm text-muted-foreground">Confidence</p>
          </div>
        </div>

        {result.timeline && (
          <div className="flex items-center gap-2 text-muted-foreground mb-4">
            <Clock size={16} />
            <p>{result.timeline}</p>
          </div>
        )}

        {result.rationale && result.rationale.length > 0 && (
          <div className="mt-4 pt-4 border-t">
            <h3 className="font-semibold mb-2">Rationale:</h3>
            <ul className="space-y-1">
              {result.rationale.map((reason, index) => (
                <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                  <span className="text-primary mt-1">•</span>
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {result.warnings && result.warnings.length > 0 && (
          <div className="mt-4 pt-4 border-t">
            <h3 className="font-semibold mb-2 text-amber-600 flex items-center gap-2">
              <AlertCircle size={16} />
              Warnings:
            </h3>
            <ul className="space-y-1">
              {result.warnings.map((warning, index) => (
                <li key={index} className="text-sm text-amber-600 flex items-start gap-2">
                  <span className="mt-1">•</span>
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Risk Summary */}
        {result.risk_summary && (
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Risk Level</h3>
            <p className="text-2xl font-bold capitalize">{result.risk_summary.risk_level}</p>
            {result.risk_summary.event_risk && (
              <p className="text-sm text-muted-foreground mt-1">
                Event Risk: {result.risk_summary.event_risk}
              </p>
            )}
          </div>
        )}

        {/* Cost Estimate */}
        {result.cost_estimate && (
          <div className="border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign size={16} />
              <h3 className="font-semibold">Estimated Cost</h3>
            </div>
            <p className="text-2xl font-bold">
              {result.cost_estimate.total_cost_bps?.toFixed(1)} bps
            </p>
            {result.cost_estimate.total_cost_absolute && (
              <p className="text-sm text-muted-foreground mt-1">
                ≈ ${result.cost_estimate.total_cost_absolute.toFixed(2)}
              </p>
            )}
          </div>
        )}

        {/* Expected Outcome */}
        {result.expected_outcome && (
          <div className="border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp size={16} />
              <h3 className="font-semibold">Expected Rate</h3>
            </div>
            <p className="text-2xl font-bold">
              {result.expected_outcome.expected_rate?.toFixed(4)}
            </p>
            {result.expected_outcome.expected_improvement_bps && (
              <p className="text-sm text-green-600 mt-1">
                +{result.expected_outcome.expected_improvement_bps.toFixed(1)} bps improvement
              </p>
            )}
          </div>
        )}
      </div>

      {/* Staged Plan (if applicable) */}
      {result.staged_plan && (
        <div className="border rounded-lg p-6">
          <h3 className="text-xl font-bold mb-4">Staged Conversion Plan</h3>
          <p className="text-muted-foreground mb-4">
            Execute in {result.staged_plan.num_tranches} tranches to minimize risk
          </p>
          {result.staged_plan.tranches && (
            <div className="space-y-2">
              {result.staged_plan.tranches.map((tranche: any, index: number) => (
                <div key={index} className="flex items-center justify-between p-3 bg-muted rounded">
                  <div>
                    <p className="font-medium">Tranche {index + 1}</p>
                    <p className="text-sm text-muted-foreground">{tranche.note}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-bold">{tranche.percentage?.toFixed(1)}%</p>
                    <p className="text-sm text-muted-foreground">Day {tranche.execute_on_day}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Component Confidences */}
      {result.component_confidences && Object.keys(result.component_confidences).length > 0 && (
        <div className="border rounded-lg p-6">
          <h3 className="text-xl font-bold mb-4">Confidence Breakdown</h3>
          <div className="space-y-3">
            {Object.entries(result.component_confidences).map(([component, confidence]) => (
              <div key={component}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm capitalize">{component.replace(/_/g, ' ')}</span>
                  <span className="text-sm font-medium">{Math.round((confidence as number) * 100)}%</span>
                </div>
                <div className="w-full bg-secondary rounded-full h-2">
                  <div
                    className="bg-primary h-2 rounded-full"
                    style={{ width: `${(confidence as number) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Visualization Charts */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold">Detailed Analytics</h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Confidence Chart */}
          {vizData.confidence.length > 0 && (
            <div className="border rounded-lg p-4">
              <ConfidenceChart data={vizData.confidence} />
            </div>
          )}

          {/* Risk Chart */}
          {vizData.risk.length > 0 && (
            <div className="border rounded-lg p-4">
              <RiskChart data={vizData.risk} />
            </div>
          )}

          {/* Cost Chart */}
          {vizData.cost.length > 0 && (
            <div className="border rounded-lg p-4">
              <CostChart data={vizData.cost} />
            </div>
          )}

          {/* Prediction Chart */}
          {vizData.prediction.length > 0 && (
            <div className="border rounded-lg p-4">
              <PredictionChart
                data={vizData.prediction}
                currency_pair={result.metadata?.currency_pair}
              />
            </div>
          )}
        </div>

        {/* Timeline Chart (full width) */}
        {vizData.timeline.length > 0 && (
          <div className="border rounded-lg p-4">
            <TimelineChart data={vizData.timeline} />
          </div>
        )}
      </div>

      {/* Evidence Viewer */}
      {vizData.evidence && Object.keys(vizData.evidence).length > 0 && (
        <EvidenceViewer data={vizData.evidence} />
      )}
    </div>
  );
}
