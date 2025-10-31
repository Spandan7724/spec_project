import { useState, useEffect, useRef } from 'react';
import { analysisService } from '../../services/analysis';
import { visualizationService } from '../../services/visualization';
import type { AnalysisResult, AnalysisStatus } from '../../types/api';
import { Loader2, AlertCircle, CheckCircle2, TrendingUp, Clock, DollarSign, Download, FileText, FileSpreadsheet, LineChart, Newspaper, Cpu, BarChart3, TrendingDown, Brain, Activity } from 'lucide-react';
import { exportAnalysisPDF, exportAnalysisCSV, exportJSON } from '../../lib/export';
import { useSession } from '../../contexts/SessionContext';
import { toast } from 'sonner';
import { ChartSyncProvider } from '../../contexts/ChartSyncContext';
import ConfidenceChart from '../visualizations/ConfidenceChart';
import RiskChart from '../visualizations/RiskChart';
import TimelineChart from '../visualizations/TimelineChart';
import PredictionChart from '../visualizations/PredictionChart';
import HistoricalPriceChart from '../visualizations/HistoricalPriceChart';
import TechnicalIndicatorsChart from '../visualizations/TechnicalIndicatorsChart';
import SentimentChart from '../visualizations/SentimentChart';
import EventsTimelineChart from '../visualizations/EventsTimelineChart';
import SHAPChart from '../visualizations/SHAPChart';
import QuantileFanChart from '../visualizations/QuantileFanChart';
import MarketRegimeChart from '../visualizations/MarketRegimeChart';
import EvidenceViewer from '../evidence/EvidenceViewer';

interface ResultsDashboardProps {
  correlationId: string;
}

export default function ResultsDashboard({ correlationId }: ResultsDashboardProps) {
  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [vizData, setVizData] = useState<any>({
    confidence: { components: [], overall: null },
    risk: [],
    timeline: [],
    prediction: { data: [], latest_close: null },
    evidence: {},
    historicalPrices: null,
    technicalIndicators: null,
    sentimentTimeline: null,
    eventsTimeline: null,
    shapExplanations: null,
    predictionQuantiles: null,
    marketRegime: null,
  });
  const [activeTab, setActiveTab] = useState<'overview' | 'technical' | 'sentiment' | 'predictions' | 'explainability'>('overview');
  const [historicalTimeframe, setHistoricalTimeframe] = useState(90);
  const { addAnalysisToHistory } = useSession();

  const renderChartFallback = (message: string) => (
    <div className="flex h-72 w-full items-center justify-center px-6 text-center text-sm text-muted-foreground">
      {message}
    </div>
  );

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
        visualizationService.getTimelineData(correlationId),
        visualizationService.getPredictionChart(correlationId),
        visualizationService.getEvidence(correlationId),
        visualizationService.getHistoricalPrices(correlationId, historicalTimeframe),
        visualizationService.getTechnicalIndicators(correlationId),
        visualizationService.getSentimentTimeline(correlationId),
        visualizationService.getEventsTimeline(correlationId),
        visualizationService.getSHAPExplanations(correlationId),
        visualizationService.getPredictionQuantiles(correlationId),
        visualizationService.getMarketRegime(correlationId),
      ]).then(([confidence, risk, timeline, prediction, evidence, historicalPrices, technicalIndicators, sentimentTimeline, eventsTimeline, shapExplanations, predictionQuantiles, marketRegime]) => {
        const ev = evidence.status === 'fulfilled' ? evidence.value : {};
        // Normalize events into UI shape expected by EventsTab
        const rawEvents = Array.isArray(ev?.calendar?.upcoming_events)
          ? ev.calendar.upcoming_events
          : [];
        const events = rawEvents.map((e: any) => ({
          title: e.event || e.title || 'Event',
          date: e.when_utc || e.date || undefined,
          category: e.category,
          impact: (e.importance || e.impact || 'medium').toLowerCase(),
          description: e.description || e.note,
          countries: e.currency ? [e.currency] : undefined,
        }));
        const mappedEvidence = {
          news: Array.isArray(ev?.news?.articles) ? ev.news.articles : [],
          events,
          market_data: ev?.market || {},
        };

        const confidenceValue = confidence.status === 'fulfilled' ? confidence.value : null;
        const confidenceComponents = confidenceValue?.components
          ? Object.entries(confidenceValue.components).map(([component, value]) => ({
              component_name: component.replace(/_/g, ' '),
              confidence: typeof value === 'number' ? value : 0,
            }))
          : [];

        const riskValue = risk.status === 'fulfilled' ? risk.value : null;
        const riskData = (() => {
          if (!riskValue || typeof riskValue !== 'object') return [];

          const normalizeLevel = (value: unknown) => {
            if (value === null || value === undefined) return 'unknown';
            const text = String(value).toLowerCase();
            if (['low', 'minimal', 'none'].includes(text)) return 'low';
            if (['medium', 'moderate'].includes(text)) return 'medium';
            if (['high', 'severe', 'elevated', 'extreme'].includes(text)) return 'high';
            return text || 'unknown';
          };

          const levelToLabel = (slug: string) => {
            if (!slug) return 'Unknown';
            return slug.charAt(0).toUpperCase() + slug.slice(1);
          };

          const levelToScore = (slug: string) => {
            switch (slug) {
              case 'low':
                return 30;
              case 'medium':
                return 60;
              case 'high':
                return 90;
              case 'unknown':
                return 45;
              default:
                return 50;
            }
          };

          const formatPct = (value: number) => `${value.toFixed(2)}%`;

          const items: Array<{ category: string; level: string; score: number; description?: string }> = [];
          const pushItem = (category: string, slug: string, description?: string) => {
            const normalized = slug || 'unknown';
            items.push({
              category,
              level: levelToLabel(normalized),
              score: levelToScore(normalized),
              description,
            });
          };

          if (riskValue.risk_level) {
            const slug = normalizeLevel(riskValue.risk_level);
            pushItem('Overall Risk', slug);
          }

          if (riskValue.event_risk) {
            const slug = normalizeLevel(riskValue.event_risk);
            const detail =
              typeof riskValue.event_details === 'string'
                ? riskValue.event_details
                : Array.isArray(riskValue.event_details)
                ? riskValue.event_details.filter(Boolean).join(', ')
                : undefined;
            pushItem('Event Risk', slug, detail);
          }

          if (riskValue.volatility_risk) {
            const slug = normalizeLevel(riskValue.volatility_risk);
            pushItem('Volatility Risk', slug);
          }

          if (typeof riskValue.realized_vol_30d === 'number') {
            const vol = Math.abs(riskValue.realized_vol_30d);
            const slug = vol >= 12 ? 'high' : vol >= 7 ? 'medium' : 'low';
            pushItem('30d Volatility', slug, `Rolling 30d stdev ~${formatPct(vol)}`);
          }

          if (typeof riskValue.var_95 === 'number') {
            const var95 = Math.abs(riskValue.var_95);
            const slug = var95 >= 1.5 ? 'high' : var95 >= 0.75 ? 'medium' : 'low';
            pushItem('Value at Risk (95%)', slug, `Estimated drawdown ${formatPct(var95)}`);
          }

          if (riskValue.liquidity_risk) {
            const slug = normalizeLevel(riskValue.liquidity_risk);
            pushItem('Liquidity Risk', slug);
          }

          return items;
        })();

        const timelineValue = timeline.status === 'fulfilled' ? timeline.value : null;
        const timelineData = (() => {
          const ensureNumber = (value: unknown, fallback: number) => {
            const numeric = typeof value === 'number' ? value : Number(value);
            return Number.isFinite(numeric) ? numeric : fallback;
          };

          const plan = (data?.staged_plan ?? timelineValue?.staged_plan) as any;
          if (plan && Array.isArray(plan.tranches) && plan.tranches.length > 0) {
            const spacing = ensureNumber(plan.spacing_days, 1);
            return plan.tranches.map((tranche: any, index: number) => {
              const start = ensureNumber(tranche.execute_day, index * spacing);
              const nextTranche = plan.tranches[index + 1];
              const nextStart = nextTranche ? ensureNumber(nextTranche.execute_day, start + spacing) : start + spacing;
              const rawDuration = Math.max(0.5, nextStart - start);
              const duration = Math.max(1, Math.round(rawDuration));
              const tasks = [
                tranche.percentage !== undefined ? `Convert ${Number(tranche.percentage).toFixed(1)}% of notional` : null,
                tranche.amount !== undefined
                  ? `~${Number(tranche.amount).toLocaleString(undefined, { maximumFractionDigits: 0 })} ${data?.metadata?.base_currency || ''}`.trim()
                  : null,
                tranche.rationale || undefined,
              ].filter(Boolean) as string[];

              return {
                phase: tranche.rationale || `Tranche ${tranche.tranche_number ?? index + 1}`,
                duration_days: duration,
                start_day: Math.max(0, Math.round(start)),
                tasks,
              };
            });
          }

          const points = Array.isArray(timelineValue?.timeline_points) ? timelineValue.timeline_points : [];
          if (points.length > 0) {
            return points.map((point: any, idx: number) => {
              const start = ensureNumber(point.day, idx);
              const next = points[idx + 1];
              const nextStart = next ? ensureNumber(next.day, start + 1) : start + 1;
              const duration = Math.max(1, Math.round(Math.max(0.5, nextStart - start)));
              const tasks = [] as string[];
              if (point.percentage !== undefined) tasks.push(`Allocation: ${Number(point.percentage).toFixed(1)}%`);
              if (point.amount !== undefined) {
                tasks.push(`Amount: ${Number(point.amount).toLocaleString(undefined, { maximumFractionDigits: 0 })}`);
              }
              if (point.note) tasks.push(point.note);
              return {
                phase: point.note || `Tranche ${point.index ?? idx + 1}`,
                duration_days: duration,
                start_day: Math.max(0, Math.round(start)),
                tasks,
              };
            });
          }

          const action = timelineValue?.action ?? data?.action;
          const timelineNote = timelineValue?.timeline ?? data?.timeline;
          const timeframeDays = ensureNumber(data?.timeframe_days, ensureNumber(data?.metadata?.timeframe_days, 1));

          if (action === 'convert_now') {
            return [
              {
                phase: 'Immediate Conversion',
                duration_days: 1,
                start_day: 0,
                tasks: [timelineNote || 'Execute conversion right away.'],
              },
            ];
          }

          if (action === 'wait' || action === 'monitor_market') {
            return [
              {
                phase: 'Monitor Market',
                duration_days: Math.max(1, timeframeDays),
                start_day: 0,
                tasks: [timelineNote || 'Track rates until target conditions are met.'],
              },
            ];
          }

          if (action === 'staged_conversion') {
            return [
              {
                phase: 'Stage Conversion Plan',
                duration_days: Math.max(1, timeframeDays),
                start_day: 0,
                tasks: [timelineNote || 'Follow the staged execution plan to reduce timing risk.'],
              },
            ];
          }

          return [];
        })();

        const predictionValue = prediction.status === 'fulfilled' ? prediction.value : null;
        const predictionChart = Array.isArray(predictionValue?.chart_data)
          ? predictionValue.chart_data.map((item: any) => ({
              date: item.horizon_label || String(item.horizon),
              predicted_price: item.mean_rate,
              confidence_lower: item.p10,
              confidence_upper: item.p90,
            }))
          : [];

        setVizData({
          confidence: {
            components: confidenceComponents,
            overall: typeof confidenceValue?.overall === 'number' ? confidenceValue.overall : null,
          },
          risk: riskData,
          timeline: timelineData,
          prediction: {
            data: predictionChart,
            latest_close: predictionValue?.latest_close ?? null,
          },
          evidence: mappedEvidence,
          historicalPrices: historicalPrices.status === 'fulfilled' ? historicalPrices.value : null,
          technicalIndicators: technicalIndicators.status === 'fulfilled' ? technicalIndicators.value : null,
          sentimentTimeline: sentimentTimeline.status === 'fulfilled' ? sentimentTimeline.value : null,
          eventsTimeline: eventsTimeline.status === 'fulfilled' ? eventsTimeline.value : null,
          shapExplanations: shapExplanations.status === 'fulfilled' ? shapExplanations.value : null,
          predictionQuantiles: predictionQuantiles.status === 'fulfilled' ? predictionQuantiles.value : null,
          marketRegime: marketRegime.status === 'fulfilled' ? marketRegime.value : null,
        });
      });
    } catch (err) {
      console.error('Error fetching result:', err);
      setError('Failed to load results');
    }
  };

  const handleTimeframeChange = async (days: number) => {
    console.log('Timeframe change requested:', days);
    setHistoricalTimeframe(days);
    try {
      console.log('Fetching historical prices for', days, 'days, correlationId:', correlationId);
      const historicalPrices = await visualizationService.getHistoricalPrices(correlationId, days);
      console.log('Received historical prices:', historicalPrices);
      setVizData((prev: any) => ({
        ...prev,
        historicalPrices: historicalPrices,
      }));
      console.log('Updated vizData with new historical prices');
    } catch (error) {
      console.error('Error fetching historical prices:', error);
      toast.error('Failed to load historical prices for selected timeframe');
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

  // Smooth animated progress + live activity feed
  const [animatedProgress, setAnimatedProgress] = useState(0);
  const prevProgressRef = useRef(0);
  const [activity, setActivity] = useState<{ time: number; message: string }[]>([]);
  const progressHistoryRef = useRef<{ t: number; p: number }[]>([]);
  const [etaLabel, setEtaLabel] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);

  useEffect(() => {
    if (status?.message) {
      setLastUpdate(Date.now());
      setActivity((prev) => {
        const last = prev[prev.length - 1];
        if (!last || last.message !== status.message) {
          const next = [...prev, { time: Date.now(), message: status.message }];
          return next.slice(-8);
        }
        return prev;
      });
    }
  }, [status?.message]);

  useEffect(() => {
    const end = Math.max(0, Math.min(100, status?.progress ?? 0));
    const start = prevProgressRef.current;
    if (end === start) return;
    const duration = 500;
    const startTs = performance.now();
    const ease = (t: number) => 1 - Math.pow(1 - t, 3); // easeOutCubic
    let raf = 0;
    const step = (ts: number) => {
      const t = Math.min(1, (ts - startTs) / duration);
      const val = start + (end - start) * ease(t);
      setAnimatedProgress(val);
      if (t < 1) raf = requestAnimationFrame(step);
      else prevProgressRef.current = end;
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [status?.progress]);

  // Track ETA based on progress velocity
  useEffect(() => {
    const p = Math.max(0, Math.min(100, status?.progress ?? 0));
    const now = Date.now() / 1000;
    const hist = progressHistoryRef.current;
    // Append point
    if (hist.length === 0 || Math.abs(hist[hist.length - 1].p - p) >= 0.5) {
      hist.push({ t: now, p });
      // Keep last ~10 points
      if (hist.length > 10) hist.shift();
    }
    if (hist.length >= 2) {
      const first = hist[0];
      const last = hist[hist.length - 1];
      const dp = last.p - first.p;
      const dt = last.t - first.t;
      if (dp > 0 && dt > 0.5) {
        const rate = dp / dt; // % per second
        const remaining = Math.max(0, 100 - last.p);
        const etaSec = remaining / rate;
        if (isFinite(etaSec)) {
          const m = Math.floor(etaSec / 60);
          const s = Math.max(0, Math.round(etaSec - m * 60));
          setEtaLabel(m > 0 ? `~${m}m ${s}s` : `~${s}s`);
        }
      } else {
        setEtaLabel(null);
      }
    }
  }, [status?.progress]);

  // Loading state
  if (status?.status === 'pending' || status?.status === 'processing') {
    const p = Math.max(0, Math.min(100, animatedProgress));
    const stageIdx = p < 25 ? 0 : p < 50 ? 1 : p < 75 ? 2 : 3;
    const stages = [
      { label: 'Market Data', icon: LineChart },
      { label: 'Intelligence', icon: Newspaper },
      { label: 'Prediction', icon: Cpu },
      { label: 'Decision', icon: CheckCircle2 },
    ];
    return (
      <div className="max-w-4xl mx-auto">
        <div className="border rounded-lg p-8 text-center">
          {/* Radial progress donut */}
          <div className="relative w-40 h-40 mx-auto mb-6">
            <div
              className="absolute inset-0 rounded-full"
              style={{
                backgroundImage: `conic-gradient(hsl(var(--primary)) ${p * 3.6}deg, hsl(var(--muted)) 0deg)`,
              }}
            />
            <div className="absolute inset-2 rounded-full bg-card" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-3xl font-bold">{p}%</div>
            </div>
          </div>

          <h2 className="text-2xl font-bold mb-1">Analyzing...</h2>
          <p className="text-muted-foreground mb-2">{status.message}</p>
          <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground mb-6">
            <span className="flex items-center gap-2">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-primary" />
              </span>
              Live
            </span>
            {etaLabel && <span>ETA {etaLabel}</span>}
            {lastUpdate && (
              <span>Updated {Math.max(0, Math.floor((Date.now() - lastUpdate) / 1000))}s ago</span>
            )}
          </div>

          {/* Pipeline stage tracker with animated connector */}
          <div className="max-w-2xl mx-auto mt-2">
            <div className="grid grid-cols-4 gap-3 mb-3">
              {stages.map((s, i) => {
                const Icon = s.icon;
                const active = i <= stageIdx;
                return (
                  <div key={s.label} className="flex flex-col items-center">
                    <div
                      className={`w-12 h-12 rounded-full flex items-center justify-center border transition-colors ${
                        active ? 'bg-primary text-primary-foreground border-primary' : 'bg-muted text-muted-foreground'
                      }`}
                      title={s.label}
                    >
                      <Icon size={20} />
                    </div>
                    <span className={`mt-2 text-xs ${active ? 'text-foreground' : 'text-muted-foreground'}`}>{s.label}</span>
                  </div>
                );
              })}
            </div>
            <div className="relative h-2 bg-secondary rounded-full overflow-hidden">
              <div
                className="absolute left-0 top-0 h-full bg-primary rounded-full transition-[width] duration-500 ease-out"
                style={{ width: `${p}%` }}
              />
              {/* Moving indicator */}
              <div
                className="absolute -top-1.5 w-5 h-5 rounded-full bg-primary border-2 border-card shadow"
                style={{ left: `calc(${p}% - 10px)` }}
              />
            </div>
          </div>

          {/* Live activity feed */}
          <div className="max-w-2xl mx-auto text-left mt-6">
            <h3 className="text-sm font-semibold mb-2">Live activity</h3>
            <div className="border rounded-lg p-3 bg-accent/30 max-h-40 overflow-y-auto">
              {activity.length === 0 ? (
                <p className="text-sm text-muted-foreground">Waiting for updates…</p>
              ) : (
                <ul className="space-y-1">
                  {activity.map((a, idx) => (
                    <li key={idx} className="text-sm text-muted-foreground">
                      <span className="text-xs opacity-70 mr-2">{new Date(a.time).toLocaleTimeString()}</span>
                      {a.message}
                    </li>
                  ))}
                </ul>
              )}
            </div>
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
                  <span className="text-primary leading-[1.4]">•</span>
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
            {result.cost_estimate.total_bps != null ? (
              <>
                <p className="text-2xl font-bold">
                  {result.cost_estimate.total_bps.toFixed(1)} bps
                </p>
                {result.cost_estimate.spread_bps != null && (
                  <p className="text-sm text-muted-foreground mt-1">
                    Spread: {result.cost_estimate.spread_bps.toFixed(1)} bps
                    {result.cost_estimate.fee_bps > 0 && ` + Fee: ${result.cost_estimate.fee_bps.toFixed(1)} bps`}
                  </p>
                )}
              </>
            ) : (
              <p className="text-2xl font-bold text-muted-foreground">N/A</p>
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
            {result.expected_outcome.expected_rate != null && result.expected_outcome.expected_rate > 0 ? (
              <>
                <p className="text-2xl font-bold">
                  {result.expected_outcome.expected_rate.toFixed(4)}
                </p>
                {result.expected_outcome.expected_improvement_bps != null && result.expected_outcome.expected_improvement_bps !== 0 && (
                  <p className={`text-sm mt-1 ${result.expected_outcome.expected_improvement_bps > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {result.expected_outcome.expected_improvement_bps > 0 ? '+' : ''}{result.expected_outcome.expected_improvement_bps.toFixed(1)} bps {result.expected_outcome.expected_improvement_bps > 0 ? 'improvement' : 'worse'}
                  </p>
                )}
              </>
            ) : (
              <p className="text-2xl font-bold text-muted-foreground">N/A</p>
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

      {/* Tabbed Visualization Section */}
      <ChartSyncProvider>
        <div className="space-y-6">
          <h2 className="text-2xl font-bold">Detailed Analytics</h2>

          {/* Tab Navigation */}
          <div className="border-b">
            <div className="flex gap-1 overflow-x-auto">
              <button
                type="button"
                onClick={() => setActiveTab('overview')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === 'overview'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-gray-300'
                }`}
              >
                <div className="flex items-center gap-2">
                  <BarChart3 size={16} />
                  Overview
                </div>
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('technical')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === 'technical'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-gray-300'
                }`}
              >
                <div className="flex items-center gap-2">
                  <TrendingDown size={16} />
                  Technical Analysis
                </div>
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('sentiment')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === 'sentiment'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-gray-300'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Newspaper size={16} />
                  Sentiment & Events
                </div>
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('predictions')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === 'predictions'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-gray-300'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Activity size={16} />
                  Predictions
                </div>
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('explainability')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === 'explainability'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-gray-300'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Brain size={16} />
                  ML Explainability
                </div>
              </button>
            </div>
          </div>

          {/* Tab Content */}
          <div className="min-h-[400px]">
            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {vizData.confidence?.overall !== null && (
                  <div className="grid grid-cols-1 gap-3 mb-6 md:grid-cols-3">
                    <div className="p-4 border rounded-lg bg-muted/40">
                      <p className="text-xs text-muted-foreground">Overall Confidence</p>
                      <p className="text-2xl font-semibold mt-1">
                        {(vizData.confidence.overall * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="border rounded-lg p-4">
                    {vizData.confidence?.components?.length > 0 ? (
                      <ConfidenceChart data={vizData.confidence.components} />
                    ) : (
                      renderChartFallback('Confidence metrics are not available for this analysis yet.')
                    )}
                  </div>
                  <div className="border rounded-lg p-4">
                    {vizData.risk.length > 0 ? (
                      <RiskChart data={vizData.risk} />
                    ) : (
                      renderChartFallback('Risk breakdown requires completed risk analytics. Start a new analysis to generate this view.')
                    )}
                  </div>
                  <div className="border rounded-lg p-4">
                    {vizData.prediction?.data?.length > 0 ? (
                      <PredictionChart
                        data={vizData.prediction.data}
                        currency_pair={result.metadata?.currency_pair}
                        latest_close={vizData.prediction.latest_close}
                      />
                    ) : (
                      renderChartFallback('Prediction data was not returned. Train forecasting models or re-run the analysis to populate this chart.')
                    )}
                  </div>
                </div>
                <div className="border rounded-lg p-4">
                  {vizData.timeline.length > 0 ? (
                    <TimelineChart data={vizData.timeline} />
                  ) : (
                    renderChartFallback('Timeline data is not available for this recommendation.')
                  )}
                </div>
              </div>
            )}

            {/* Technical Analysis Tab */}
            {activeTab === 'technical' && (
              <div className="space-y-6">
                <div className="border rounded-lg p-4">
                  {vizData.historicalPrices && vizData.historicalPrices.data?.length ? (
                    <HistoricalPriceChart
                      data={vizData.historicalPrices.data || []}
                      indicators={vizData.historicalPrices.indicators}
                      currency_pair={result.metadata?.currency_pair}
                      onTimeframeChange={handleTimeframeChange}
                      currentTimeframe={historicalTimeframe}
                    />
                  ) : (
                    renderChartFallback('Historical price data is unavailable. YFinance access may be disabled or the pair lacks history.')
                  )}
                </div>
                <div className="border rounded-lg p-4">
                  {vizData.technicalIndicators && vizData.technicalIndicators.data?.length ? (
                    <TechnicalIndicatorsChart
                      data={vizData.technicalIndicators.data || []}
                      indicators={vizData.technicalIndicators.indicators}
                      currency_pair={result.metadata?.currency_pair}
                    />
                  ) : (
                    renderChartFallback('Technical indicators were not generated for this analysis.')
                  )}
                </div>
                <div className="border rounded-lg p-4">
                  {vizData.marketRegime ? (
                    <MarketRegimeChart
                      regime_history={vizData.marketRegime.regime_history || []}
                      current_regime={vizData.marketRegime.current_regime}
                      currency_pair={result.metadata?.currency_pair}
                    />
                  ) : (
                    renderChartFallback('Market regime classification requires historical data. Re-run the analysis when data is available.')
                  )}
                </div>
              </div>
            )}

            {/* Sentiment & Events Tab */}
            {activeTab === 'sentiment' && (
              <div className="space-y-6">
                <div className="border rounded-lg p-4">
                  {vizData.sentimentTimeline ? (
                    <SentimentChart
                      current_sentiment={vizData.sentimentTimeline.current_sentiment}
                      timeline={vizData.sentimentTimeline.timeline || []}
                      currency_pair={result.metadata?.currency_pair}
                    />
                  ) : (
                    renderChartFallback('Sentiment analytics were not returned. Enable news intelligence to populate this view.')
                  )}
                </div>
                <div className="border rounded-lg p-4">
                  {vizData.eventsTimeline ? (
                    <EventsTimelineChart
                      events={vizData.eventsTimeline.events || []}
                      currency_pair={result.metadata?.currency_pair}
                    />
                  ) : (
                    renderChartFallback('Economic event data is unavailable for this analysis.')
                  )}
                </div>
              </div>
            )}

            {/* Predictions Tab */}
            {activeTab === 'predictions' && (
              <div className="space-y-6">
                <div className="border rounded-lg p-4">
                  {vizData.predictionQuantiles && vizData.predictionQuantiles.predictions?.length ? (
                    <QuantileFanChart
                      predictions={vizData.predictionQuantiles.predictions || []}
                      latest_close={vizData.predictionQuantiles.latest_close || 0}
                      currency_pair={result.metadata?.currency_pair}
                    />
                  ) : (
                    renderChartFallback('Quantile forecasts are not available. Train a prediction model or re-run the analysis to generate these bands.')
                  )}
                </div>
              </div>
            )}

            {/* ML Explainability Tab */}
            {activeTab === 'explainability' && (
              <div className="space-y-6">
                {vizData.shapExplanations && (
                  <div className="border rounded-lg p-4">
                    <SHAPChart
                      feature_importance={vizData.shapExplanations.feature_importance || []}
                      waterfall_plot={vizData.shapExplanations.waterfall_plot}
                      has_waterfall={vizData.shapExplanations.has_waterfall}
                      waterfall_data={vizData.shapExplanations.waterfall_data}
                      currency_pair={result.metadata?.currency_pair}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </ChartSyncProvider>

      {/* Evidence Viewer */}
      {vizData.evidence && Object.keys(vizData.evidence).length > 0 && (
        <EvidenceViewer data={vizData.evidence} />
      )}
    </div>
  );
}
