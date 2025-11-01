import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ComposedChart,
  Line,
  Legend,
} from 'recharts';

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface WaterfallFeatureContribution {
  feature: string;
  value: number;
  contribution: number;
}

interface WaterfallData {
  base_value: number;
  output_value: number;
  features: WaterfallFeatureContribution[];
}

interface SHAPChartProps {
  feature_importance: FeatureImportance[];
  waterfall_plot?: string | null;
  has_waterfall?: boolean;
  currency_pair?: string;
  waterfall_data?: WaterfallData | null;
}

export default function SHAPChart({
  feature_importance,
  waterfall_plot,
  has_waterfall,
  waterfall_data,
  currency_pair,
}: SHAPChartProps) {
  if (!feature_importance || feature_importance.length === 0) {
    return (
      <div className="w-full">
        <h3 className="text-base md:text-lg font-semibold mb-4">
          Feature Importance (SHAP) {currency_pair && `(${currency_pair})`}
        </h3>
        <div className="flex items-center justify-center h-48 md:h-64 text-muted-foreground text-sm">
          SHAP explanations not available for this analysis
        </div>
      </div>
    );
  }

  // Sort and take top 10
  const topFeatures = [...feature_importance]
    .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
    .slice(0, 5);

  // Format feature names for readability
  const formatFeatureName = (name: string) => {
    return name
      .replace(/_/g, ' ')
      .split(' ')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const chartData = topFeatures.map((f) => ({
    feature: formatFeatureName(f.feature),
    importance: f.importance,
    absImportance: Math.abs(f.importance),
  }));

  const waterfallSeries = (() => {
    if (!waterfall_data || !waterfall_data.features || waterfall_data.features.length === 0) {
      return null;
    }

    let cumulative = waterfall_data.base_value;
    const steps = waterfall_data.features.map((feat) => {
      const contribution = feat.contribution || 0;
      const next = cumulative + contribution;
      const start = Math.min(cumulative, next);
      const range = Math.abs(contribution);
      const row = {
        name: formatFeatureName(feat.feature),
        start,
        range,
        contribution,
        cumulative: next,
        value: feat.value,
        direction: contribution >= 0 ? 'positive' : 'negative',
      };
      cumulative = next;
      return row;
    });

    return {
      baseValue: waterfall_data.base_value,
      finalValue: waterfall_data.output_value ?? cumulative,
      steps,
    };
  })();

  return (
    <div className="w-full">
      <h3 className="text-base md:text-lg font-semibold mb-4">
        Feature Importance (SHAP) {currency_pair && `(${currency_pair})`}
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature Importance Bar Chart */}
        <div>
          <h4 className="text-sm font-medium mb-2">Top 5 Features</h4>
          <div className="h-64 md:h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" label={{ value: 'SHAP Value', position: 'bottom', style: { fontSize: 12 } }} tick={{ fontSize: 12 }} />
                <YAxis
                  dataKey="feature"
                  type="category"
                  width={110}
                  tick={{ fontSize: 11 }}
                />
                <Tooltip
                  formatter={(value: any) => Number(value).toFixed(4)}
                  labelStyle={{ fontWeight: 'bold', color: '#000' }}
                  contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', color: '#000' }}
                  itemStyle={{ color: '#000' }}
                />
                <Bar dataKey="importance" name="SHAP Value">
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.importance > 0 ? '#00C49F' : '#FF6B6B'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4 p-3 bg-muted/40 border border-border rounded text-xs">
            <p className="font-semibold mb-1 text-foreground">Interpretation:</p>
            <p className="text-muted-foreground">
              <span className="text-green-600">■</span> Positive values: Features pushing the
              prediction higher
              <br />
              <span className="text-red-600">■</span> Negative values: Features pushing the
              prediction lower
            </p>
          </div>
        </div>

        {/* SHAP Waterfall Plot */}
        <div>
          <h4 className="text-sm font-medium mb-2">SHAP Waterfall</h4>
          {waterfallSeries && waterfallSeries.steps.length > 0 ? (
            <div className="h-64 md:h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={waterfallSeries.steps} margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 11 }}
                    angle={-20}
                    textAnchor="end"
                    height={70}
                    interval={0}
                  />
                  <YAxis domain={['auto', 'auto']} tickFormatter={(v) => Number(v).toFixed(4)} width={70} tick={{ fontSize: 12 }} />
                  <Tooltip
                    formatter={(value: any, name: any, payload) => {
                      if (name === 'range') {
                        return [Number(value).toFixed(4), 'Contribution'];
                      }
                      if (name === 'cumulative') {
                        return [Number(value).toFixed(4), 'Cumulative'];
                      }
                      return [value, name];
                    }}
                    labelFormatter={(_, items) => {
                      if (!items || items.length === 0) return '';
                      const { payload } = items[1] ?? items[0];
                      const contribution = payload?.contribution ?? 0;
                      const featureValue = payload?.value;
                      return `${payload?.name ?? ''} • Feature value: ${featureValue !== undefined ? Number(featureValue).toFixed(4) : '—'} • Contribution: ${contribution >= 0 ? '+' : ''}${contribution.toFixed(4)}`;
                    }}
                    contentStyle={{ backgroundColor: 'rgba(255,255,255,0.95)', color: '#000' }}
                  />
                  <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="start" stackId="stack" fill="transparent" />
                  <Bar dataKey="range" stackId="stack" name="Contribution">
                    {waterfallSeries.steps.map((entry, idx) => (
                      <Cell key={`waterfall-cell-${idx}`} fill={entry.direction === 'positive' ? '#22c55e' : '#ef4444'} />
                    ))}
                  </Bar>
                  <Line type="monotone" dataKey="cumulative" stroke="#0ea5e9" strokeWidth={2} dot={{ r: 3 }} name="Cumulative" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          ) : has_waterfall && waterfall_plot ? (
            <div className="border rounded p-2 bg-white h-64 md:h-80 overflow-y-auto">
              <img
                src={`data:image/png;base64,${waterfall_plot}`}
                alt="SHAP Waterfall Plot"
                className="w-full h-auto"
              />
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 md:h-80 border rounded bg-muted text-muted-foreground">
              <div className="text-center">
                <p className="text-sm">Waterfall plot not available</p>
                <p className="text-xs mt-1">Enable explanations in config to generate SHAP plots</p>
              </div>
            </div>
          )}

          {waterfallSeries && (
            <div className="mt-4 grid grid-cols-1 gap-3 text-xs md:grid-cols-3">
              <div className="p-3 bg-muted/40 border border-border rounded">
                <p className="font-semibold text-foreground">Base prediction</p>
                <p className="text-muted-foreground mt-1">{waterfallSeries.baseValue.toFixed(4)}</p>
              </div>
              <div className="p-3 bg-muted/40 border border-border rounded">
                <p className="font-semibold text-foreground">Model output</p>
                <p className="text-muted-foreground mt-1">{waterfallSeries.finalValue.toFixed(4)}</p>
              </div>
              <div className="p-3 bg-muted/40 border border-border rounded">
                <p className="font-semibold text-foreground">How to read</p>
                <p className="text-muted-foreground mt-1">
                  Green bars push the forecast higher, red bars push it lower. The line shows the running effect of each feature.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Feature Descriptions */}
      <div className="mt-6 p-4 bg-muted/40 border border-border rounded">
        <h4 className="text-sm font-semibold mb-2 text-foreground">What is SHAP?</h4>
        <p className="text-xs text-muted-foreground leading-relaxed">
          SHAP (SHapley Additive exPlanations) values explain individual predictions by showing how
          each feature contributes to the model's output. This helps understand which technical
          indicators, market conditions, or other factors had the biggest impact on the prediction
          for this specific currency pair and timeframe.
        </p>
      </div>
    </div>
  );
}
