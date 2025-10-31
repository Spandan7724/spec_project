import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface PredictionPoint {
  horizon: number;
  horizon_label: string;
  mean_rate: number;
  p10_rate: number;
  p25_rate: number;
  p50_rate: number;
  p75_rate: number;
  p90_rate: number;
  direction_probability?: number;
  confidence?: number;
}

interface QuantileFanChartProps {
  predictions: PredictionPoint[];
  latest_close: number;
  currency_pair?: string;
  onHoverChange?: (horizon: string | null) => void;
}

export default function QuantileFanChart({
  predictions,
  latest_close,
  currency_pair,
  onHoverChange,
}: QuantileFanChartProps) {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="w-full h-96">
        <h3 className="text-lg font-semibold mb-4">
          Prediction Uncertainty (Fan Chart) {currency_pair && `(${currency_pair})`}
        </h3>
        <div className="flex items-center justify-center h-80 text-muted-foreground">
          No prediction data available
        </div>
      </div>
    );
  }

  // Add current price as first point
  const chartData = [
    {
      horizon: 0,
      horizon_label: 'Now',
      p10_rate: latest_close,
      p25_rate: latest_close,
      p50_rate: latest_close,
      p75_rate: latest_close,
      p90_rate: latest_close,
      mean_rate: latest_close,
    },
    ...predictions,
  ];

  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold mb-4">
        Prediction Uncertainty (Fan Chart) {currency_pair && `(${currency_pair})`}
      </h3>

      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart
          data={chartData}
          onMouseMove={(e) => {
            if (e && e.activeLabel && onHoverChange) {
              onHoverChange(e.activeLabel as string);
            }
          }}
          onMouseLeave={() => onHoverChange && onHoverChange(null)}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="horizon_label"
            label={{ value: 'Forecast Horizon', position: 'insideBottom', offset: -5 }}
          />
          <YAxis
            label={{ value: 'Exchange Rate', angle: -90, position: 'insideLeft' }}
            domain={['auto', 'auto']}
          />
          <Tooltip
            formatter={(value: any, name: any) => {
              const formattedValue = Number(value).toFixed(4);
              const prettyName = name
                .replace('_rate', '')
                .replace('p', 'P')
                .replace('mean', 'Expected');
              return [formattedValue, prettyName];
            }}
            contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)' }}
          />
          <Legend verticalAlign="bottom" height={50} wrapperStyle={{ paddingTop: '20px' }} />

          {/* 80% confidence interval (P10-P90) */}
          <Area
            type="monotone"
            dataKey="p90_rate"
            stroke="none"
            fill="#0088FE"
            fillOpacity={0.1}
            name="80% CI (Upper)"
          />
          <Area
            type="monotone"
            dataKey="p10_rate"
            stroke="none"
            fill="#0088FE"
            fillOpacity={0.1}
            name="80% CI (Lower)"
          />

          {/* 50% confidence interval (P25-P75) */}
          <Area
            type="monotone"
            dataKey="p75_rate"
            stroke="none"
            fill="#0088FE"
            fillOpacity={0.2}
            name="50% CI (Upper)"
          />
          <Area
            type="monotone"
            dataKey="p25_rate"
            stroke="none"
            fill="#0088FE"
            fillOpacity={0.2}
            name="50% CI (Lower)"
          />

          {/* Median (P50) */}
          <Line
            type="monotone"
            dataKey="p50_rate"
            stroke="#FF8042"
            strokeWidth={2}
            dot={{ r: 4 }}
            name="Median (P50)"
          />

          {/* Mean prediction */}
          <Line
            type="monotone"
            dataKey="mean_rate"
            stroke="#0088FE"
            strokeWidth={3}
            dot={{ r: 5 }}
            name="Expected Rate"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Confidence Intervals Legend */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-3 bg-muted/40 border border-border rounded">
          <p className="text-xs font-semibold mb-1 text-foreground">Current Rate</p>
          <p className="text-lg font-bold text-foreground">{latest_close.toFixed(4)}</p>
        </div>

        <div className="p-3 bg-muted/40 border border-border rounded">
          <p className="text-xs font-semibold mb-1 text-foreground">Outer Band (80% of scenarios)</p>
          <p className="text-xs text-muted-foreground">
            Future rates typically land between the upper and lower edges of the wider blue band.
          </p>
        </div>

        <div className="p-3 bg-muted/40 border border-border rounded">
          <p className="text-xs font-semibold mb-1 text-foreground">Inner Band (50% of scenarios)</p>
          <p className="text-xs text-muted-foreground">
            Half of the forecasted paths sit inside the darker band, highlighting the most likely outcomes.
          </p>
        </div>
      </div>

      {/* Prediction Details */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b">
              <th className="text-left p-2">Horizon</th>
              <th className="text-right p-2">P10</th>
              <th className="text-right p-2">P25</th>
              <th className="text-right p-2">Median</th>
              <th className="text-right p-2">P75</th>
              <th className="text-right p-2">P90</th>
              <th className="text-right p-2">Direction</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((pred, idx) => (
              <tr key={idx} className="border-b hover:bg-accent">
                <td className="p-2">{pred.horizon_label}</td>
                <td className="text-right p-2">{pred.p10_rate.toFixed(4)}</td>
                <td className="text-right p-2">{pred.p25_rate.toFixed(4)}</td>
                <td className="text-right p-2 font-medium">{pred.p50_rate.toFixed(4)}</td>
                <td className="text-right p-2">{pred.p75_rate.toFixed(4)}</td>
                <td className="text-right p-2">{pred.p90_rate.toFixed(4)}</td>
                <td className="text-right p-2">
                  {pred.direction_probability
                    ? `${(pred.direction_probability * 100).toFixed(0)}% ↑`
                    : 'N/A'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
