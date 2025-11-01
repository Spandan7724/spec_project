import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  ComposedChart,
  ReferenceLine,
} from 'recharts';

interface PredictionPoint {
  date: string;
  predicted_price: number;
  confidence_lower?: number;
  confidence_upper?: number;
  actual_price?: number;
}

interface PredictionChartProps {
  data: PredictionPoint[];
  currency_pair?: string;
  latest_close?: number | null;
}

export default function PredictionChart({ data, currency_pair, latest_close }: PredictionChartProps) {
  const hasConfidenceBounds = data.some(
    (point) => point.confidence_lower !== undefined && point.confidence_upper !== undefined
  );

  const hasActualPrices = data.some((point) => point.actual_price !== undefined);

  return (
    <div className="w-full">
      <h3 className="text-base md:text-lg font-semibold mb-4">
        Price Predictions {currency_pair && `(${currency_pair})`}
      </h3>
      {data.length === 0 ? (
        <div className="flex items-center justify-center h-48 md:h-64 text-muted-foreground text-sm">
          No prediction data available
        </div>
      ) : (
        <div className="scroll-container h-64 md:h-80 -mx-2 px-2">
          <div className="min-w-[600px] h-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    if (!Number.isNaN(date.getTime())) {
                      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                    }
                    return String(value);
                  }}
                />
                <YAxis
                  label={{ value: 'Price', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  labelFormatter={(value) => {
                    const date = new Date(value);
                    if (!Number.isNaN(date.getTime())) {
                      return date.toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                      });
                    }
                    return String(value);
                  }}
                  formatter={(value: any) => Number(value).toFixed(4)}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />

                {typeof latest_close === 'number' && Number.isFinite(latest_close) && (
                  <ReferenceLine
                    y={latest_close}
                    stroke="#94a3b8"
                    strokeDasharray="4 4"
                    label={{ value: 'Spot', position: 'insideRight', fill: '#64748b', fontSize: 12 }}
                  />
                )}

                {/* Confidence bounds as area */}
                {hasConfidenceBounds && (
                  <Area
                    type="monotone"
                    dataKey="confidence_upper"
                    stroke="none"
                    fill="#8884d8"
                    fillOpacity={0.2}
                    name="Confidence Range"
                  />
                )}
                {hasConfidenceBounds && (
                  <Area
                    type="monotone"
                    dataKey="confidence_lower"
                    stroke="none"
                    fill="#8884d8"
                    fillOpacity={0.2}
                  />
                )}

                {/* Actual prices */}
                {hasActualPrices && (
                  <Line
                    type="monotone"
                    dataKey="actual_price"
                    stroke="#00C49F"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name="Actual Price"
                  />
                )}

                {/* Predicted prices */}
                <Line
                  type="monotone"
                  dataKey="predicted_price"
                  stroke="#0088FE"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Predicted Price"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
