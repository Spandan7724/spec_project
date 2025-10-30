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
}

export default function PredictionChart({ data, currency_pair }: PredictionChartProps) {
  const hasConfidenceBounds = data.some(
    (point) => point.confidence_lower !== undefined && point.confidence_upper !== undefined
  );

  const hasActualPrices = data.some((point) => point.actual_price !== undefined);

  return (
    <div className="w-full h-80">
      <h3 className="text-lg font-semibold mb-4">
        Price Predictions {currency_pair && `(${currency_pair})`}
      </h3>
      {data.length === 0 ? (
        <div className="flex items-center justify-center h-64 text-muted-foreground">
          No prediction data available
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              angle={-45}
              textAnchor="end"
              height={80}
              tickFormatter={(value) => {
                const date = new Date(value);
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
              }}
            />
            <YAxis
              label={{ value: 'Price', angle: -90, position: 'insideLeft' }}
              domain={['auto', 'auto']}
            />
            <Tooltip
              labelFormatter={(value) => {
                const date = new Date(value);
                return date.toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                });
              }}
              formatter={(value: any) => Number(value).toFixed(4)}
            />
            <Legend />

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
      )}
    </div>
  );
}
