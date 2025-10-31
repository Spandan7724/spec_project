import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  RadialBarChart,
  RadialBar,
  PolarAngleAxis,
} from 'recharts';

interface SentimentPoint {
  date: string;
  sentiment_base: number;
  sentiment_quote: number;
  title: string;
  source: string;
}

interface SentimentChartProps {
  current_sentiment: {
    base_currency: string;
    quote_currency: string;
    sentiment_base: number;
    sentiment_quote: number;
    pair_bias: string;
    narrative: string;
  };
  timeline: SentimentPoint[];
  currency_pair?: string;
  onHoverChange?: (date: string | null) => void;
}

export default function SentimentChart({
  current_sentiment,
  timeline,
  currency_pair,
  onHoverChange,
}: SentimentChartProps) {
  // Convert sentiment scores (-1 to +1) to 0-100 scale for radial chart
  const sentimentToPercentage = (score: number) => ((score + 1) / 2) * 100;

  const getSentimentColor = (score: number) => {
    if (score > 0.2) return '#00C49F'; // Positive (green)
    if (score < -0.2) return '#FF6B6B'; // Negative (red)
    return '#FFBB28'; // Neutral (yellow)
  };

  const gaugeData = [
    {
      name: current_sentiment.base_currency,
      value: sentimentToPercentage(current_sentiment.sentiment_base),
      fill: getSentimentColor(current_sentiment.sentiment_base),
    },
    {
      name: current_sentiment.quote_currency,
      value: sentimentToPercentage(current_sentiment.sentiment_quote),
      fill: getSentimentColor(current_sentiment.sentiment_quote),
    },
  ];

  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold mb-4">
        Market Sentiment {currency_pair && `(${currency_pair})`}
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Current Sentiment Gauge */}
        <div>
          <h4 className="text-sm font-medium mb-2">Current Sentiment</h4>
          <ResponsiveContainer width="100%" height={250}>
            <RadialBarChart
              cx="50%"
              cy="50%"
              innerRadius="30%"
              outerRadius="100%"
              data={gaugeData}
              startAngle={180}
              endAngle={0}
            >
              <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
              <RadialBar
                background
                dataKey="value"
                cornerRadius={10}
                label={{
                  position: 'insideStart',
                  fill: '#fff',
                  formatter: (value: number) => `${(value / 50 - 1).toFixed(2)}`,
                }}
              />
              <Legend
                iconSize={10}
                layout="horizontal"
                verticalAlign="bottom"
                align="center"
                formatter={(value, entry: any) => {
                  const score = (entry.payload.value / 50 - 1).toFixed(2);
                  return `${value}: ${score}`;
                }}
              />
              <Tooltip
                formatter={(value: number) => {
                  const score = (value / 50 - 1).toFixed(2);
                  return [`Sentiment: ${score}`, ''];
                }}
              />
            </RadialBarChart>
          </ResponsiveContainer>

          {/* Bias Indicator */}
          <div className="text-center mt-2">
            <p className="text-sm">
              <strong>Pair Bias:</strong>{' '}
              <span
                className={`font-semibold ${
                  current_sentiment.pair_bias === 'bullish'
                    ? 'text-green-600'
                    : current_sentiment.pair_bias === 'bearish'
                    ? 'text-red-600'
                    : 'text-yellow-600'
                }`}
              >
                {current_sentiment.pair_bias.toUpperCase()}
              </span>
            </p>
          </div>
        </div>

        {/* Narrative */}
        <div className="flex flex-col justify-center">
          <h4 className="text-sm font-medium mb-2">Market Narrative</h4>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {current_sentiment.narrative || 'No narrative available'}
          </p>

          {/* Sentiment Scale Reference */}
          <div className="mt-4 p-3 bg-accent rounded">
            <p className="text-xs font-semibold mb-2">Sentiment Scale:</p>
            <div className="flex items-center justify-between text-xs">
              <span className="text-red-600">-1.0 (Very Negative)</span>
              <span className="text-yellow-600">0.0 (Neutral)</span>
              <span className="text-green-600">+1.0 (Very Positive)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Sentiment Timeline */}
      {timeline && timeline.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-medium mb-2">Sentiment Timeline</h4>
          <ResponsiveContainer width="100%" height={250}>
            <ComposedChart
              data={timeline}
              onMouseMove={(e) => {
                if (e && e.activeLabel && onHoverChange) {
                  onHoverChange(e.activeLabel as string);
                }
              }}
              onMouseLeave={() => onHoverChange && onHoverChange(null)}
            >
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
                domain={[-1, 1]}
                label={{ value: 'Sentiment', angle: -90, position: 'insideLeft' }}
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
                formatter={(value: any, name: any, props: any) => {
                  return [Number(value).toFixed(2), name];
                }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-background border rounded p-2 shadow-lg max-w-xs">
                        <p className="font-semibold text-xs mb-1">{data.title}</p>
                        <p className="text-xs text-muted-foreground mb-1">Source: {data.source}</p>
                        <p className="text-xs">
                          {current_sentiment.base_currency}:{' '}
                          <span
                            className={
                              data.sentiment_base > 0
                                ? 'text-green-600'
                                : data.sentiment_base < 0
                                ? 'text-red-600'
                                : ''
                            }
                          >
                            {data.sentiment_base.toFixed(2)}
                          </span>
                        </p>
                        <p className="text-xs">
                          {current_sentiment.quote_currency}:{' '}
                          <span
                            className={
                              data.sentiment_quote > 0
                                ? 'text-green-600'
                                : data.sentiment_quote < 0
                                ? 'text-red-600'
                                : ''
                            }
                          >
                            {data.sentiment_quote.toFixed(2)}
                          </span>
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend />
              <ReferenceLine y={0} stroke="#888" strokeDasharray="2 2" />
              <Line
                type="monotone"
                dataKey="sentiment_base"
                stroke="#0088FE"
                strokeWidth={2}
                dot={{ r: 4 }}
                name={current_sentiment.base_currency}
              />
              <Line
                type="monotone"
                dataKey="sentiment_quote"
                stroke="#FF8042"
                strokeWidth={2}
                dot={{ r: 4 }}
                name={current_sentiment.quote_currency}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
