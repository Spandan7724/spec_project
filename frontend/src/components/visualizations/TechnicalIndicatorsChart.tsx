import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
} from 'recharts';
import { useState } from 'react';

interface TechnicalIndicatorsChartProps {
  data: Array<{ date: string; close: number }>;
  indicators?: {
    rsi?: number;
    macd?: number;
    macd_signal?: number;
    macd_histogram?: number;
    atr?: number;
    volatility_20d?: number;
    volatility_30d?: number;
  };
  currency_pair?: string;
  onHoverChange?: (date: string | null) => void;
}

export default function TechnicalIndicatorsChart({
  data,
  indicators,
  currency_pair,
  onHoverChange,
}: TechnicalIndicatorsChartProps) {
  const [activeIndicator, setActiveIndicator] = useState<'rsi' | 'macd' | 'volatility'>('rsi');

  if (!data || data.length === 0) {
    return (
      <div className="w-full h-96">
        <h3 className="text-lg font-semibold mb-4">
          Technical Indicators {currency_pair && `(${currency_pair})`}
        </h3>
        <div className="flex items-center justify-center h-80 text-muted-foreground">
          No technical indicator data available
        </div>
      </div>
    );
  }

  // Prepare chart data with indicators
  const chartData = data.map((point) => ({
    date: point.date,
    close: point.close,
    rsi: indicators?.rsi,
    macd: indicators?.macd,
    macd_signal: indicators?.macd_signal,
    macd_histogram: indicators?.macd_histogram,
    atr: indicators?.atr,
    volatility_20d: indicators?.volatility_20d,
    volatility_30d: indicators?.volatility_30d,
  }));

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">
          Technical Indicators {currency_pair && `(${currency_pair})`}
        </h3>

        {/* Indicator Selector */}
        <div className="flex gap-2">
          <button
            onClick={() => setActiveIndicator('rsi')}
            className={`px-3 py-1 rounded text-sm ${
              activeIndicator === 'rsi'
                ? 'bg-primary text-primary-foreground'
                : 'bg-secondary text-secondary-foreground hover:bg-accent'
            }`}
          >
            RSI
          </button>
          <button
            onClick={() => setActiveIndicator('macd')}
            className={`px-3 py-1 rounded text-sm ${
              activeIndicator === 'macd'
                ? 'bg-primary text-primary-foreground'
                : 'bg-secondary text-secondary-foreground hover:bg-accent'
            }`}
          >
            MACD
          </button>
          <button
            onClick={() => setActiveIndicator('volatility')}
            className={`px-3 py-1 rounded text-sm ${
              activeIndicator === 'volatility'
                ? 'bg-primary text-primary-foreground'
                : 'bg-secondary text-secondary-foreground hover:bg-accent'
            }`}
          >
            Volatility
          </button>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        {activeIndicator === 'rsi' ? (
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
              dataKey="date"
              angle={-45}
              textAnchor="end"
              height={80}
              tickFormatter={(value) => {
                const date = new Date(value);
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
              }}
            />
            <YAxis domain={[0, 100]} label={{ value: 'RSI', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              labelFormatter={(value) => {
                const date = new Date(value);
                return date.toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                });
              }}
              formatter={(value: any) => Number(value).toFixed(2)}
            />
            <Legend />

            {/* RSI Zones */}
            <ReferenceLine
              y={70}
              stroke="#FF6B6B"
              strokeDasharray="3 3"
              label={{ value: 'Overbought (70)', fill: '#FF6B6B', fontSize: 12 }}
            />
            <ReferenceLine
              y={30}
              stroke="#4ECDC4"
              strokeDasharray="3 3"
              label={{ value: 'Oversold (30)', fill: '#4ECDC4', fontSize: 12 }}
            />
            <ReferenceLine y={50} stroke="#888" strokeDasharray="2 2" />

            {/* RSI Line */}
            <Line
              type="monotone"
              dataKey="rsi"
              stroke="#0088FE"
              strokeWidth={2}
              dot={false}
              name="RSI"
            />

            {/* Color zones */}
            <Area
              type="monotone"
              dataKey="rsi"
              stroke="none"
              fill="#FF6B6B"
              fillOpacity={0.1}
              name="Overbought Zone"
            />
          </ComposedChart>
        ) : activeIndicator === 'macd' ? (
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
              label={{ value: 'MACD', angle: -90, position: 'insideLeft' }}
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

            <ReferenceLine y={0} stroke="#888" strokeDasharray="2 2" />

            {/* MACD Histogram */}
            <Bar dataKey="macd_histogram" fill="#8884d8" name="MACD Histogram" />

            {/* MACD Lines */}
            <Line
              type="monotone"
              dataKey="macd"
              stroke="#0088FE"
              strokeWidth={2}
              dot={false}
              name="MACD"
            />
            <Line
              type="monotone"
              dataKey="macd_signal"
              stroke="#FF8042"
              strokeWidth={2}
              dot={false}
              name="Signal Line"
            />
          </ComposedChart>
        ) : (
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
              label={{ value: 'Volatility / ATR', angle: -90, position: 'insideLeft' }}
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

            <Line
              type="monotone"
              dataKey="volatility_20d"
              stroke="#4ECDC4"
              strokeWidth={2}
              dot={false}
              name="Volatility (20d)"
            />
            <Line
              type="monotone"
              dataKey="volatility_30d"
              stroke="#FFE66D"
              strokeWidth={2}
              dot={false}
              name="Volatility (30d)"
            />
            <Line
              type="monotone"
              dataKey="atr"
              stroke="#FF6B6B"
              strokeWidth={2}
              dot={false}
              name="ATR"
            />
          </ComposedChart>
        )}
      </ResponsiveContainer>

      {/* Indicator Explanation */}
      <div className="mt-4 text-xs text-muted-foreground">
        {activeIndicator === 'rsi' && (
          <p>
            <strong>RSI (Relative Strength Index):</strong> Momentum oscillator measuring speed and
            magnitude of price changes. Values above 70 indicate overbought conditions, below 30
            indicate oversold.
          </p>
        )}
        {activeIndicator === 'macd' && (
          <p>
            <strong>MACD (Moving Average Convergence Divergence):</strong> Trend-following momentum
            indicator. When MACD crosses above signal line, it's a bullish signal; below is bearish.
          </p>
        )}
        {activeIndicator === 'volatility' && (
          <p>
            <strong>Volatility & ATR:</strong> Measures market turbulence and price movement range.
            Higher values indicate increased market uncertainty and risk.
          </p>
        )}
      </div>
    </div>
  );
}
