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

interface IndicatorSeries {
  rsi?: Array<{ date: string; value: number }>;
  macd?: Array<{ date: string; macd?: number; macd_signal?: number; macd_histogram?: number }>;
  atr?: Array<{ date: string; atr?: number }>;
  volatility?: Array<{ date: string; volatility_20d?: number; volatility_30d?: number }>;
}

interface IndicatorSnapshot {
  latest?: {
    rsi?: number;
    macd?: number;
    macd_signal?: number;
    macd_histogram?: number;
    atr?: number;
    volatility_30d?: number;
  };
  series?: IndicatorSeries;
}

interface TechnicalIndicatorsChartProps {
  data: Array<{ date: string; close: number; volume?: number | null }>;
  indicators?: IndicatorSnapshot;
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
  const indicatorSeries = indicators?.series ?? {};
  const latest = indicators?.latest;

  if (!data || data.length === 0) {
    return (
      <div className="w-full">
        <h3 className="text-base md:text-lg font-semibold mb-4">
          Technical Indicators {currency_pair && `(${currency_pair})`}
        </h3>
        <div className="flex items-center justify-center h-48 md:h-64 text-muted-foreground text-sm">
          No technical indicator data available
        </div>
      </div>
    );
  }

  // Prepare chart data with indicators
  const rsiData = indicatorSeries.rsi ?? [];
  const macdData = indicatorSeries.macd ?? [];
  const atrData = indicatorSeries.atr ?? [];
  const volatilityData = indicatorSeries.volatility ?? [];

  const atrMap = new Map<string, number | undefined>(
    atrData.map((item) => [item.date, item.atr])
  );

  const volatilityMerged = volatilityData.map((entry) => ({
    ...entry,
    atr: atrMap.get(entry.date),
  }));

  const formatTooltipDate = (value: any) => {
    const date = new Date(value);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  const formatNumber = (value: unknown, digits = 4) => {
    const num = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(num)) return '—';
    return num.toFixed(digits);
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-base md:text-lg font-semibold">
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

      {latest && (
        <div className="grid grid-cols-1 gap-3 mb-4 text-xs text-muted-foreground md:grid-cols-3">
          <div className="p-3 bg-muted/40 border border-border rounded">
            <p className="font-semibold text-foreground">RSI (14)</p>
            <p className="text-sm text-foreground mt-1">{formatNumber(latest.rsi, 2)}</p>
          </div>
          <div className="p-3 bg-muted/40 border border-border rounded">
            <p className="font-semibold text-foreground">MACD</p>
            <p className="text-sm text-foreground mt-1">
              {formatNumber(latest.macd, 4)}
              <span className="ml-2 text-xs text-muted-foreground">Signal: {formatNumber(latest.macd_signal, 4)}</span>
            </p>
          </div>
          <div className="p-3 bg-muted/40 border border-border rounded">
            <p className="font-semibold text-foreground">Volatility (30d)</p>
            <p className="text-sm text-foreground mt-1">{formatNumber(latest.volatility_30d, 4)}</p>
          </div>
        </div>
      )}

      <div className="scroll-container h-64 md:h-80 -mx-2 px-2">
        <div className="min-w-[700px] h-full">
          <ResponsiveContainer width="100%" height="100%">
            {activeIndicator === 'rsi' ? (
              rsiData.length > 0 ? (
              <ComposedChart
                data={rsiData}
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
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                  }}
                />
                <YAxis domain={[0, 100]} label={{ value: 'RSI', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} tick={{ fontSize: 12 }} />
                <Tooltip
                  labelFormatter={formatTooltipDate}
                  formatter={(value: any) => formatNumber(value, 2)}
                  contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', color: '#000' }}
                  labelStyle={{ color: '#000' }}
                  itemStyle={{ color: '#000' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />

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
                  dataKey="value"
                  stroke="#0088FE"
                  strokeWidth={2}
                  dot={false}
                  name="RSI"
                />

                {/* Color zones */}
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="none"
                  fill="#FF6B6B"
                  fillOpacity={0.1}
                  name="Overbought Zone"
                />
              </ComposedChart>
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground text-sm">RSI data unavailable</div>
              )
            ) : activeIndicator === 'macd' ? (
              macdData.length > 0 ? (
              <ComposedChart
                data={macdData}
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
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                  }}
                />
                <YAxis
                  label={{ value: 'MACD', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  labelFormatter={formatTooltipDate}
                  formatter={(value: any) => formatNumber(value, 4)}
                  contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', color: '#000' }}
                  labelStyle={{ color: '#000' }}
                  itemStyle={{ color: '#000' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />

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
                <div className="flex items-center justify-center h-full text-muted-foreground text-sm">MACD data unavailable</div>
              )
            ) : (
              volatilityMerged.length > 0 ? (
              <ComposedChart
                data={volatilityMerged}
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
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                  }}
                />
                <YAxis
                  label={{ value: 'Volatility / ATR', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  labelFormatter={formatTooltipDate}
                  formatter={(value: any, name: string) => `${formatNumber(value, 4)} (${name})`}
                  contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', color: '#000' }}
                  labelStyle={{ color: '#000' }}
                  itemStyle={{ color: '#000' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />

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
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground text-sm">Volatility data unavailable</div>
              )
            )}
          </ResponsiveContainer>
        </div>
      </div>

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
