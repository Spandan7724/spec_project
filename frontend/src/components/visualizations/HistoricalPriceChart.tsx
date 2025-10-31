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
  Brush,
} from 'recharts';
import { useState } from 'react';

interface PricePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface HistoricalPriceChartProps {
  data: PricePoint[];
  indicators?: {
    sma_5?: number;
    sma_20?: number;
    sma_50?: number;
    ema_12?: number;
    ema_26?: number;
    bb_upper?: number;
    bb_middle?: number;
    bb_lower?: number;
  };
  currency_pair?: string;
  onHoverChange?: (date: string | null) => void;
}

export default function HistoricalPriceChart({
  data,
  indicators,
  currency_pair,
  onHoverChange,
}: HistoricalPriceChartProps) {
  const [showSMA, setShowSMA] = useState({ sma5: true, sma20: true, sma50: false });
  const [showEMA, setShowEMA] = useState({ ema12: false, ema26: false });
  const [showBB, setShowBB] = useState(false);
  const [showVolume, setShowVolume] = useState(true);

  if (!data || data.length === 0) {
    return (
      <div className="w-full h-96">
        <h3 className="text-lg font-semibold mb-4">
          Historical Prices {currency_pair && `(${currency_pair})`}
        </h3>
        <div className="flex items-center justify-center h-80 text-muted-foreground">
          No historical price data available
        </div>
      </div>
    );
  }

  // Merge indicators with price data
  const chartData = data.map((point, idx) => ({
    ...point,
    sma_5: indicators?.sma_5,
    sma_20: indicators?.sma_20,
    sma_50: indicators?.sma_50,
    ema_12: indicators?.ema_12,
    ema_26: indicators?.ema_26,
    bb_upper: indicators?.bb_upper,
    bb_middle: indicators?.bb_middle,
    bb_lower: indicators?.bb_lower,
  }));

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">
          Historical Prices {currency_pair && `(${currency_pair})`}
        </h3>

        {/* Toggle Controls */}
        <div className="flex gap-4 text-xs">
          <div className="flex gap-2">
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={showSMA.sma5}
                onChange={(e) => setShowSMA({ ...showSMA, sma5: e.target.checked })}
                className="rounded"
              />
              <span>SMA(5)</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={showSMA.sma20}
                onChange={(e) => setShowSMA({ ...showSMA, sma20: e.target.checked })}
                className="rounded"
              />
              <span>SMA(20)</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={showSMA.sma50}
                onChange={(e) => setShowSMA({ ...showSMA, sma50: e.target.checked })}
                className="rounded"
              />
              <span>SMA(50)</span>
            </label>
          </div>
          <div className="flex gap-2">
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={showEMA.ema12}
                onChange={(e) => setShowEMA({ ...showEMA, ema12: e.target.checked })}
                className="rounded"
              />
              <span>EMA(12)</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={showEMA.ema26}
                onChange={(e) => setShowEMA({ ...showEMA, ema26: e.target.checked })}
                className="rounded"
              />
              <span>EMA(26)</span>
            </label>
          </div>
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={showBB}
              onChange={(e) => setShowBB(e.target.checked)}
              className="rounded"
            />
            <span>Bollinger Bands</span>
          </label>
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={showVolume}
              onChange={(e) => setShowVolume(e.target.checked)}
              className="rounded"
            />
            <span>Volume</span>
          </label>
        </div>
      </div>

      {/* Price Chart */}
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

          {/* Bollinger Bands */}
          {showBB && indicators?.bb_upper && (
            <>
              <Line
                type="monotone"
                dataKey="bb_upper"
                stroke="#82ca9d"
                strokeWidth={1}
                dot={false}
                name="BB Upper"
                strokeDasharray="3 3"
              />
              <Line
                type="monotone"
                dataKey="bb_middle"
                stroke="#8884d8"
                strokeWidth={1}
                dot={false}
                name="BB Middle"
                strokeDasharray="3 3"
              />
              <Line
                type="monotone"
                dataKey="bb_lower"
                stroke="#82ca9d"
                strokeWidth={1}
                dot={false}
                name="BB Lower"
                strokeDasharray="3 3"
              />
            </>
          )}

          {/* Moving Averages */}
          {showSMA.sma5 && indicators?.sma_5 && (
            <Line
              type="monotone"
              dataKey="sma_5"
              stroke="#FF6B6B"
              strokeWidth={2}
              dot={false}
              name="SMA(5)"
            />
          )}
          {showSMA.sma20 && indicators?.sma_20 && (
            <Line
              type="monotone"
              dataKey="sma_20"
              stroke="#4ECDC4"
              strokeWidth={2}
              dot={false}
              name="SMA(20)"
            />
          )}
          {showSMA.sma50 && indicators?.sma_50 && (
            <Line
              type="monotone"
              dataKey="sma_50"
              stroke="#FFE66D"
              strokeWidth={2}
              dot={false}
              name="SMA(50)"
            />
          )}

          {/* EMAs */}
          {showEMA.ema12 && indicators?.ema_12 && (
            <Line
              type="monotone"
              dataKey="ema_12"
              stroke="#A8E6CF"
              strokeWidth={2}
              dot={false}
              name="EMA(12)"
            />
          )}
          {showEMA.ema26 && indicators?.ema_26 && (
            <Line
              type="monotone"
              dataKey="ema_26"
              stroke="#FFD3B6"
              strokeWidth={2}
              dot={false}
              name="EMA(26)"
            />
          )}

          {/* Price Line */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#0088FE"
            strokeWidth={2}
            dot={false}
            name="Close Price"
          />

          <Brush dataKey="date" height={30} stroke="#8884d8" />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Volume Chart */}
      {showVolume && (
        <ResponsiveContainer width="100%" height={100} className="mt-4">
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              angle={-45}
              textAnchor="end"
              height={60}
              tickFormatter={(value) => {
                const date = new Date(value);
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
              }}
            />
            <YAxis label={{ value: 'Volume', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value: any) => Number(value).toLocaleString()} />
            <Bar dataKey="volume" fill="#8884d8" name="Volume" />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
