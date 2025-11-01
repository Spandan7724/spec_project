import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
} from 'recharts';

interface RegimePoint {
  date: string;
  regime: string;
  close: number;
}

interface MarketRegimeChartProps {
  regime_history: RegimePoint[];
  current_regime: string;
  currency_pair?: string;
  onHoverChange?: (date: string | null) => void;
}

export default function MarketRegimeChart({
  regime_history,
  current_regime,
  currency_pair,
  onHoverChange,
}: MarketRegimeChartProps) {
  const getRegimeColor = (regime: string) => {
    switch ((regime || '').toString().toLowerCase()) {
      case 'trending_up':
      case 'trending':
        return '#00C49F'; // Green for uptrend
      case 'trending_down':
        return '#FF6B6B'; // Red for downtrend
      case 'ranging':
      case 'sideways':
        return '#FFBB28'; // Yellow for ranging
      case 'volatile':
      case 'high_volatility':
        return '#FF8042'; // Orange for volatile
      default:
        return '#8884d8'; // Blue for unknown
    }
  };

  const getRegimeLabel = (regime: string) => {
    const normalized = (regime ?? '').toString();
    const lower = normalized.toLowerCase();

    switch (lower) {
      case 'trending_up':
        return 'Trending Up';
      case 'trending_down':
        return 'Trending Down';
      case 'trending':
        return 'Trending';
      case 'ranging':
      case 'sideways':
        return 'Ranging';
      case 'volatile':
      case 'high_volatility':
        return 'Volatile';
      default:
        return normalized;
    }
  };

  // Normalize regime key
  const normalizeRegime = (r: string) => {
    const s = (r || '').toString().toLowerCase();
    if (s === 'trending' || s === 'trend_up') return 'trending_up';
    if (s === 'sideways') return 'ranging';
    if (s === 'high_volatility') return 'volatile';
    return s;
  };

  // Create background areas for different regimes by splitting the series
  const chartData = regime_history.map((point) => {
    const key = normalizeRegime(point.regime);
    return {
      ...point,
      regimeKey: key,
      regimeColor: getRegimeColor(key),
      regimeLabel: getRegimeLabel(key),
      // These area series were an attempt to shade under the price; we will prefer full-height bands via ReferenceArea below.
      area_trending_up: null,
      area_trending_down: null,
      area_ranging: null,
      area_volatile: null,
    };
  });

  // Build contiguous regime bands for background shading (full height)
  const segments = (() => {
    if (!regime_history || regime_history.length === 0) return [] as Array<{start: string; end: string; key: string}>;
    const items = regime_history.map((p) => ({ date: p.date, key: normalizeRegime(p.regime) }));
    const segs: Array<{start: string; end: string; key: string}> = [];
    let start = items[0].date;
    let key = items[0].key;
    for (let i = 1; i < items.length; i++) {
      const it = items[i];
      if (it.key !== key) {
        const prev = items[i - 1];
        segs.push({ start, end: prev.date, key });
        start = it.date;
        key = it.key;
      }
    }
    // Close last segment
    segs.push({ start, end: items[items.length - 1].date, key });
    return segs;
  })();

  if (!regime_history || regime_history.length === 0) {
    return (
      <div className="w-full">
        <h3 className="text-base md:text-lg font-semibold mb-4">
          Market Regime {currency_pair && `(${currency_pair})`}
        </h3>
        <div className="flex flex-col items-center justify-center h-48 md:h-64">
          <div className="text-center">
            <p className="text-base font-medium mb-2">Current Regime</p>
            <p
              className="text-xl font-bold"
              style={{ color: getRegimeColor(current_regime) }}
            >
              {getRegimeLabel(current_regime).toUpperCase()}
            </p>
          </div>
          <p className="text-muted-foreground text-sm mt-4">No historical regime data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-base md:text-lg font-semibold">
          Market Regime {currency_pair && `(${currency_pair})`}
        </h3>

        {/* Current Regime Badge */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Current:</span>
          <span
            className="px-3 py-1 rounded font-semibold text-sm"
            style={{
              backgroundColor: getRegimeColor(current_regime) + '20',
              color: getRegimeColor(current_regime),
            }}
          >
            {getRegimeLabel(current_regime).toUpperCase()}
          </span>
        </div>
      </div>

      <div className="scroll-container h-64 md:h-80 -mx-2 px-2">
        <div className="min-w-[600px] h-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartData}
              onMouseMove={(e) => {
                if (e && e.activeLabel && onHoverChange) {
                  onHoverChange(e.activeLabel as string);
                }
              }}
              onMouseLeave={() => onHoverChange && onHoverChange(null)}
            >
              <defs>
                <linearGradient id="trendingUp" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00C49F" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00C49F" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="trendingDown" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#FF6B6B" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#FF6B6B" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="ranging" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#FFBB28" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#FFBB28" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="volatile" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#FF8042" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#FF8042" stopOpacity={0} />
                </linearGradient>
              </defs>
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
                label={{ value: 'Price', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
                domain={['auto', 'auto']}
                tick={{ fontSize: 12 }}
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
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-background border rounded p-3 shadow-lg">
                        <p className="font-semibold text-sm mb-1">{data.regimeLabel}</p>
                        <p className="text-xs text-muted-foreground mb-1">
                          {new Date(data.date).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric',
                          })}
                        </p>
                        <p className="text-sm">
                          Price: <span className="font-medium">{data.close.toFixed(4)}</span>
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />

              {/* Regime background bands across full height */}
              {segments.map((seg, idx) => (
                <ReferenceArea
                  key={`regime-band-${idx}`}
                  x1={seg.start}
                  x2={seg.end}
                  y1="auto"
                  y2="auto"
                  fill={getRegimeColor(seg.key)}
                  fillOpacity={0.18}
                  strokeOpacity={0}
                  ifOverflow="hidden"
                />
              ))}

              {/* Price line */}
              <Line
                type="monotone"
                dataKey="close"
                stroke="#0088FE"
                strokeWidth={2}
                dot={false}
                name="Close Price"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Regime Legend */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="flex items-center gap-2 p-2 border rounded">
          <div
            className="w-4 h-4 rounded"
            style={{ backgroundColor: getRegimeColor('trending_up') }}
          />
          <span className="text-xs font-medium">Trending Up</span>
        </div>
        <div className="flex items-center gap-2 p-2 border rounded">
          <div
            className="w-4 h-4 rounded"
            style={{ backgroundColor: getRegimeColor('trending_down') }}
          />
          <span className="text-xs font-medium">Trending Down</span>
        </div>
        <div className="flex items-center gap-2 p-2 border rounded">
          <div
            className="w-4 h-4 rounded"
            style={{ backgroundColor: getRegimeColor('ranging') }}
          />
          <span className="text-xs font-medium">Ranging</span>
        </div>
        <div className="flex items-center gap-2 p-2 border rounded">
          <div
            className="w-4 h-4 rounded"
            style={{ backgroundColor: getRegimeColor('volatile') }}
          />
          <span className="text-xs font-medium">Volatile</span>
        </div>
      </div>

      {/* Regime Explanation */}
      <div className="mt-4 p-4 bg-muted/40 border border-border rounded">
        <h4 className="text-sm font-semibold mb-2 text-foreground">Market Regime Classification</h4>
        <div className="text-xs text-muted-foreground space-y-1">
          <p>
            <strong>Trending:</strong> Consistent directional movement with clear momentum
          </p>
          <p>
            <strong>Ranging:</strong> Price moving sideways within a defined range, low volatility
          </p>
          <p>
            <strong>Volatile:</strong> High price fluctuations, uncertain direction, increased risk
          </p>
        </div>
      </div>
    </div>
  );
}
