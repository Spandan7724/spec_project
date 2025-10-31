import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis, Cell } from 'recharts';

interface Event {
  date: string;
  title: string;
  country: string;
  currency: string;
  importance: string;
  impact: string;
  description: string;
}

interface EventsTimelineChartProps {
  events: Event[];
  currency_pair?: string;
  onHoverChange?: (date: string | null) => void;
}

export default function EventsTimelineChart({
  events,
  currency_pair,
  onHoverChange,
}: EventsTimelineChartProps) {
  if (!events || events.length === 0) {
    return (
      <div className="w-full h-96">
        <h3 className="text-lg font-semibold mb-4">
          Economic Events Timeline {currency_pair && `(${currency_pair})`}
        </h3>
        <div className="flex items-center justify-center h-80 text-muted-foreground">
          No upcoming economic events
        </div>
      </div>
    );
  }

  const getImpactColor = (impact: string) => {
    switch (impact.toLowerCase()) {
      case 'high':
        return '#FF6B6B';
      case 'medium':
        return '#FFBB28';
      case 'low':
        return '#00C49F';
      default:
        return '#8884d8';
    }
  };

  const getImpactScore = (impact: string) => {
    switch (impact.toLowerCase()) {
      case 'high':
        return 3;
      case 'medium':
        return 2;
      case 'low':
        return 1;
      default:
        return 1;
    }
  };

  // Format events for scatter plot
  const chartData = events.map((event) => ({
    ...event,
    date: new Date(event.date).getTime(),
    impact_score: getImpactScore(event.impact),
    fill: getImpactColor(event.impact),
  }));

  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold mb-4">
        Economic Events Timeline {currency_pair && `(${currency_pair})`}
      </h3>

      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart
          margin={{ top: 20, right: 20, bottom: 60, left: 20 }}
          onMouseMove={(e: any) => {
            if (e && e.activePayload && e.activePayload[0] && onHoverChange) {
              onHoverChange(new Date(e.activePayload[0].payload.date).toISOString());
            }
          }}
          onMouseLeave={() => onHoverChange && onHoverChange(null)}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="date"
            name="Date"
            domain={['auto', 'auto']}
            tickFormatter={(value) => {
              const date = new Date(value);
              return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis
            type="number"
            dataKey="impact_score"
            name="Impact"
            domain={[0, 4]}
            ticks={[1, 2, 3]}
            tickFormatter={(value) => {
              if (value === 1) return 'Low';
              if (value === 2) return 'Medium';
              if (value === 3) return 'High';
              return '';
            }}
          />
          <ZAxis type="number" dataKey="impact_score" range={[100, 400]} />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const event = payload[0].payload;
                return (
                  <div className="bg-background border rounded p-3 shadow-lg max-w-sm">
                    <p className="font-semibold mb-1">{event.title}</p>
                    <p className="text-xs text-muted-foreground mb-1">
                      {new Date(event.date).toLocaleString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </p>
                    <p className="text-xs mb-1">
                      <strong>Country:</strong> {event.country || 'N/A'} | <strong>Currency:</strong>{' '}
                      {event.currency || 'N/A'}
                    </p>
                    <p className="text-xs mb-1">
                      <strong>Impact:</strong>{' '}
                      <span
                        className="font-medium"
                        style={{ color: getImpactColor(event.impact) }}
                      >
                        {event.impact.toUpperCase()}
                      </span>
                    </p>
                    {event.description && (
                      <p className="text-xs mt-2 text-muted-foreground">{event.description}</p>
                    )}
                  </div>
                );
              }
              return null;
            }}
          />
          <Scatter name="Events" data={chartData} shape="circle">
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>

      {/* Event List Below Chart */}
      <div className="mt-4 space-y-2 max-h-64 overflow-y-auto">
        <h4 className="text-sm font-medium mb-2">Upcoming Events</h4>
        {events.map((event, idx) => (
          <div
            key={idx}
            className="flex items-start gap-3 p-2 rounded border hover:bg-muted/60 transition-colors group"
          >
            <div
              className="w-3 h-3 rounded-full mt-1 flex-shrink-0"
              style={{ backgroundColor: getImpactColor(event.impact) }}
              title={`${event.impact} impact`}
            />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate group-hover:text-foreground">{event.title}</p>
              <p className="text-xs text-muted-foreground group-hover:text-foreground/80">
                {new Date(event.date).toLocaleString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })}
                {event.currency && ` • ${event.currency}`}
              </p>
            </div>
            <span
              className="text-xs font-medium px-2 py-0.5 rounded flex-shrink-0"
              style={{
                backgroundColor: getImpactColor(event.impact) + '20',
                color: getImpactColor(event.impact),
              }}
            >
              {event.impact}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
