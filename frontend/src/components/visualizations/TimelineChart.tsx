import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface TimelinePhase {
  phase: string;
  duration_days: number;
  start_day?: number;
  tasks?: string[];
}

interface TimelineChartProps {
  data: TimelinePhase[];
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

export default function TimelineChart({ data }: TimelineChartProps) {
  // Calculate start days if not provided
  let cumulativeDays = 0;
  const chartData = data.map((phase) => {
    const startDay = phase.start_day ?? cumulativeDays;
    cumulativeDays = startDay + phase.duration_days;
    return {
      phase: phase.phase,
      start: startDay,
      duration: phase.duration_days,
      end: startDay + phase.duration_days,
      tasks: phase.tasks,
    };
  });

  return (
    <div className="w-full h-80">
      <h3 className="text-lg font-semibold mb-4">Implementation Timeline</h3>
      {chartData.length === 0 ? (
        <div className="flex items-center justify-center h-64 text-muted-foreground">
          No timeline data available
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" label={{ value: 'Days', position: 'insideBottom', offset: -5 }} />
            <YAxis dataKey="phase" type="category" />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-background border rounded p-3 shadow-lg max-w-xs">
                      <p className="font-semibold mb-1">{data.phase}</p>
                      <p className="text-sm">Duration: <span className="font-medium">{data.duration} days</span></p>
                      <p className="text-sm">Days {data.start} - {data.end}</p>
                      {data.tasks && data.tasks.length > 0 && (
                        <div className="mt-2">
                          <p className="text-xs font-semibold">Tasks:</p>
                          <ul className="text-xs list-disc list-inside">
                            {data.tasks.slice(0, 3).map((task: string, i: number) => (
                              <li key={i}>{task}</li>
                            ))}
                            {data.tasks.length > 3 && (
                              <li className="text-muted-foreground">+{data.tasks.length - 3} more...</li>
                            )}
                          </ul>
                        </div>
                      )}
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="duration" name="Duration (days)">
              {chartData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
