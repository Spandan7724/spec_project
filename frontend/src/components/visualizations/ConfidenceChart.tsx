import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface ConfidenceData {
  component_name: string;
  confidence: number;
}

interface ConfidenceChartProps {
  data: ConfidenceData[];
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

export default function ConfidenceChart({ data }: ConfidenceChartProps) {
  const chartData = data.map((item) => ({
    name: item.component_name,
    value: Math.round(item.confidence * 100),
  }));

  return (
    <div className="w-full h-64 md:h-80">
      <h3 className="text-base md:text-lg font-semibold mb-4">Component Confidence Breakdown</h3>
      {chartData.length === 0 ? (
        <div className="flex items-center justify-center h-48 md:h-64 text-muted-foreground text-sm">
          No confidence data available
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
            <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(value) => [`${value}%`, 'Confidence']} />
            <Bar
              dataKey="value"
              radius={[0, 4, 4, 0]}
              label={{ position: 'right', formatter: (value) => `${Number(value)}%` }}
            >
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
