import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface ConfidenceData {
  component_name: string;
  confidence: number;
}

interface ConfidenceChartProps {
  data: ConfidenceData[];
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

export default function ConfidenceChart({ data }: ConfidenceChartProps) {
  // Transform data for Recharts
  const chartData = data.map((item) => ({
    name: item.component_name,
    value: Math.round(item.confidence * 100), // Convert to percentage
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
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {chartData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => `${value}%`} />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
