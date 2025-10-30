import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface RiskData {
  category: string;
  level: string;
  score: number;
  description?: string;
}

interface RiskChartProps {
  data: RiskData[];
}

const getRiskColor = (level: string) => {
  switch (level.toLowerCase()) {
    case 'low':
      return '#00C49F';
    case 'medium':
      return '#FFBB28';
    case 'high':
      return '#FF8042';
    default:
      return '#8884d8';
  }
};

export default function RiskChart({ data }: RiskChartProps) {
  // Transform data for Recharts
  const chartData = data.map((item) => ({
    category: item.category,
    score: item.score,
    level: item.level,
    fill: getRiskColor(item.level),
  }));

  return (
    <div className="w-full h-80">
      <h3 className="text-lg font-semibold mb-4">Risk Breakdown</h3>
      {chartData.length === 0 ? (
        <div className="flex items-center justify-center h-64 text-muted-foreground">
          No risk data available
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="category"
              angle={-45}
              textAnchor="end"
              height={100}
            />
            <YAxis label={{ value: 'Risk Score', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-background border rounded p-2 shadow-lg">
                      <p className="font-semibold">{data.category}</p>
                      <p className="text-sm">Level: <span className="font-medium">{data.level}</span></p>
                      <p className="text-sm">Score: <span className="font-medium">{data.score}</span></p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend />
            <Bar dataKey="score" name="Risk Score" />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
