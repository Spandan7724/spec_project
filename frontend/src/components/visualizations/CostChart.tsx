import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface CostItem {
  category: string;
  amount: number;
  currency: string;
  breakdown?: Record<string, number>;
}

interface CostChartProps {
  data: CostItem[];
}

const COLORS = {
  fees: '#0088FE',
  spread: '#00C49F',
  conversion: '#FFBB28',
  other: '#FF8042',
};

export default function CostChart({ data }: CostChartProps) {
  // Transform data for stacked bar chart
  const chartData = data.map((item) => {
    const base = {
      category: item.category,
      total: item.amount,
    };
    // Add breakdown if available
    if (item.breakdown) {
      return { ...base, ...item.breakdown };
    }
    return base;
  });

  // Get all unique breakdown keys
  const breakdownKeys = Array.from(
    new Set(
      data.flatMap((item) => (item.breakdown ? Object.keys(item.breakdown) : []))
    )
  );

  return (
    <div className="w-full h-80">
      <h3 className="text-lg font-semibold mb-4">Cost Breakdown</h3>
      {chartData.length === 0 ? (
        <div className="flex items-center justify-center h-64 text-muted-foreground">
          No cost data available
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
            <YAxis label={{ value: 'Amount', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              formatter={(value) => {
                const currency = data[0]?.currency || 'USD';
                return `${currency} ${Number(value).toFixed(2)}`;
              }}
            />
            <Legend />
            {breakdownKeys.length > 0 ? (
              breakdownKeys.map((key, index) => (
                <Bar
                  key={key}
                  dataKey={key}
                  stackId="a"
                  fill={COLORS[key as keyof typeof COLORS] || `hsl(${index * 60}, 70%, 50%)`}
                  name={key.charAt(0).toUpperCase() + key.slice(1)}
                />
              ))
            ) : (
              <Bar dataKey="total" fill="#0088FE" name="Total Cost" />
            )}
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
