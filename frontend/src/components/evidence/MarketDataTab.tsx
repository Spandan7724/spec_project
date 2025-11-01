import { LineChart, TrendingUp, TrendingDown, DollarSign } from 'lucide-react';

interface MarketDataTabProps {
  data: any;
}

export default function MarketDataTab({ data }: MarketDataTabProps) {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
        <LineChart size={48} className="mb-3 opacity-50" />
        <p>No market data available</p>
      </div>
    );
  }

  // Helper function to render different data types
  const renderValue = (value: any): string => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') {
      return value.toFixed(4);
    }
    if (typeof value === 'object') {
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  };

  // Helper to determine if value is positive/negative for styling
  const getValueStyle = (key: string, value: any) => {
    if (typeof value !== 'number') return '';

    const positiveKeys = ['gain', 'increase', 'profit', 'growth'];
    const negativeKeys = ['loss', 'decrease', 'cost', 'decline'];

    const keyLower = key.toLowerCase();
    const isPositive = positiveKeys.some(k => keyLower.includes(k));
    const isNegative = negativeKeys.some(k => keyLower.includes(k));

    if (isPositive && value > 0) return 'text-green-600';
    if (isPositive && value < 0) return 'text-red-600';
    if (isNegative && value > 0) return 'text-red-600';
    if (isNegative && value < 0) return 'text-green-600';
    if (value > 0) return 'text-green-600';
    if (value < 0) return 'text-red-600';

    return '';
  };

  // Recursive function to render nested objects
  const renderData = (obj: any, depth: number = 0): React.ReactNode => {
    if (typeof obj !== 'object' || obj === null) {
      return null;
    }

    return Object.entries(obj).map(([key, value]) => {
      const formattedKey = key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, (char) => char.toUpperCase());

      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        return (
          <div key={key} className={`${depth > 0 ? 'ml-2 md:ml-4' : ''} mb-3`}>
            <h4 className="font-semibold text-xs md:text-sm mb-2">{formattedKey}</h4>
            <div className="border-l-2 border-accent pl-2 md:pl-3">
              {renderData(value, depth + 1)}
            </div>
          </div>
        );
      }

      if (Array.isArray(value)) {
        return (
          <div key={key} className={`${depth > 0 ? 'ml-2 md:ml-4' : ''} mb-2`}>
            <span className="text-xs md:text-sm font-medium">{formattedKey}:</span>
            <span className="ml-2 text-xs md:text-sm text-muted-foreground">
              {value.length} items
            </span>
          </div>
        );
      }

      return (
        <div key={key} className={`${depth > 0 ? 'ml-2 md:ml-4' : ''} grid grid-cols-1 sm:grid-cols-2 gap-2 py-2 border-b last:border-0`}>
          <span className="text-xs md:text-sm font-medium flex items-center gap-2">
            {key.toLowerCase().includes('price') && <DollarSign size={14} />}
            {key.toLowerCase().includes('trend') && value > 0 && <TrendingUp size={14} className="text-green-600" />}
            {key.toLowerCase().includes('trend') && value < 0 && <TrendingDown size={14} className="text-red-600" />}
            {formattedKey}
          </span>
          <span className={`text-xs md:text-sm font-mono ${getValueStyle(key, value)}`}>
            {renderValue(value)}
          </span>
        </div>
      );
    });
  };

  return (
    <div className="space-y-3 md:space-y-4">
      <div className="border rounded-lg p-4 md:p-5 bg-accent/20 overflow-x-auto">
        {renderData(data)}
      </div>
    </div>
  );
}
