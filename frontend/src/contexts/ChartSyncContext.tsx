import { createContext, useContext, useState, ReactNode } from 'react';

interface ChartSyncContextType {
  hoveredDate: string | null;
  setHoveredDate: (date: string | null) => void;
}

const ChartSyncContext = createContext<ChartSyncContextType | undefined>(undefined);

export function ChartSyncProvider({ children }: { children: ReactNode }) {
  const [hoveredDate, setHoveredDate] = useState<string | null>(null);

  return (
    <ChartSyncContext.Provider value={{ hoveredDate, setHoveredDate }}>
      {children}
    </ChartSyncContext.Provider>
  );
}

export function useChartSync() {
  const context = useContext(ChartSyncContext);
  if (context === undefined) {
    throw new Error('useChartSync must be used within a ChartSyncProvider');
  }
  return context;
}
