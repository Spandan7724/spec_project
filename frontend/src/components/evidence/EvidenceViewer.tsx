import { useState } from 'react';
import NewsTab from './NewsTab';
import EventsTab from './EventsTab';
import MarketDataTab from './MarketDataTab';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface EvidenceData {
  news?: any[];
  events?: any[];
  market_data?: any;
}

interface EvidenceViewerProps {
  data: EvidenceData;
}

type TabType = 'news' | 'events' | 'market_data';

export default function EvidenceViewer({ data }: EvidenceViewerProps) {
  const [activeTab, setActiveTab] = useState<TabType>('news');
  const [isExpanded, setIsExpanded] = useState(false);

  const tabs = [
    { id: 'news' as TabType, label: 'News', count: data.news?.length || 0 },
    { id: 'events' as TabType, label: 'Events', count: data.events?.length || 0 },
    { id: 'market_data' as TabType, label: 'Market Data', count: data.market_data ? 1 : 0 },
  ];

  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 bg-accent hover:bg-accent/80 transition-colors flex items-center justify-between text-accent-foreground"
      >
        <h3 className="text-lg font-semibold">Evidence & Supporting Data</h3>
        {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
      </button>

      {isExpanded && (
        <div className="p-4">
          {/* Tabs */}
          <div className="flex gap-2 border-b mb-4">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 transition-colors relative ${
                  activeTab === tab.id
                    ? 'text-primary'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {tab.label}
                {tab.count > 0 && (
                  <span className="ml-2 px-2 py-0.5 text-xs rounded-full bg-primary/10 text-primary">
                    {tab.count}
                  </span>
                )}
                {activeTab === tab.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                )}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="min-h-[200px]">
            {activeTab === 'news' && <NewsTab data={data.news || []} />}
            {activeTab === 'events' && <EventsTab data={data.events || []} />}
            {activeTab === 'market_data' && <MarketDataTab data={data.market_data} />}
          </div>
        </div>
      )}
    </div>
  );
}
