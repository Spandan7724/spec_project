import { Calendar, AlertTriangle, Info, CheckCircle } from 'lucide-react';

interface Event {
  title: string;
  date?: string;
  category?: string;
  impact?: 'high' | 'medium' | 'low';
  description?: string;
  countries?: string[];
}

interface EventsTabProps {
  data: Event[];
}

const ImpactBadge = ({ impact }: { impact?: string }) => {
  const config = {
    high: { color: 'bg-red-100 text-red-700', icon: AlertTriangle },
    medium: { color: 'bg-yellow-100 text-yellow-700', icon: Info },
    low: { color: 'bg-green-100 text-green-700', icon: CheckCircle },
  };

  const { color, icon: Icon } = config[impact as keyof typeof config] || config.medium;

  return (
    <span className={`px-2 py-1 text-xs rounded flex items-center gap-1 ${color}`}>
      <Icon size={12} />
      {impact ? impact.charAt(0).toUpperCase() + impact.slice(1) : 'Medium'} Impact
    </span>
  );
};

export default function EventsTab({ data }: EventsTabProps) {
  if (!data || data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
        <Calendar size={48} className="mb-3 opacity-50" />
        <p>No events available</p>
      </div>
    );
  }

  return (
    <div className="space-y-3 md:space-y-4">
      {data.map((event, index) => (
        <div key={index} className="border rounded-lg p-4 md:p-5 hover:bg-accent/50 transition-colors">
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-3 mb-2">
            <h4 className="font-semibold flex-1 text-sm md:text-base line-clamp-2">{event.title}</h4>
            <ImpactBadge impact={event.impact} />
          </div>

          {event.description && (
            <p className="text-xs md:text-sm text-muted-foreground mb-3 line-clamp-2">{event.description}</p>
          )}

          <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4 text-xs text-muted-foreground flex-wrap">
            {event.date && (
              <span className="flex items-center gap-1">
                <Calendar size={12} />
                {new Date(event.date).toLocaleDateString('en-US', {
                  month: 'long',
                  day: 'numeric',
                  year: 'numeric',
                  hour: event.date.includes('T') ? '2-digit' : undefined,
                  minute: event.date.includes('T') ? '2-digit' : undefined,
                })}
              </span>
            )}
            {event.category && (
              <span className="px-2 py-0.5 rounded bg-primary/10 text-primary text-xs">
                {event.category}
              </span>
            )}
            {event.countries && event.countries.length > 0 && (
              <span className="text-xs">
                Countries: {event.countries.join(', ')}
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
