import { Link } from 'react-router-dom';
import type { ChatSession } from '../../contexts/SessionContext';
import { MessageSquare, Clock, ArrowRight } from 'lucide-react';

interface SessionCardProps {
  session: ChatSession;
}

export default function SessionCard({ session }: SessionCardProps) {
  const messageCount = session.messages.length;
  const lastMessage = session.messages[session.messages.length - 1];
  const lastUserMessage = [...session.messages]
    .reverse()
    .find((msg) => msg.role === 'user');

  const formattedDate = new Date(session.updatedAt).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  // Extract currency pair from messages if available
  const currencyPairMatch = session.messages
    .map((msg) => msg.content.match(/([A-Z]{3})\s*\/\s*([A-Z]{3})|([A-Z]{3})\s+to\s+([A-Z]{3})/i))
    .find((match) => match);

  const currencyPair = currencyPairMatch
    ? `${currencyPairMatch[1] || currencyPairMatch[3]}/${currencyPairMatch[2] || currencyPairMatch[4]}`
    : null;

  return (
    <Link
      to={`/chat?session=${session.sessionId}`}
      className="block border rounded-lg p-4 hover:bg-accent/50 transition-colors group"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <MessageSquare size={18} className="text-primary" />
            <h3 className="font-semibold text-sm">
              Session {session.sessionId.slice(-8)}
            </h3>
            {currencyPair && (
              <span className="px-2 py-1 text-xs rounded bg-primary/10 text-primary">
                {currencyPair}
              </span>
            )}
          </div>

          <div className="flex items-center gap-4 text-sm text-muted-foreground mb-3">
            <div className="flex items-center gap-2">
              <Clock size={14} />
              <span>{formattedDate}</span>
            </div>
            <span>{messageCount} message{messageCount !== 1 ? 's' : ''}</span>
          </div>

          {lastUserMessage && (
            <p className="text-sm text-muted-foreground">
              {lastUserMessage.content.substring(0, 120)}
              {lastUserMessage.content.length > 120 && '...'}
            </p>
          )}
        </div>

        <div className="flex items-center text-primary group-hover:translate-x-1 transition-transform">
          <ArrowRight size={20} />
        </div>
      </div>
    </Link>
  );
}
