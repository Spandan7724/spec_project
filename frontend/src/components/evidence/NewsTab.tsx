import { Newspaper, ExternalLink, TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface NewsArticle {
  title: string;
  source?: string;
  published_at?: string;
  url?: string;
  sentiment?: 'positive' | 'negative' | 'neutral';
  relevance_score?: number;
  summary?: string;
}

interface NewsTabProps {
  data: NewsArticle[];
}

const SentimentIcon = ({ sentiment }: { sentiment?: string }) => {
  switch (sentiment) {
    case 'positive':
      return <TrendingUp size={16} className="text-green-600" />;
    case 'negative':
      return <TrendingDown size={16} className="text-red-600" />;
    default:
      return <Minus size={16} className="text-gray-600" />;
  }
};

export default function NewsTab({ data }: NewsTabProps) {
  if (!data || data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
        <Newspaper size={48} className="mb-3 opacity-50" />
        <p>No news articles available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {data.map((article, index) => (
        <div key={index} className="border rounded-lg p-4 hover:bg-accent/50 transition-colors">
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <SentimentIcon sentiment={article.sentiment} />
                <h4 className="font-semibold">{article.title}</h4>
              </div>

              {article.summary && (
                <p className="text-sm text-muted-foreground mb-3">{article.summary}</p>
              )}

              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                {article.source && (
                  <span className="flex items-center gap-1">
                    <Newspaper size={12} />
                    {article.source}
                  </span>
                )}
                {article.published_at && (
                  <span>
                    {new Date(article.published_at).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric',
                    })}
                  </span>
                )}
                {article.relevance_score !== undefined && (
                  <span className="px-2 py-0.5 rounded bg-primary/10 text-primary">
                    Relevance: {(article.relevance_score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            </div>

            {/^https?:\/\//i.test(article.url ?? '') ? (
              <a
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 hover:bg-accent rounded transition-colors"
                title="Read full article"
              >
                <ExternalLink size={16} />
              </a>
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}
