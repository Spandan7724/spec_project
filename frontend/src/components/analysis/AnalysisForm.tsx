import { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { analysisService } from '../../services/analysis';
import { Loader2 } from 'lucide-react';
import { useSession } from '../../contexts/SessionContext';

export default function AnalysisForm() {
  const navigate = useNavigate();
  const { analysisHistory } = useSession();

  const recentAnalyses = useMemo(() => analysisHistory.slice(0, 5), [analysisHistory]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formData, setFormData] = useState({
    baseCurrency: 'USD',
    quoteCurrency: 'EUR',
    amount: '',
    riskTolerance: 'moderate',
    urgency: 'normal',
    timeframe: '1_week',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const correlationId = `analysis-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const sessionId = `session-${Date.now()}`;

      const request = {
        session_id: sessionId,
        correlation_id: correlationId,
        base_currency: formData.baseCurrency,
        quote_currency: formData.quoteCurrency,
        currency_pair: `${formData.baseCurrency}/${formData.quoteCurrency}`,
        amount: parseFloat(formData.amount),
        risk_tolerance: formData.riskTolerance,
        urgency: formData.urgency,
        timeframe: formData.timeframe,
      };

      await analysisService.startAnalysis(request);
      navigate(`/results/${correlationId}`);
    } catch (error) {
      console.error('Error starting analysis:', error);
      alert('Failed to start analysis. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div className="max-w-2xl mx-auto space-y-10">
      <h1 className="text-3xl font-bold mb-6">Quick Analysis</h1>

      <form onSubmit={handleSubmit} className="space-y-6 border rounded-lg p-6">
        {/* Currency Pair */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">From Currency</label>
            <select
              value={formData.baseCurrency}
              onChange={(e) => handleChange('baseCurrency', e.target.value)}
              className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary bg-card text-foreground border-input"
            >
              <option value="USD">USD</option>
              <option value="EUR">EUR</option>
              <option value="GBP">GBP</option>
              <option value="JPY">JPY</option>
              <option value="CHF">CHF</option>
              <option value="CAD">CAD</option>
              <option value="AUD">AUD</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">To Currency</label>
            <select
              value={formData.quoteCurrency}
              onChange={(e) => handleChange('quoteCurrency', e.target.value)}
              className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary bg-card text-foreground border-input"
            >
              <option value="EUR">EUR</option>
              <option value="USD">USD</option>
              <option value="GBP">GBP</option>
              <option value="JPY">JPY</option>
              <option value="CHF">CHF</option>
              <option value="CAD">CAD</option>
              <option value="AUD">AUD</option>
            </select>
          </div>
        </div>

        {/* Amount */}
        <div>
          <label className="block text-sm font-medium mb-2">Amount</label>
          <input
            type="number"
            value={formData.amount}
            onChange={(e) => handleChange('amount', e.target.value)}
            placeholder="e.g., 5000"
            required
            min="0"
            step="0.01"
            className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary bg-card text-foreground border-input placeholder:text-muted-foreground"
          />
        </div>

        {/* Risk Tolerance */}
        <div>
          <label className="block text-sm font-medium mb-2">Risk Tolerance</label>
          <select
            value={formData.riskTolerance}
            onChange={(e) => handleChange('riskTolerance', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary bg-card text-foreground border-input"
          >
            <option value="conservative">Conservative</option>
            <option value="moderate">Moderate</option>
            <option value="aggressive">Aggressive</option>
          </select>
          <p className="text-sm text-muted-foreground mt-1">
            {formData.riskTolerance === 'conservative' && 'Minimize risk, prefer stability'}
            {formData.riskTolerance === 'moderate' && 'Balance risk and opportunity'}
            {formData.riskTolerance === 'aggressive' && 'Accept higher risk for better rates'}
          </p>
        </div>

        {/* Urgency */}
        <div>
          <label className="block text-sm font-medium mb-2">Urgency</label>
          <select
            value={formData.urgency}
            onChange={(e) => handleChange('urgency', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary bg-card text-foreground border-input"
          >
            <option value="urgent">Urgent</option>
            <option value="normal">Normal</option>
            <option value="flexible">Flexible</option>
          </select>
          <p className="text-sm text-muted-foreground mt-1">
            {formData.urgency === 'urgent' && 'Need to convert immediately'}
            {formData.urgency === 'normal' && 'Convert within a reasonable timeframe'}
            {formData.urgency === 'flexible' && 'Can wait for optimal rate'}
          </p>
        </div>

        {/* Timeframe */}
        <div>
          <label className="block text-sm font-medium mb-2">Timeframe</label>
          <select
            value={formData.timeframe}
            onChange={(e) => handleChange('timeframe', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary bg-card text-foreground border-input"
          >
            <option value="immediate">Immediate</option>
            <option value="1_day">1 Day</option>
            <option value="1_week">1 Week</option>
            <option value="1_month">1 Month</option>
          </select>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isSubmitting || !formData.amount}
          className="w-full py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="animate-spin" size={20} />
              Starting Analysis...
            </>
          ) : (
            'Start Analysis'
          )}
        </button>
      </form>

      {recentAnalyses.length > 0 && (
        <div className="border rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Recent Analyses</h2>
            <Link to="/history" className="text-sm text-primary hover:underline">
              View all
            </Link>
          </div>
          <ul className="space-y-3">
            {recentAnalyses.map((item) => (
              <li key={item.correlationId} className="flex items-center justify-between gap-4 border rounded-md px-3 py-2">
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">
                    {item.currencyPair || 'Unknown pair'}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {new Date(item.createdAt).toLocaleString()}
                  </p>
                </div>
                <Link
                  to={`/results/${item.correlationId}`}
                  className="text-sm font-medium text-primary hover:underline flex-shrink-0"
                >
                  Open
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
