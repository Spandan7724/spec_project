import { useState } from 'react';
import { modelsService } from '../../services/models';
import type { TrainModelRequest } from '../../services/models';
import { Loader2, CheckCircle2 } from 'lucide-react';
import { toast } from 'sonner';

export default function TrainModelForm({ onTrainingStarted }: { onTrainingStarted: () => void }) {
  type ModelType = 'lightgbm' | 'lstm' | 'catboost';
  interface TrainingFormState {
    currencyPair: string;
    modelType: ModelType;
    horizons: string;
    version: string;
    historyDays: string;
    lstmInterval: string;
    catboostRounds: string;
    catboostLearningRate: string;
    catboostDepth: string;
    catboostPatience: string;
    catboostTaskType: 'auto' | 'cpu' | 'gpu';
  }

  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [formData, setFormData] = useState<TrainingFormState>({
    currencyPair: 'USD/EUR',
    modelType: 'lightgbm',
    horizons: '1,7,30',
    version: '1.0',
    historyDays: '365',
    lstmInterval: '1h',
    catboostRounds: '4000',
    catboostLearningRate: '0.005',
    catboostDepth: '6',
    catboostPatience: '300',
    catboostTaskType: 'auto',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsTraining(true);
    setProgress(0);
    setStatusMessage('Starting training...');

    try {
      const horizons = formData.horizons
        .split(',')
        .map((h) => parseInt(h.trim(), 10))
        .filter((value) => !Number.isNaN(value));
      const historyDays = parseInt(formData.historyDays.trim(), 10);

      const request: TrainModelRequest = {
        currency_pair: formData.currencyPair,
        model_type: formData.modelType,
        horizons: horizons.length ? horizons : undefined,
        version: formData.version,
        history_days: Number.isNaN(historyDays) ? undefined : historyDays,
      };

      if (formData.modelType === 'lstm') {
        request.lstm_interval = formData.lstmInterval;
      }

      if (formData.modelType === 'catboost') {
        const rounds = parseInt(formData.catboostRounds.trim(), 10);
        const patience = parseInt(formData.catboostPatience.trim(), 10);
        const depth = parseInt(formData.catboostDepth.trim(), 10);
        const learningRate = parseFloat(formData.catboostLearningRate.trim());

        if (!Number.isNaN(rounds)) request.catboost_rounds = rounds;
        if (!Number.isNaN(patience)) request.catboost_patience = patience;
        if (!Number.isNaN(depth)) request.catboost_depth = depth;
        if (!Number.isNaN(learningRate)) request.catboost_learning_rate = learningRate;
        request.catboost_task_type = formData.catboostTaskType;
      }

      const response = await modelsService.trainModel(request);

      pollTrainingStatus(response.job_id);
    } catch (error) {
      console.error('Error starting training:', error);
      toast.error('Failed to start training');
      setIsTraining(false);
    }
  };

  const pollTrainingStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const status = await modelsService.getTrainingStatus(jobId);
        setProgress(status.progress);
        setStatusMessage(status.message);

        if (status.status === 'completed') {
          clearInterval(interval);
          setIsTraining(false);
          toast.success('Model training completed successfully');
          onTrainingStarted();
        } else if (status.status === 'error') {
          clearInterval(interval);
          setIsTraining(false);
          toast.error(status.error ? `Training failed: ${status.error}` : 'Training failed');
        }
      } catch (error) {
        console.error('Error polling status:', error);
        clearInterval(interval);
        setIsTraining(false);
        toast.error('Lost connection while monitoring training status');
      }
    }, 2000);
  };

  const handleChange = <K extends keyof TrainingFormState>(field: K, value: TrainingFormState[K]) => {
    setFormData((prev) => {
      if (field === 'modelType') {
        const nextModel = value as ModelType;
        return {
          ...prev,
          modelType: nextModel,
          horizons: nextModel === 'lstm' ? '1,4,24' : '1,7,30',
          historyDays: nextModel === 'lstm' ? '180' : '365',
          lstmInterval: nextModel === 'lstm' ? prev.lstmInterval || '1h' : prev.lstmInterval,
        };
      }

      return { ...prev, [field]: value };
    });
  };

  const modelDescriptions: Record<ModelType, string> = {
    lightgbm: 'Gradient boosting for daily price predictions',
    lstm: 'Neural network for hourly price predictions',
    catboost: 'CatBoost gradient boosting with quantiles + direction signals',
  };

  return (
    <div className="border rounded-lg p-4 md:p-6">
      <h2 className="text-lg md:text-xl font-bold mb-4 md:mb-6">Train New Model</h2>

      <form onSubmit={handleSubmit} className="space-y-4 md:space-y-6">
        <div>
          <label className="block text-xs md:text-sm font-medium mb-2">Currency Pair</label>
          <input
            type="text"
            value={formData.currencyPair}
            onChange={(e) => handleChange('currencyPair', e.target.value)}
            placeholder="e.g., USD/EUR"
            required
            disabled={isTraining}
            className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
          />
        </div>

        <div>
          <label className="block text-xs md:text-sm font-medium mb-2">Model Type</label>
          <select
            value={formData.modelType}
            onChange={(e) => handleChange('modelType', e.target.value as TrainingFormState['modelType'])}
            disabled={isTraining}
            className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
          >
            <option value="lightgbm">LightGBM (daily predictions)</option>
            <option value="lstm">LSTM (intraday predictions)</option>
            <option value="catboost">CatBoost (daily, quantiles + direction)</option>
          </select>
          <p className="text-xs text-muted-foreground mt-1">{modelDescriptions[formData.modelType]}</p>
        </div>

        <div>
          <label className="block text-xs md:text-sm font-medium mb-2">History Window (days)</label>
          {(() => {
            // Mirror backend caps: unlimited for daily; interval-specific caps for intraday
            const intradayCaps: Record<string, number> = {
              '4h': 730,
              '1h': 730,
              '30m': 60,
              '15m': 60,
            };
            const isLstm = formData.modelType === 'lstm';
            const maxDays = isLstm ? intradayCaps[formData.lstmInterval] ?? 730 : undefined;
            const historyPlaceholder = isLstm ? '180' : '365';
            const historyHelpText = isLstm
              ? `${formData.lstmInterval} up to ${maxDays} days.`
              : formData.modelType === 'catboost'
                ? 'Daily CatBoost: fetch as much history as available (365+ recommended).'
                : 'No hard max; backend fetches as many daily years as available.';

            return (
              <>
                <input
                  type="number"
                  min={60}
                  // Daily (LightGBM): no max. LSTM: cap based on interval like backend.
                  max={maxDays}
                  step={10}
                  value={formData.historyDays}
                  onChange={(e) => handleChange('historyDays', e.target.value)}
                  placeholder={historyPlaceholder}
                  required
                  disabled={isTraining}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
                />
                <p className="text-xs text-muted-foreground mt-1">{historyHelpText}</p>
              </>
            );
          })()}
        </div>

        <div>
          <label className="block text-xs md:text-sm font-medium mb-2">
            Horizons ({formData.modelType === 'lstm' ? 'hours' : 'days'})
          </label>
          <input
            type="text"
            value={formData.horizons}
            onChange={(e) => handleChange('horizons', e.target.value)}
            placeholder="e.g., 1,7,30"
            required
            disabled={isTraining}
            className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
          />
          <p className="text-xs text-muted-foreground mt-1">Comma-separated values</p>
        </div>

        {formData.modelType === 'catboost' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 md:gap-4">
            <div>
              <label className="block text-xs md:text-sm font-medium mb-2">Boosting Rounds</label>
              <input
                type="number"
                min={500}
                step={100}
                value={formData.catboostRounds}
                onChange={(e) => handleChange('catboostRounds', e.target.value)}
                disabled={isTraining}
                className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
              />
              <p className="text-xs text-muted-foreground mt-1">Default 4000 (best accuracy)</p>
            </div>

            <div>
              <label className="block text-xs md:text-sm font-medium mb-2">Learning Rate</label>
              <input
                type="number"
                step="0.001"
                min="0.001"
                value={formData.catboostLearningRate}
                onChange={(e) => handleChange('catboostLearningRate', e.target.value)}
                disabled={isTraining}
                className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
              />
              <p className="text-xs text-muted-foreground mt-1">Smaller values = more conservative</p>
            </div>

            <div>
              <label className="block text-xs md:text-sm font-medium mb-2">Tree Depth</label>
              <input
                type="number"
                min={4}
                max={10}
                value={formData.catboostDepth}
                onChange={(e) => handleChange('catboostDepth', e.target.value)}
                disabled={isTraining}
                className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
              />
              <p className="text-xs text-muted-foreground mt-1">Default 6 (balanced)</p>
            </div>

            <div>
              <label className="block text-xs md:text-sm font-medium mb-2">Early-Stopping Patience</label>
              <input
                type="number"
                min={50}
                step={10}
                value={formData.catboostPatience}
                onChange={(e) => handleChange('catboostPatience', e.target.value)}
                disabled={isTraining}
                className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
              />
              <p className="text-xs text-muted-foreground mt-1">Default 300 (matches backend)</p>
            </div>

            <div>
              <label className="block text-xs md:text-sm font-medium mb-2">Task Type</label>
              <select
                value={formData.catboostTaskType}
                onChange={(e) =>
                  handleChange('catboostTaskType', e.target.value as TrainingFormState['catboostTaskType'])
                }
                disabled={isTraining}
                className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
              >
                <option value="auto">Auto (detect GPU)</option>
                <option value="gpu">GPU</option>
                <option value="cpu">CPU</option>
              </select>
              <p className="text-xs text-muted-foreground mt-1">Auto tries GPU first, falls back to CPU</p>
            </div>
          </div>
        )}

        {formData.modelType === 'lstm' && (
          <div>
            <label className="block text-xs md:text-sm font-medium mb-2">Intraday Interval</label>
            <select
              value={formData.lstmInterval}
              onChange={(e) => handleChange('lstmInterval', e.target.value)}
              disabled={isTraining}
              className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
            >
              <option value="1h">1 hour</option>
              <option value="4h">4 hours</option>
              <option value="30m">30 minutes</option>
              <option value="15m">15 minutes</option>
            </select>
            <p className="text-xs text-muted-foreground mt-1">
              Shorter intervals capture more granular patterns at the cost of longer training time.
            </p>
          </div>
        )}

        <div>
          <label className="block text-xs md:text-sm font-medium mb-2">Version</label>
          <input
            type="text"
            value={formData.version}
            onChange={(e) => handleChange('version', e.target.value)}
            placeholder="e.g., 1.0"
            disabled={isTraining}
            className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 mobile-tap min-h-[44px] text-sm"
          />
        </div>

        {isTraining && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs md:text-sm">
              <span className="text-muted-foreground">{statusMessage}</span>
              <span className="font-medium">{progress}%</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2">
              <div
                className="bg-primary h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={isTraining}
          className="w-full py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2 mobile-tap min-h-[44px] font-medium text-sm md:text-base"
        >
          {isTraining ? (
            <>
              <Loader2 className="animate-spin" size={20} />
              Training in Progress...
            </>
          ) : (
            <>
              <CheckCircle2 size={20} />
              Start Training
            </>
          )}
        </button>
      </form>
    </div>
  );
}
