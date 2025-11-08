import { useState, useEffect } from 'react';
import { modelsService } from '../../services/models';
import type { Model } from '../../types/api';
import { Trash2, Info, Loader2, FileSpreadsheet } from 'lucide-react';
import { exportModelMetricsCSV } from '../../lib/export';
import { toast } from 'sonner';

export default function ModelsList() {
  const [models, setModels] = useState<Model[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState({ pair: '', type: '' });

  useEffect(() => {
    fetchModels();
  }, [filter]);

  const fetchModels = async () => {
    setIsLoading(true);
    try {
      const data = await modelsService.listModels({
        currency_pair: filter.pair || undefined,
        model_type: filter.type || undefined,
      });
      setModels(data.models || []);
    } catch (error) {
      console.error('Error fetching models:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (modelId: string) => {
    if (!confirm('Are you sure you want to delete this model?')) return;

    try {
      await modelsService.deleteModel(modelId);
      fetchModels();
    } catch (error) {
      console.error('Error deleting model:', error);
      alert('Failed to delete model');
    }
  };

  const handleExport = async () => {
    if (models.length === 0) {
      toast.error('No models to export');
      return;
    }

    try {
      await exportModelMetricsCSV(models);
      toast.success('Model metrics exported successfully');
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Failed to export model metrics');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  return (
    <div className="space-y-3 md:space-y-4">
      <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 md:gap-4">
        <input
          type="text"
          placeholder="Filter by currency pair..."
          value={filter.pair}
          onChange={(e) => setFilter({ ...filter, pair: e.target.value })}
          className="flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary mobile-tap min-h-[44px]"
        />
        <select
          value={filter.type}
          onChange={(e) => setFilter({ ...filter, type: e.target.value })}
          className="px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary mobile-tap min-h-[44px]"
        >
          <option value="">All Types</option>
          <option value="lightgbm">LightGBM</option>
          <option value="lstm">LSTM</option>
          <option value="catboost">CatBoost</option>
        </select>
        <button
          onClick={handleExport}
          disabled={models.length === 0}
          className="flex items-center gap-2 px-3 md:px-4 py-2 border rounded-lg hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed mobile-tap min-h-[44px] whitespace-nowrap text-sm md:text-base"
          title="Export models as CSV"
        >
          <FileSpreadsheet size={16} />
          <span className="hidden sm:inline">Export CSV</span>
          <span className="sm:hidden">Export</span>
        </button>
      </div>

      {models.length === 0 ? (
        <div className="text-center py-12 border rounded-lg">
          <p className="text-muted-foreground">No models found</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 md:gap-4">
          {models.map((model) => (
            <div key={model.model_id} className="border rounded-lg p-4 md:p-5 hover:bg-accent/50 transition-colors">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex flex-col sm:flex-row sm:items-center gap-2 mb-3 flex-wrap">
                    <h3 className="font-semibold text-base md:text-lg">{model.currency_pair}</h3>
                    <span className="px-2 py-1 text-xs rounded bg-primary/10 text-primary w-fit">
                      {model.model_type.toUpperCase()}
                    </span>
                    {model.calibration_ok && (
                      <span className="px-2 py-1 text-xs rounded bg-green-100 text-green-700 w-fit">
                        Calibrated
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 md:gap-4 text-xs md:text-sm mb-3">
                    <div>
                      <p className="text-muted-foreground">Trained</p>
                      <p className="font-medium text-xs md:text-sm">
                        {new Date(model.trained_at).toLocaleDateString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Version</p>
                      <p className="font-medium text-xs md:text-sm">{model.version}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Horizons</p>
                      <p className="font-medium text-xs md:text-sm">{model.horizons.join(', ')}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Samples</p>
                      <p className="font-medium text-xs md:text-sm">{model.min_samples}</p>
                    </div>
                  </div>

                  <p className="text-xs text-muted-foreground truncate">{model.model_id}</p>
                </div>

                <div className="flex gap-2 flex-shrink-0">
                  <button
                    onClick={() => alert(`Model details:\n${JSON.stringify(model, null, 2)}`)}
                    className="p-2 hover:bg-accent rounded transition-colors mobile-tap min-h-[44px] flex items-center justify-center"
                    title="View details"
                  >
                    <Info size={16} />
                  </button>
                  <button
                    onClick={() => handleDelete(model.model_id)}
                    className="p-2 hover:bg-destructive hover:text-destructive-foreground rounded transition-colors mobile-tap min-h-[44px] flex items-center justify-center"
                    title="Delete model"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
