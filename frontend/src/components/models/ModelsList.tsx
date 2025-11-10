import { useState, useEffect } from 'react';
import { modelsService } from '../../services/models';
import type { Model } from '../../types/api';
import { Trash2, Info, Loader2, FileSpreadsheet, ChevronDown, ChevronUp } from 'lucide-react';
import { exportModelMetricsCSV } from '../../lib/export';
import { toast } from 'sonner';

export default function ModelsList() {
  const [models, setModels] = useState<Model[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState({ pair: '', type: '' });
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

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
          {models.map((model) => {
            const isExpanded = expandedModel === model.model_id;

            return (
              <div key={model.model_id} className="border rounded-lg overflow-hidden hover:shadow-md transition-all">
                <div className="p-4 md:p-5 hover:bg-accent/30 transition-colors">
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
                        onClick={() => setExpandedModel(isExpanded ? null : model.model_id)}
                        className="p-2 hover:bg-accent rounded transition-colors mobile-tap min-h-[44px] flex items-center justify-center"
                        title={isExpanded ? "Hide details" : "View details"}
                      >
                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
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

                {/* Expanded Details Section */}
                {isExpanded && (
                  <div className="border-t bg-muted/30 p-4 animate-in slide-in-from-top duration-200">
                    <div className="space-y-3">
                      {/* Performance Metrics */}
                      {model.validation_metrics && Object.keys(model.validation_metrics).length > 0 && (
                        <div className="space-y-2">
                          <p className="text-xs font-medium text-muted-foreground">Performance Metrics</p>
                          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                            {Object.entries(model.validation_metrics).map(([horizon, metrics]: [string, any]) => {
                              const r2 = metrics.r2 !== undefined ? metrics.r2 * 100 : null;
                              return (
                                <div key={horizon} className="bg-background rounded p-2.5 border text-xs">
                                  <div className="flex items-center justify-between mb-1.5">
                                    <span className="font-medium">{horizon}</span>
                                    {r2 !== null && (
                                      <span className="text-primary font-semibold">{r2.toFixed(1)}% R²</span>
                                    )}
                                  </div>
                                  <div className="space-y-0.5 text-muted-foreground">
                                    {metrics.rmse !== undefined && (
                                      <div className="flex justify-between">
                                        <span>RMSE</span>
                                        <span className="font-mono">{metrics.rmse.toFixed(6)}</span>
                                      </div>
                                    )}
                                    {metrics.mae !== undefined && (
                                      <div className="flex justify-between">
                                        <span>MAE</span>
                                        <span className="font-mono">{metrics.mae.toFixed(6)}</span>
                                      </div>
                                    )}
                                    {metrics.directional_accuracy !== undefined && (
                                      <div className="flex justify-between">
                                        <span>Direction</span>
                                        <span>{(metrics.directional_accuracy * 100).toFixed(1)}%</span>
                                      </div>
                                    )}
                                    {metrics.n_samples !== undefined && (
                                      <div className="flex justify-between pt-0.5 border-t">
                                        <span>Samples</span>
                                        <span className="font-medium">{metrics.n_samples}</span>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Training & Configuration */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                        <div className="bg-background rounded p-2.5 border">
                          <p className="font-medium mb-1.5 text-muted-foreground">Training</p>
                          <div className="space-y-0.5 text-muted-foreground">
                            <div className="flex justify-between">
                              <span>Date</span>
                              <span className="font-medium">{new Date(model.trained_at).toLocaleDateString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Min Test Samples</span>
                              <span className="font-medium">{model.min_samples}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Calibrated</span>
                              <span className={model.calibration_ok ? 'text-green-600' : 'text-amber-600'}>
                                {model.calibration_ok ? 'Yes' : 'No'}
                              </span>
                            </div>
                          </div>
                        </div>

                        <div className="bg-background rounded p-2.5 border">
                          <p className="font-medium mb-1.5 text-muted-foreground">Configuration</p>
                          <div className="space-y-0.5 text-muted-foreground">
                            <div className="flex justify-between">
                              <span>Type</span>
                              <span className="font-medium uppercase">{model.model_type}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Horizons</span>
                              <span className="font-medium">{model.horizons.join(', ')} days</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Features</span>
                              <span className="font-medium">
                                {typeof model.features_used === 'number'
                                  ? model.features_used
                                  : Array.isArray(model.features_used)
                                  ? model.features_used.length
                                  : model.features_used}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Technical Details - Collapsed */}
                      <details className="group text-xs">
                        <summary className="cursor-pointer flex items-center justify-between p-2 hover:bg-accent/50 rounded transition-colors">
                          <span className="font-medium text-muted-foreground">Technical Details</span>
                          <ChevronDown className="group-open:rotate-180 transition-transform" size={14} />
                        </summary>
                        <div className="mt-2 space-y-1.5">
                          <div className="flex items-start gap-2 p-2 bg-muted/50 rounded">
                            <span className="text-muted-foreground shrink-0">ID:</span>
                            <span className="font-mono text-xs break-all">{model.model_id}</span>
                          </div>
                          {model.model_path && (
                            <div className="flex items-start gap-2 p-2 bg-muted/50 rounded">
                              <span className="text-muted-foreground shrink-0">Path:</span>
                              <span className="font-mono text-xs break-all">{model.model_path}</span>
                            </div>
                          )}
                          {Array.isArray(model.features_used) && model.features_used.length > 0 && (
                            <div className="p-2 bg-muted/50 rounded">
                              <p className="text-muted-foreground mb-1">Features ({model.features_used.length}):</p>
                              <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                                {model.features_used.map((feature, idx) => (
                                  <span key={idx} className="px-1.5 py-0.5 bg-primary/10 text-primary rounded text-xs">
                                    {feature}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </details>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
