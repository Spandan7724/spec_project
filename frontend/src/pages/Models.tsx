import { useState } from 'react';
import ModelsList from '../components/models/ModelsList';
import TrainModelForm from '../components/models/TrainModelForm';

export default function Models() {
  const [refreshKey, setRefreshKey] = useState(0);

  const handleTrainingStarted = () => {
    // Refresh the models list
    setRefreshKey((prev) => prev + 1);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Model Training</h1>
        <p className="text-muted-foreground mt-1">
          Train and manage prediction models for currency pairs
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <h2 className="text-xl font-bold mb-4">Trained Models</h2>
          <ModelsList key={refreshKey} />
        </div>

        <div>
          <TrainModelForm onTrainingStarted={handleTrainingStarted} />
        </div>
      </div>
    </div>
  );
}
