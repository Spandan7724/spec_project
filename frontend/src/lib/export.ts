import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { converter as json2csv } from 'json-2-csv';
import type { AnalysisResult, Model } from '../types/api';

/**
 * Export analysis results as PDF
 */
export async function exportAnalysisPDF(result: AnalysisResult): Promise<void> {
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  let yPos = 20;

  // Title
  doc.setFontSize(20);
  doc.text('Currency Analysis Report', pageWidth / 2, yPos, { align: 'center' });
  yPos += 15;

  // Metadata
  doc.setFontSize(10);
  doc.text(`Correlation ID: ${result.correlation_id}`, 14, yPos);
  yPos += 6;
  doc.text(`Date: ${new Date(result.created_at || Date.now()).toLocaleString()}`, 14, yPos);
  yPos += 10;

  // Recommendation section
  doc.setFontSize(14);
  doc.setFont(undefined, 'bold');
  doc.text('Recommendation', 14, yPos);
  yPos += 8;

  doc.setFontSize(11);
  doc.setFont(undefined, 'normal');
  doc.text(`Action: ${result.action}`, 14, yPos);
  yPos += 6;
  doc.text(`Confidence: ${(result.confidence * 100).toFixed(1)}%`, 14, yPos);
  yPos += 6;
  doc.text(`Timeline: ${result.timeline}`, 14, yPos);
  yPos += 10;

  // Rationale
  if (result.rationale && result.rationale.length > 0) {
    doc.setFontSize(12);
    doc.setFont(undefined, 'bold');
    doc.text('Rationale', 14, yPos);
    yPos += 7;

    doc.setFontSize(10);
    doc.setFont(undefined, 'normal');
    result.rationale.forEach((item, index) => {
      const lines = doc.splitTextToSize(`${index + 1}. ${item}`, pageWidth - 28);
      doc.text(lines, 14, yPos);
      yPos += lines.length * 5;
      if (yPos > 270) {
        doc.addPage();
        yPos = 20;
      }
    });
    yPos += 5;
  }

  // Warnings
  if (result.warnings && result.warnings.length > 0) {
    doc.setFontSize(12);
    doc.setFont(undefined, 'bold');
    doc.text('Warnings', 14, yPos);
    yPos += 7;

    doc.setFontSize(10);
    doc.setFont(undefined, 'normal');
    result.warnings.forEach((warning, index) => {
      const lines = doc.splitTextToSize(`${index + 1}. ${warning}`, pageWidth - 28);
      doc.text(lines, 14, yPos);
      yPos += lines.length * 5;
      if (yPos > 270) {
        doc.addPage();
        yPos = 20;
      }
    });
    yPos += 5;
  }

  // Risk Summary Table
  if (result.risk_summary) {
    if (yPos > 200) {
      doc.addPage();
      yPos = 20;
    }

    doc.setFontSize(12);
    doc.setFont(undefined, 'bold');
    doc.text('Risk Summary', 14, yPos);
    yPos += 7;

    const riskData = Object.entries(result.risk_summary).map(([key, value]) => [
      key.replace(/_/g, ' ').toUpperCase(),
      typeof value === 'object' ? JSON.stringify(value) : String(value),
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [['Category', 'Value']],
      body: riskData,
      theme: 'grid',
    });

    yPos = (doc as any).lastAutoTable.finalY + 10;
  }

  // Cost Estimate Table
  if (result.cost_estimate) {
    if (yPos > 200) {
      doc.addPage();
      yPos = 20;
    }

    doc.setFontSize(12);
    doc.setFont(undefined, 'bold');
    doc.text('Cost Estimate', 14, yPos);
    yPos += 7;

    const costData = Object.entries(result.cost_estimate).map(([key, value]) => [
      key.replace(/_/g, ' ').toUpperCase(),
      typeof value === 'object' ? JSON.stringify(value) : String(value),
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [['Category', 'Value']],
      body: costData,
      theme: 'grid',
    });
  }

  // Save PDF
  doc.save(`analysis-${result.correlation_id}.pdf`);
}

/**
 * Export analysis results as CSV
 */
export async function exportAnalysisCSV(result: AnalysisResult): Promise<void> {
  // Flatten the result object for CSV export
  const flatData = {
    correlation_id: result.correlation_id,
    action: result.action,
    confidence: result.confidence,
    timeline: result.timeline,
    status: result.status,
    created_at: result.created_at,
    rationale: result.rationale?.join(' | '),
    warnings: result.warnings?.join(' | '),
    utility_scores: JSON.stringify(result.utility_scores),
    component_confidences: JSON.stringify(result.component_confidences),
    risk_summary: JSON.stringify(result.risk_summary),
    cost_estimate: JSON.stringify(result.cost_estimate),
  };

  try {
    const csv = await json2csv([flatData]);
    downloadCSV(csv, `analysis-${result.correlation_id}.csv`);
  } catch (error) {
    console.error('Error exporting to CSV:', error);
    throw error;
  }
}

/**
 * Export model metrics as CSV
 */
export async function exportModelMetricsCSV(models: Model[]): Promise<void> {
  if (models.length === 0) {
    throw new Error('No models to export');
  }

  const modelData = models.map((model) => ({
    model_id: model.model_id,
    model_type: model.model_type,
    currency_pair: model.currency_pair,
    version: model.version,
    trained_at: model.trained_at,
    horizons: model.horizons.join(', '),
    calibration_ok: model.calibration_ok,
    min_samples: model.min_samples,
  }));

  try {
    const csv = await json2csv(modelData);
    downloadCSV(csv, `model-metrics-${Date.now()}.csv`);
  } catch (error) {
    console.error('Error exporting models to CSV:', error);
    throw error;
  }
}

/**
 * Helper function to download CSV string as file
 */
function downloadCSV(csvContent: string, filename: string): void {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);

  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

/**
 * Export data as JSON
 */
export function exportJSON(data: any, filename: string): void {
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);

  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}
