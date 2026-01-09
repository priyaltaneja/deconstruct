/**
 * Job Monitoring Dashboard
 * Minimalistic batch overview with status and JSON output
 */

import React from 'react';
import { useExtractionStore } from '../store/extractionStore';
import { formatCost, formatDuration } from '../schemas/validation';

export const JobMonitor: React.FC = () => {
  const { batches, activeBatchId, getBatchStats, getJobsByBatch } = useExtractionStore();

  const activeBatch = activeBatchId ? getBatchStats(activeBatchId) : null;
  const jobs = activeBatchId ? getJobsByBatch(activeBatchId) : [];

  if (!activeBatch) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
        No batch selected.
      </div>
    );
  }

  const batchData = batches.get(activeBatchId!);
  const createdAt = batchData ? new Date(batchData.createdAt).toLocaleString() : '';

  // Determine overall batch status
  const getOverallStatus = () => {
    if (jobs.length === 0) return 'queued';
    if (jobs.some(j => j.status === 'processing' || j.status === 'verifying')) return 'processing';
    if (jobs.every(j => j.status === 'completed')) return 'completed';
    if (jobs.some(j => j.status === 'failed')) return 'failed';
    if (jobs.every(j => j.status === 'queued')) return 'queued';
    return 'processing';
  };

  const status = getOverallStatus();

  const getStatusStyle = (s: string) => {
    switch (s) {
      case 'completed':
        return { color: '#22c55e', bg: 'rgba(34, 197, 94, 0.1)', border: 'rgba(34, 197, 94, 0.3)' };
      case 'processing':
      case 'verifying':
        return { color: '#3b82f6', bg: 'rgba(59, 130, 246, 0.1)', border: 'rgba(59, 130, 246, 0.3)' };
      case 'failed':
        return { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', border: 'rgba(239, 68, 68, 0.3)' };
      default:
        return { color: '#6b7280', bg: 'rgba(107, 114, 128, 0.1)', border: 'rgba(107, 114, 128, 0.3)' };
    }
  };

  const statusStyle = getStatusStyle(status);

  // Collect all extracted data from completed jobs
  const extractedResults = jobs
    .filter(j => j.status === 'completed' && j.extractedData)
    .map(j => ({
      fileName: j.fileName,
      data: j.extractedData,
    }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Batch Info Table */}
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: '12px',
        padding: '24px',
        border: '1px solid var(--border-color)',
      }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
              <th style={thStyle}>Batch</th>
              <th style={thStyle}>Time</th>
              <th style={thStyle}>Total Time</th>
              <th style={thStyle}>Total Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={tdStyle}>{activeBatch.name}</td>
              <td style={tdStyle}>{createdAt}</td>
              <td style={tdStyle}>{formatDuration(activeBatch.totalProcessingTimeMs)}</td>
              <td style={tdStyle}>{formatCost(activeBatch.totalCostUsd)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Status */}
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: '12px',
        padding: '16px 24px',
        border: '1px solid var(--border-color)',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
      }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.9em' }}>Status:</span>
        <span style={{
          padding: '6px 14px',
          borderRadius: '6px',
          fontSize: '0.85em',
          fontWeight: 600,
          textTransform: 'uppercase',
          backgroundColor: statusStyle.bg,
          color: statusStyle.color,
          border: `1px solid ${statusStyle.border}`,
        }}>
          {status}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.85em', marginLeft: 'auto' }}>
          {activeBatch.completedJobs}/{activeBatch.totalJobs} jobs
        </span>
      </div>

      {/* Extracted JSON */}
      {extractedResults.length > 0 && (
        <div style={{
          backgroundColor: 'var(--bg-secondary)',
          borderRadius: '12px',
          padding: '24px',
          border: '1px solid var(--border-color)',
        }}>
          <div style={{
            color: 'var(--text-muted)',
            fontSize: '0.9em',
            marginBottom: '12px',
            fontWeight: 500,
          }}>
            Extracted JSON
          </div>
          <div style={{
            backgroundColor: '#0a0e1a',
            borderRadius: '8px',
            padding: '16px',
            maxHeight: '500px',
            overflowY: 'auto',
            border: '1px solid var(--border-color)',
          }}>
            <pre style={{
              margin: 0,
              fontSize: '0.8em',
              color: '#e5e7eb',
              lineHeight: '1.6',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontFamily: 'monospace',
            }}>
              {JSON.stringify(
                extractedResults.length === 1
                  ? extractedResults[0].data
                  : extractedResults,
                null,
                2
              )}
            </pre>
          </div>
        </div>
      )}

      {/* Error display for failed jobs */}
      {jobs.some(j => j.status === 'failed') && (
        <div style={{
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderRadius: '12px',
          padding: '16px 24px',
          border: '1px solid rgba(239, 68, 68, 0.3)',
        }}>
          <div style={{ color: '#ef4444', fontWeight: 500, marginBottom: '8px' }}>Errors</div>
          {jobs.filter(j => j.status === 'failed').map(j => (
            <div key={j.id} style={{ color: '#fca5a5', fontSize: '0.85em' }}>
              {j.fileName}: {j.error || 'Unknown error'}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '12px 16px',
  color: 'var(--text-muted)',
  fontWeight: 500,
  fontSize: '0.9em',
};

const tdStyle: React.CSSProperties = {
  padding: '12px 16px',
  color: 'var(--text-primary)',
};
